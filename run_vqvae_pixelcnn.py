import os, math, time, random, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from torchvision.models import inception_v3, Inception_V3_Weights

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
OUT_DIR = Path("./outputs"); OUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 128
NUM_WORKERS = min(4, os.cpu_count() or 1)
EPOCHS_VQVAE = int(os.environ.get("EPOCHS_VQVAE", 50))
EPOCHS_PIXELCNN = int(os.environ.get("EPOCHS_PIXELCNN", 30))

print("Device:", DEVICE)
print(f"Running VQ-VAE (epochs={EPOCHS_VQVAE}) and PixelCNN (epochs={EPOCHS_PIXELCNN})")

tf01 = transforms.ToTensor()
train01 = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf01)
test01 = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf01)
train_loader = DataLoader(train01, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"), drop_last=True)
test_loader = DataLoader(test01, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))

def save_json(obj, name):
    with open(OUT_DIR/name, "w") as f: json.dump(obj, f, indent=2)

def show_grid(x, title="", nrow=10, value_range=(0,1), filename=None):
    x=x.detach().cpu().clamp(*value_range)
    if value_range != (0,1): x=(x-value_range[0])/(value_range[1]-value_range[0])
    grid=utils.make_grid(x, nrow=nrow, padding=2)
    plt.figure(figsize=(10,10)); plt.imshow(grid.permute(1,2,0)); plt.axis("off"); plt.title(title)
    if filename: plt.savefig(OUT_DIR/filename, bbox_inches="tight", dpi=160)
    plt.close()

def psnr_batch(x, recon, data_range=1.0):
    mse=(x-recon).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return (10*torch.log10((data_range**2)/mse)).mean().item()

@torch.no_grad()
def inception_features(images, model, batch=64):
    feats=[]
    for chunk in images.split(batch):
        z=F.interpolate(chunk, size=(299,299), mode="bilinear", align_corners=False)
        z=(z - torch.tensor([0.485,0.456,0.406],device=z.device)[None,:,None,None]) / torch.tensor([0.229,0.224,0.225],device=z.device)[None,:,None,None]
        f=model(z).squeeze(-1).squeeze(-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def calculate_fid(real_images, fake_images, max_images=5000):
    n=min(len(real_images),len(fake_images),max_images)
    real_images=real_images[:n].to(DEVICE); fake_images=fake_images[:n].to(DEVICE)
    net=inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    net.fc=nn.Identity(); net.eval().to(DEVICE)
    fr=inception_features(real_images,net); ff=inception_features(fake_images,net)
    mu1,mu2=fr.mean(0),ff.mean(0)
    c1,c2=np.cov(fr,rowvar=False),np.cov(ff,rowvar=False)
    from scipy.linalg import sqrtm
    covmean=sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean): covmean=covmean.real
    fid=float(np.sum((mu1-mu2)**2)+np.trace(c1+c2-2*covmean))
    return max(0.0,fid)

@torch.no_grad()
def collect_real_images(n=5000):
    out=[]
    for x,_ in test_loader:
        out.append(x)
        if sum(t.size(0) for t in out)>=n: break
    return torch.cat(out)[:n]

def export_metrics(rows, filename):
    import pandas as pd
    df=pd.DataFrame(rows); df.to_csv(OUT_DIR/filename,index=False)
    print(df.to_string())
    return df

# VQ-VAE Model
class VectorQuantizer(nn.Module):
    def __init__(self,K=256,D=64,beta=0.25):
        super().__init__(); self.K=K; self.D=D; self.beta=beta
        self.embedding=nn.Embedding(K,D); self.embedding.weight.data.uniform_(-1/K,1/K)
    def forward(self,z_e):
        z=z_e.permute(0,2,3,1).contiguous()
        flat=z.view(-1,self.D)
        dist=(flat.pow(2).sum(1,keepdim=True)+self.embedding.weight.pow(2).sum(1)
              -2*flat@self.embedding.weight.t())
        indices=dist.argmin(1)
        z_q=self.embedding(indices).view(z.shape)
        codebook=F.mse_loss(z_q,z.detach())
        commitment=F.mse_loss(z_q.detach(),z)
        loss=codebook+self.beta*commitment
        z_st=z+(z_q-z).detach()
        return z_st.permute(0,3,1,2).contiguous(),loss,indices.view(z.shape[0],z.shape[1],z.shape[2])

class VQVAE(nn.Module):
    def __init__(self,K=256,D=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,128,4,2,1),nn.ReLU(True),
            nn.Conv2d(128, D,4,2,1)) # B,D,8,8
        self.quantizer=VectorQuantizer(K,D)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(D,128,4,2,1),nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(True),
            nn.Conv2d(64,3,3,1,1),nn.Sigmoid())
    def encode(self,x): return self.encoder(x)
    def decode_indices(self,idx):
        z_q=self.quantizer.embedding(idx).permute(0,3,1,2).contiguous()
        return self.decoder(z_q)
    def forward(self,x):
        z_e=self.encoder(x)
        z_q,q_loss,idx=self.quantizer(z_e)
        recon=self.decoder(z_q)
        rec_loss=F.mse_loss(recon,x)
        return recon, rec_loss+q_loss, rec_loss, q_loss, idx

def train_vqvae(K, epochs=EPOCHS_VQVAE):
    model=VQVAE(K=K).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    history=[]; start=time.time()
    for ep in range(1,epochs+1):
        model.train(); tot_l=[]; rec_l=[]
        for x,_ in train_loader:
            x=x.to(DEVICE); opt.zero_grad(set_to_none=True)
            r,loss,rec,_,_=model(x); loss.backward(); opt.step()
            tot_l.append(loss.item()); rec_l.append(rec.item())
        model.eval(); v_psnr=[]; usage_counts=torch.zeros(K)
        with torch.no_grad():
            for x,_ in test_loader:
                x=x.to(DEVICE); r,_,_,_,idx=model(x)
                v_psnr.append(psnr_batch(x,r))
                usage_counts += torch.bincount(idx.flatten().cpu(), minlength=K)
        used_codes=int((usage_counts>0).sum().item())
        probs=(usage_counts/usage_counts.sum()).clamp_min(1e-12)
        perplexity=float(torch.exp(-(probs*probs.log()).sum()).item())
        row={"K":K,"epoch":ep,"loss":float(np.mean(tot_l)),
             "recon_loss":float(np.mean(rec_l)),"val_psnr":float(np.mean(v_psnr)),
             "active_codes":used_codes,"utilization_pct":100.0*used_codes/K,
             "perplexity":perplexity,"seconds":time.time()-start}
        history.append(row)
        print(f"VQ-VAE K={K:3d} ep={ep:2d}/{epochs} loss={row['loss']:.4f} "
              f"PSNR={row['val_psnr']:.2f} active={used_codes}/{K} ({row['utilization_pct']:.1f}%) "
              f"perp={perplexity:.1f}", flush=True)
    torch.save(model.state_dict(),OUT_DIR/f"vqvae_{K}.pt")
    save_json(history,f"vqvae_{K}_history.json")
    return model,history

vq_models={}; vq_histories=[]
for K in [512,256,128]:
    m,h=train_vqvae(K)
    vq_models[K]=m; vq_histories.extend(h)

last_vq_rows=[h[-1] for h in [vq_models[k] for k in [512,256,128]] if False]
last_vq_rows=[]
for K in [512,256,128]:
    sub=[r for r in vq_histories if r["K"]==K]
    last_vq_rows.append(sub[-1])
export_metrics(last_vq_rows,"vqvae_codebook_metrics.csv")

x,_=next(iter(test_loader)); x=x[:100].to(DEVICE)
vq_models[256].eval()
with torch.no_grad():
    recon,_,_,_,_=vq_models[256](x)
show_grid(x,"CIFAR-10 originals",10,filename="vqvae_originals_100.png")
show_grid(recon,"VQ-VAE K=256 reconstructions",10,filename="vqvae_reconstructions_100.png")

# PixelCNN
class MaskedConv2d(nn.Conv2d):
    def __init__(self,mask_type,*args,**kwargs):
        super().__init__(*args,**kwargs); self.register_buffer("mask",self.weight.data.clone())
        _,_,h,w=self.weight.size(); self.mask.fill_(1)
        self.mask[:,:,h//2,w//2 + (1 if mask_type=="B" else 0):]=0
        self.mask[:,:,h//2 + 1:,:]=0
    def forward(self,x):
        self.weight.data*=self.mask; return super().forward(x)

class PixelCNN(nn.Module):
    def __init__(self,K,channels=128,layers=6):
        super().__init__()
        self.K=K; self.embedding=nn.Embedding(K,channels)
        net=[MaskedConv2d("A",channels,channels,kernel_size=3,padding=1),nn.ReLU(True)]
        for _ in range(layers-2):
            net += [MaskedConv2d("B",channels,channels,kernel_size=3,padding=1),
                    nn.BatchNorm2d(channels), nn.ReLU(True)]
        net += [MaskedConv2d("B",channels,K,kernel_size=1)]
        self.net=nn.Sequential(*net)
    def forward(self,idx):
        h=self.embedding(idx).permute(0,3,1,2).contiguous()
        return self.net(h)

def train_pixelcnn(K,vq_model,epochs=EPOCHS_PIXELCNN):
    prior=PixelCNN(K=K).to(DEVICE); opt=torch.optim.Adam(prior.parameters(),lr=1e-3)
    vq_model.eval(); start=time.time(); hist=[]
    for ep in range(1,epochs+1):
        prior.train(); losses=[]
        for x,_ in train_loader:
            x=x.to(DEVICE)
            with torch.no_grad(): _,_,_,_,idx=vq_model(x)
            logits=prior(idx)
            loss=F.cross_entropy(logits,idx)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.item())
        row={"K":K,"epoch":ep,"cross_entropy":float(np.mean(losses)),"seconds":time.time()-start}
        hist.append(row)
        print(f"PixelCNN K={K:3d} ep={ep:2d}/{epochs} CE={row['cross_entropy']:.4f}", flush=True)
    torch.save(prior.state_dict(),OUT_DIR/f"pixelcnn_{K}.pt")
    save_json(hist,f"pixelcnn_{K}_history.json")
    return prior,hist

pixel_models={}
for K in [512,256,128]:
    p,h=train_pixelcnn(K,vq_models[K])
    pixel_models[K]=p

@torch.no_grad()
def sample_pixelcnn(prior, vq_model, num_samples=100, shape=(8,8)):
    prior.eval(); vq_model.eval()
    idx=torch.zeros((num_samples,*shape),dtype=torch.long,device=DEVICE)
    for i in range(shape[0]):
        for j in range(shape[1]):
            logits=prior(idx)[:,:,i,j]
            probs=F.softmax(logits,dim=-1)
            idx[:,i,j]=torch.multinomial(probs,num_samples=1).squeeze(-1)
    return vq_model.decode_indices(idx)

samples_pixel=sample_pixelcnn(pixel_models[256],vq_models[256],100)
show_grid(samples_pixel,"PixelCNN prior samples (VQ-VAE K=256)",10,filename="vqvae_pixelcnn_samples_100.png")

real=collect_real_images(1000)
vq_fid_rows=[]
for K in [512,256,128]:
    fake_prior=sample_pixelcnn(pixel_models[K],vq_models[K],1000)
    fid_val=calculate_fid(real,fake_prior,max_images=1000)
    vq_fid_rows.append({"model":f"VQ-VAE K={K} + PixelCNN","FID":fid_val})
    print(f"FID VQ-VAE K={K}:",fid_val, flush=True)
export_metrics(vq_fid_rows,"vqvae_fid.csv")

print("Part A complete! All VQ-VAE and PixelCNN models and metrics saved successfully.", flush=True)
