# %% [markdown]
# # Assignment 2 – Part A: β-VAE, VQ-VAE and PixelCNN
# 
# **Course:** Unsupervised Deep Learning  
# **Dataset:** CIFAR-10  
# **Framework:** PyTorch
# 
# Run on the BITS WILP Prayogshala GPU infrastructure as required by the assignment. The notebook downloads CIFAR-10 and saves checkpoints, plots, generated samples and CSV metrics under `./outputs`.

# %% [markdown]
# ## 0. Environment and reproducibility
# 
# If needed, install dependencies in the lab environment:
# 
# ```bash
# pip install torch torchvision matplotlib numpy scipy pandas pillow
# ```
# 
# Keep the official CIFAR-10 train/test split. Pixel tensors are normalized to `[0,1]`; GAN-specific scaling is handled in Part B.

# %%
import os, math, time, random, json, csv, copy
from pathlib import Path
import numpy as np
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
EPOCHS_VAE = int(os.environ.get("EPOCHS_VAE", 50))
EPOCHS_VQVAE = int(os.environ.get("EPOCHS_VQVAE", 50))
EPOCHS_PIXELCNN = int(os.environ.get("EPOCHS_PIXELCNN", 30))
EPOCHS_GAN = int(os.environ.get("EPOCHS_GAN", 100))
MAX_TRAIN_BATCHES = int(os.environ.get("MAX_TRAIN_BATCHES", 0))
print("Device:", DEVICE)
if DEVICE.type != "cuda":
    print("WARNING: Run this notebook on the mandatory BITS WILP/Prayogshala GPU infrastructure.")

# CIFAR-10 is stored as RGB tensors in [0,1]. For GAN training we additionally
# map the same images to [-1,1], matching the generator's Tanh output.
tf01 = transforms.ToTensor()
train01 = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf01)
test01 = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf01)
train_loader = DataLoader(train01, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"), drop_last=True)
test_loader = DataLoader(test01, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))
classes = train01.classes

def save_json(obj, name):
    with open(OUT_DIR/name, "w") as f: json.dump(obj, f, indent=2)

def show_grid(x, title="", nrow=10, value_range=(0,1), filename=None):
    x=x.detach().cpu().clamp(*value_range)
    if value_range != (0,1): x=(x-value_range[0])/(value_range[1]-value_range[0])
    grid=utils.make_grid(x, nrow=nrow, padding=2)
    plt.figure(figsize=(10,10)); plt.imshow(grid.permute(1,2,0)); plt.axis("off"); plt.title(title)
    if filename: plt.savefig(OUT_DIR/filename, bbox_inches="tight", dpi=160)
    plt.show()

def psnr_batch(x, recon, data_range=1.0):
    mse=(x-recon).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return (10*torch.log10((data_range**2)/mse)).mean().item()

# FID: Inception features from torchvision, with inputs resized to 299x299.
# This implementation avoids depending on a particular torchmetrics API.
@torch.no_grad()
def inception_features(images, model, batch=64):
    feats=[]
    for chunk in images.split(batch):
        z=F.interpolate(chunk, size=(299,299), mode="bilinear", align_corners=False)
        z=(z - torch.tensor([0.485,0.456,0.406],device=z.device)[None,:,None,None]) / torch.tensor([0.229,0.224,0.225],device=z.device)[None,:,None,None]
        feats.append(model(z).flatten(1).cpu().double())
    return torch.cat(feats,0).numpy()

def calculate_fid(real_images, fake_images, max_images=5000):
    # Use the same number of real and generated images; values are estimates
    # when max_images is smaller than the full CIFAR-10 test set.
    n=min(len(real_images),len(fake_images),max_images)
    real_images=real_images[:n].to(DEVICE); fake_images=fake_images[:n].to(DEVICE)
    net=inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    net.fc=nn.Identity(); net.eval().to(DEVICE)
    fr=inception_features(real_images,net); ff=inception_features(fake_images,net)
    mu1,mu2=fr.mean(0),ff.mean(0)
    c1,c2=np.cov(fr,rowvar=False),np.cov(ff,rowvar=False)
    # Stable FID via symmetric eigendecomposition of covariance product.
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
    display(df)
    return df

# %% [markdown]
# ## 1. β-VAE: β ∈ {1, 2, 4, 10}
# 
# The encoder predicts a Gaussian mean and log-variance; the decoder reconstructs the image. The objective is reconstruction MSE plus β times KL divergence. Validation PSNR is recorded after every epoch.

# %%
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.ReLU(True),       # 16x16
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True), # 8x8
            nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True), # 4x4
            nn.Flatten())
        self.fc_mu=nn.Linear(256*4*4,latent_dim)
        self.fc_logvar=nn.Linear(256*4*4,latent_dim)
        self.fc_dec=nn.Linear(latent_dim,256*4*4)
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1), nn.Sigmoid())
    def encode(self,x):
        h=self.enc(x); return self.fc_mu(h), self.fc_logvar(h).clamp(-12,12)
    def reparameterize(self,mu,logvar):
        return mu+torch.randn_like(mu)*torch.exp(0.5*logvar)
    def decode(self,z): return self.dec(self.fc_dec(z).view(-1,256,4,4))
    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparameterize(mu,lv)
        return self.decode(z),mu,lv

def vae_loss(x,recon,mu,logvar,beta):
    # Mean pixel-wise reconstruction term plus KL divergence per image.
    rec=F.mse_loss(recon,x,reduction="sum")/x.size(0)
    kl=-0.5*(1+logvar-mu.pow(2)-logvar.exp()).sum(1).mean()
    return rec+beta*kl, rec.item(), kl.item()

def train_vae(beta, epochs=EPOCHS_VAE):
    model=ConvVAE().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=2e-4)
    history=[]; start=time.time()
    for ep in range(1,epochs+1):
        model.train(); losses=[]; psnrs=[]
        for x,_ in train_loader:
            x=x.to(DEVICE); opt.zero_grad(set_to_none=True)
            recon,mu,lv=model(x); loss,_,_=vae_loss(x,recon,mu,lv,beta)
            loss.backward(); opt.step()
            losses.append(loss.item()); psnrs.append(psnr_batch(x,recon.detach()))
        model.eval(); val_psnr=[]
        with torch.no_grad():
            for x,_ in test_loader:
                x=x.to(DEVICE); r,_,_=model(x); val_psnr.append(psnr_batch(x,r))
        row={"beta":beta,"epoch":ep,"train_loss":float(np.mean(losses)),
             "val_psnr":float(np.mean(val_psnr)),"seconds":time.time()-start}
        history.append(row)
        print(f"beta={beta} epoch={ep}/{epochs} loss={row['train_loss']:.3f} val PSNR={row['val_psnr']:.2f}")
    torch.save(model.state_dict(),OUT_DIR/f"beta_vae_{beta}.pt")
    save_json(history,f"beta_vae_{beta}_history.json")
    return model,history

vae_models={}; vae_histories=[]
for beta in [1,2,4,10]:
    model,hist=train_vae(beta)
    vae_models[beta]=model; vae_histories.extend(hist)

import pandas as pd
hist_df=pd.DataFrame(vae_histories)
plt.figure(figsize=(8,5))
for b in [1,2,4,10]:
    d=hist_df[hist_df.beta==b]; plt.plot(d.epoch,d.val_psnr,label=f"β={b}")
plt.xlabel("Epoch"); plt.ylabel("Validation PSNR (dB)"); plt.title("β-VAE reconstruction quality")
plt.legend(); plt.grid(); plt.savefig(OUT_DIR/"vae_psnr_vs_epochs.png",dpi=160); plt.show()

real=collect_real_images(5000)
vae_fid_rows=[]
for beta,model in vae_models.items():
    model.eval(); generated=[]
    with torch.no_grad():
        for x,_ in test_loader:
            x=x.to(DEVICE); r,_,_=model(x); generated.append(r.cpu())
    fake=torch.cat(generated)
    fid=calculate_fid(real,fake)
    vae_fid_rows.append({"model":f"beta-VAE beta={beta}","FID":fid})
    print("FID beta",beta,":",fid)
export_metrics(vae_fid_rows,"vae_fid.csv")

# Visual reconstruction and prior samples for the required β=1 model.
x,_=next(iter(test_loader)); x=x[:100].to(DEVICE)
vae_models[1].eval()
with torch.no_grad():
    recon,_,_=vae_models[1](x)
show_grid(x,"CIFAR-10 originals",10,filename="vae_originals_100.png")
show_grid(recon,"β-VAE β=1 reconstructions",10,filename="vae_reconstructions_100.png")
with torch.no_grad():
    z=torch.randn(100,128,device=DEVICE); samples=vae_models[1].decode(z)
show_grid(samples,"β-VAE β=1 prior samples",10,filename="vae_samples_100.png")

# %% [markdown]
# ## 2. VQ-VAE: K ∈ {512, 256, 128}
# 
# The encoder produces an 8×8 latent grid. Vector quantization maps each latent vector to its nearest codebook entry. Codebook quality is summarized by active-code count, utilization percentage and perplexity.

# %%
class VectorQuantizer(nn.Module):
    def __init__(self,K=256,D=64,beta=0.25):
        super().__init__(); self.K=K; self.D=D; self.beta=beta
        self.embedding=nn.Embedding(K,D); self.embedding.weight.data.uniform_(-1/K,1/K)
    def forward(self,z_e):
        # z_e: B,D,H,W. Return straight-through quantized map and indices.
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
    def quantize(self,z): return self.quantizer(z)
    def decode(self,zq): return self.decoder(zq)
    def forward(self,x):
        ze=self.encode(x); zq,q_loss,indices=self.quantize(ze)
        return self.decode(zq),q_loss,indices

class MaskedConv2d(nn.Conv2d):
    def __init__(self,mask_type,*args,**kwargs):
        super().__init__(*args,**kwargs); self.register_buffer("mask",torch.ones_like(self.weight))
        kh,kw=self.kernel_size; self.mask[:,:,kh//2,kw//2+ (1 if mask_type=="A" else 0):]=0
        self.mask[:,:,kh//2+1:,:]=0
    def forward(self,x): return F.conv2d(x,self.weight*self.mask,self.bias,self.stride,self.padding,self.dilation,self.groups)

class PixelCNN(nn.Module):
    def __init__(self,K=256,hidden=128):
        super().__init__()
        self.net=nn.Sequential(
            MaskedConv2d("A",K,hidden,7,padding=3),nn.ReLU(True),
            MaskedConv2d("B",hidden,hidden,3,padding=1),nn.ReLU(True),
            MaskedConv2d("B",hidden,hidden,3,padding=1),nn.ReLU(True),
            nn.Conv2d(hidden,K,1))
    def forward(self,x): return self.net(x)

def train_vqvae(K,epochs=EPOCHS_VQVAE):
    model=VQVAE(K=K).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=2e-4)
    history=[]; start=time.time()
    for ep in range(1,epochs+1):
        model.train(); losses=[]; pvals=[]
        for x,_ in train_loader:
            x=x.to(DEVICE); opt.zero_grad(set_to_none=True)
            r,ql,_=model(x); rec=F.mse_loss(r,x); loss=rec+ql
            loss.backward(); opt.step(); losses.append(loss.item()); pvals.append(psnr_batch(x,r.detach()))
        model.eval(); vals=[]
        with torch.no_grad():
            for x,_ in test_loader:
                x=x.to(DEVICE); r,_,_=model(x); vals.append(psnr_batch(x,r))
        row={"K":K,"epoch":ep,"train_loss":float(np.mean(losses)),
             "val_psnr":float(np.mean(vals)),"seconds":time.time()-start}
        history.append(row); print(f"K={K} epoch={ep}/{epochs} loss={row['train_loss']:.4f} PSNR={row['val_psnr']:.2f}")
    torch.save(model.state_dict(),OUT_DIR/f"vqvae_K{K}.pt")
    save_json(history,f"vqvae_K{K}_history.json")
    return model,history

vq_models={}; vq_histories=[]
for K in [512,256,128]:
    m,h=train_vqvae(K); vq_models[K]=m; vq_histories.extend(h)

# Plot reconstruction PSNR versus epochs for all codebook sizes.
vqdf=pd.DataFrame(vq_histories)
plt.figure(figsize=(8,5))
for K in [512,256,128]:
    d=vqdf[vqdf.K==K]; plt.plot(d.epoch,d.val_psnr,label=f"K={K}")
plt.xlabel("Epoch"); plt.ylabel("Validation PSNR (dB)"); plt.title("VQ-VAE reconstruction quality")
plt.legend(); plt.grid(); plt.savefig(OUT_DIR/"vqvae_psnr_vs_epochs.png",dpi=160); plt.show()

# Encode every training image to a discrete 8x8 latent map.
def extract_latents(model, dataset):
    loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    maps=[]
    model.eval()
    with torch.no_grad():
        for x,_ in loader:
            _,_,idx=model(x.to(DEVICE)); maps.append(idx.cpu())
    return torch.cat(maps)
latents={}
for K,m in vq_models.items():
    latents[K]=extract_latents(m,train01)
    torch.save(latents[K],OUT_DIR/f"latents_K{K}.pt")
    print("K",K,"latent tensor",tuple(latents[K].shape))

def train_pixelcnn(K,latent_maps,epochs=EPOCHS_PIXELCNN):
    prior=PixelCNN(K).to(DEVICE); opt=torch.optim.Adam(prior.parameters(),lr=2e-4)
    loader=DataLoader(latent_maps,batch_size=256,shuffle=True)
    for ep in range(1,epochs+1):
        prior.train(); losses=[]
        for idx in loader:
            idx=idx.long().to(DEVICE); opt.zero_grad(set_to_none=True)
            logits=prior(F.one_hot(idx,num_classes=K).permute(0,3,1,2).float())
            loss=F.cross_entropy(logits,idx); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"PixelCNN K={K} epoch={ep}/{epochs} CE={np.mean(losses):.4f}")
    torch.save(prior.state_dict(),OUT_DIR/f"pixelcnn_K{K}.pt")
    return prior

def sample_pixelcnn(prior,K,n=100,H=8,W=8):
    prior.eval(); x=torch.zeros(n,H,W,dtype=torch.long,device=DEVICE)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                logits=prior(F.one_hot(x,num_classes=K).permute(0,3,1,2).float())
                probs=F.softmax(logits[:,:,i,j],dim=1)
                x[:,i,j]=torch.multinomial(probs,1).squeeze(1)
    return x

pixelcnns={}; vq_fid_rows=[]; codebook_rows=[]
for K,m in vq_models.items():
    prior=train_pixelcnn(K,latents[K]); pixelcnns[K]=prior
    # Usage statistics: perplexity, active codes, and utilization percentage.
    counts=torch.bincount(latents[K].flatten(),minlength=K).float()
    probs=counts/counts.sum(); entropy=-(probs[probs>0]*torch.log(probs[probs>0])).sum()
    perplexity=float(torch.exp(entropy)); active=int((counts>0).sum())
    codebook_rows.append({"K":K,"active_codes":active,"utilization_pct":100*active/K,"perplexity":perplexity})
    # Reconstruction PSNR and FID of prior-generated samples.
    m.eval(); recs=[]; ps=[]
    with torch.no_grad():
        for x,_ in test_loader:
            x=x.to(DEVICE); r,_,_=m(x); recs.append(r.cpu()); ps.append(psnr_batch(x,r))
    recon=torch.cat(recs)
    sampled=sample_pixelcnn(prior,K,1000)
    emb=m.quantizer.embedding(sampled).permute(0,3,1,2).contiguous()
    with torch.no_grad(): generated=m.decode(emb).cpu()
    fid=calculate_fid(real,generated,max_images=1000)
    vq_fid_rows.append({"model":f"VQ-VAE+PixelCNN K={K}","FID":fid,
                        "reconstruction_PSNR":float(np.mean(ps))})
    print("K",K,"FID",fid)
    if K==256:
        show_grid(generated[:100],"VQ-VAE K=256 + PixelCNN samples",10,filename="vqvae_pixelcnn_samples_100.png")
        x,_=next(iter(test_loader)); x=x[:100].to(DEVICE)
        with torch.no_grad(): rr,_,_=m(x)
        show_grid(rr,"VQ-VAE K=256 reconstructions",10,filename="vqvae_reconstructions_100.png")
export_metrics(vq_fid_rows,"vqvae_fid_psnr.csv")
export_metrics(codebook_rows,"vqvae_codebook_quality.csv")

# %% [markdown]
# ## 3. Required outputs and interpretation
# 
# Check `outputs/` for model checkpoints, per-epoch histories, PSNR curves, FID CSVs, codebook statistics and 100-image grids. In your report, discuss the rate–distortion trade-off in β-VAE: increasing β generally regularizes the latent distribution more strongly, which can improve prior sampling but often reduces reconstruction fidelity. For VQ-VAE, interpret codebook utilization and perplexity alongside FID and PSNR; a large nominal K does not guarantee that all codes are used.
# 
# **Important:** FID is computationally expensive and is an estimate when `max_images` is limited. For final reported scores, use the same sample count and evaluation procedure for every model. The provided Inception implementation uses torchvision's pretrained Inception-v3 features; state this protocol in the report.

# %% [markdown]
# ## 4. Export notebook to HTML
# 
# Run this in a terminal from the notebook directory after executing all cells:
# 
# ```bash
# jupyter nbconvert --to html --execute PartA_VAE_VQVAE_PixelCNN.ipynb
# ```
# 
# If execution is already complete and you only need to export saved outputs, omit `--execute`.

