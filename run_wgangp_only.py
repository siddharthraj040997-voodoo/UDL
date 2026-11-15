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
from torch.utils.data import DataLoader
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
EPOCHS_GAN = int(os.environ.get("EPOCHS_GAN", 100))
LATENT = 128

print("Device:", DEVICE)
print(f"Starting dedicated WGAN-GP run for {EPOCHS_GAN} epochs...")

gan_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
gan_train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=gan_tf)
gan_test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=gan_tf)
gan_loader = DataLoader(gan_train, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"), drop_last=True)
test_loader = DataLoader(gan_test, batch_size=BATCH_SIZE, shuffle=False,
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
        # map back to [0,1] for FID calculation
        out.append(((x+1)/2).clamp(0,1))
        if sum(t.size(0) for t in out)>=n: break
    return torch.cat(out)[:n]

class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc=nn.Linear(latent_dim,256*4*4)
        self.net=nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1), nn.Tanh())
    def forward(self,z): return self.net(self.fc(z).view(-1,256,4,4))

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,1,4,1,0))
    def forward(self,x): return self.net(x).view(-1)

def gradient_penalty(critic,real,fake):
    alpha=torch.rand(real.size(0),1,1,1,device=DEVICE)
    interp=(alpha*real+(1-alpha)*fake).requires_grad_(True)
    scores=critic(interp)
    grads=torch.autograd.grad(outputs=scores,inputs=interp,
        grad_outputs=torch.ones_like(scores),create_graph=True,retain_graph=True,only_inputs=True)[0]
    return ((grads.view(grads.size(0),-1).norm(2,dim=1)-1)**2).mean()

# Load WGAN if available, or initialize
wgan_G = Generator().to(DEVICE)
if (OUT_DIR/"wgan_generator.pt").exists():
    print("Found existing WGAN generator weights, loading...")
    wgan_G.load_state_dict(torch.load(OUT_DIR/"wgan_generator.pt", map_location=DEVICE))

# Train WGAN-GP
print("Training WGAN-GP...")
G = Generator().to(DEVICE)
C = Critic().to(DEVICE)
optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
optC = torch.optim.Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))
wgangp_hist = []; start = time.time()
critic_steps = 5; lambda_gp = 10.0

for ep in range(1, EPOCHS_GAN+1):
    G.train(); C.train(); c_losses=[]; g_losses=[]; gps=[]
    for step,(real,_) in enumerate(gan_loader):
        real=real.to(DEVICE); bs=real.size(0)
        for _ in range(critic_steps):
            z=torch.randn(bs,LATENT,device=DEVICE); fake=G(z).detach()
            optC.zero_grad(set_to_none=True)
            c_loss=C(fake).mean()-C(real).mean()
            gp=gradient_penalty(C,real,fake)
            loss_c=c_loss+lambda_gp*gp
            gps.append(gp.item())
            loss_c.backward(); optC.step()
            c_losses.append(c_loss.item())
        z=torch.randn(bs,LATENT,device=DEVICE)
        optG.zero_grad(set_to_none=True); g_loss=-C(G(z)).mean()
        g_loss.backward(); optG.step(); g_losses.append(g_loss.item())
    row={"epoch":ep,"critic_loss":float(np.mean(c_losses)),"generator_loss":float(np.mean(g_losses)),
         "mean_gradient_penalty":float(np.mean(gps)),"seconds":time.time()-start}
    wgangp_hist.append(row)
    print(f"WGAN-GP {ep:3d}/{EPOCHS_GAN}: C={row['critic_loss']:.3f} G={row['generator_loss']:.3f} GP={row['mean_gradient_penalty']:.3f}", flush=True)
    if ep%10==0:
        G.eval()
        with torch.no_grad(): preview=G(torch.randn(100,LATENT,device=DEVICE))
        show_grid(preview, f"WGAN-GP epoch {ep}", 10, (-1,1), filename=f"wgangp_epoch_{ep}.png")

torch.save(G.state_dict(), OUT_DIR/"wgangp_generator.pt")
torch.save(C.state_dict(), OUT_DIR/"wgangp_critic.pt")
save_json(wgangp_hist, "wgangp_history.json")

# Evaluate FID
print("Generating evaluation samples...")
@torch.no_grad()
def generate_gan_images(model, n=5000):
    model.eval(); out=[]
    for s in range(0,n,BATCH_SIZE):
        bs=min(BATCH_SIZE,n-s)
        out.append(((model(torch.randn(bs,LATENT,device=DEVICE))+1)/2).clamp(0,1).cpu())
    return torch.cat(out)

real=collect_real_images(5000)
wgan_samples=generate_gan_images(wgan_G,5000)
wgangp_samples=generate_gan_images(G,5000)

show_grid(wgan_samples[:100],"WGAN generated samples (100)",10,filename="wgan_samples_100.png")
show_grid(wgangp_samples[:100],"WGAN-GP generated samples (100)",10,filename="wgangp_samples_100.png")

print("Calculating FID for WGAN...")
wgan_fid=calculate_fid(real,wgan_samples)
print("Calculating FID for WGAN-GP...")
wgangp_fid=calculate_fid(real,wgangp_samples)

wgan_time = 0
if (OUT_DIR/"wgan_history.json").exists():
    try:
        w_hist = json.load(open(OUT_DIR/"wgan_history.json"))
        wgan_time = w_hist[-1]["seconds"]
    except: pass

gan_results=[
 {"model":"WGAN","FID":wgan_fid,"training_seconds":wgan_time},
 {"model":"WGAN-GP","FID":wgangp_fid,"training_seconds":wgangp_hist[-1]["seconds"]}
]
import pandas as pd
df=pd.DataFrame(gan_results)
df.to_csv(OUT_DIR/"gan_fid_training_time.csv", index=False)
print("GAN Results:")
print(df.to_string())
print("Part B WGAN-GP Complete! All metrics saved.")
