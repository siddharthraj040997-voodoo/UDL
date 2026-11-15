# %% [markdown]
# # Assignment 2 – Part B: WGAN and WGAN-GP
# 
# **Dataset:** CIFAR-10  
# **Framework:** PyTorch
# 
# Run on the BITS WILP Prayogshala GPU infrastructure. Checkpoints, generated images, histories and metrics are saved in `./outputs`.

# %% [markdown]
# ## 0. Setup
# 
# Install if required:
# 
# ```bash
# pip install torch torchvision matplotlib numpy scipy pandas pillow
# ```
# 
# CIFAR-10 is first represented in `[0,1]` and mapped to `[-1,1]` for GAN optimization so that it matches the generator's Tanh output.

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
EPOCHS_VAE = 50
EPOCHS_VQVAE = 50
EPOCHS_PIXELCNN = 30
EPOCHS_GAN = 100
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
# ## 1. WGAN and WGAN-GP
# 
# The critic (not a probability discriminator) assigns real-valued scores. The original WGAN uses weight clipping and RMSProp. WGAN-GP replaces clipping with a gradient penalty and uses Adam. The critic is updated multiple times per generator update.

# %%
# GAN dataset: CIFAR-10 in [-1,1], as required by a Tanh generator.
gan_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)])
gan_train=datasets.CIFAR10(DATA_DIR,train=True,download=True,transform=gan_tf)
gan_test=datasets.CIFAR10(DATA_DIR,train=False,download=True,transform=gan_tf)
gan_loader=DataLoader(gan_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,
                      pin_memory=(DEVICE.type=="cuda"),drop_last=True)

LATENT=128
class Generator(nn.Module):
    def __init__(self,z_dim=LATENT):
        super().__init__(); self.fc=nn.Linear(z_dim,256*4*4)
        self.net=nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),nn.BatchNorm2d(64),nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1),nn.Tanh())
    def forward(self,z): return self.net(self.fc(z).view(-1,256,4,4))

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,64,4,2,1),nn.LeakyReLU(.2,True),
            nn.Conv2d(64,128,4,2,1),nn.LayerNorm([128,8,8]),nn.LeakyReLU(.2,True),
            nn.Conv2d(128,256,4,2,1),nn.LayerNorm([256,4,4]),nn.LeakyReLU(.2,True),
            nn.Flatten(),nn.Linear(256*4*4,1))
    def forward(self,x): return self.net(x).view(-1)

def gradient_penalty(critic,real,fake):
    alpha=torch.rand(real.size(0),1,1,1,device=DEVICE)
    interp=(alpha*real+(1-alpha)*fake).requires_grad_(True)
    scores=critic(interp)
    grads=torch.autograd.grad(outputs=scores,inputs=interp,
        grad_outputs=torch.ones_like(scores),create_graph=True,retain_graph=True,only_inputs=True)[0]
    return ((grads.view(grads.size(0),-1).norm(2,dim=1)-1)**2).mean()

def train_wgan(use_gp=True,epochs=EPOCHS_GAN,critic_steps=5,lambda_gp=10.0):
    G=Generator().to(DEVICE); C=Critic().to(DEVICE)
    optG=torch.optim.RMSprop(G.parameters(),lr=5e-5) if not use_gp else torch.optim.Adam(G.parameters(),lr=1e-4,betas=(0.0,0.9))
    optC=torch.optim.RMSprop(C.parameters(),lr=5e-5) if not use_gp else torch.optim.Adam(C.parameters(),lr=1e-4,betas=(0.0,0.9))
    history=[]; start=time.time()
    for ep in range(1,epochs+1):
        G.train(); C.train(); c_losses=[]; g_losses=[]; gps=[]
        for step,(real,_) in enumerate(gan_loader):
            real=real.to(DEVICE); bs=real.size(0)
            for _ in range(critic_steps):
                z=torch.randn(bs,LATENT,device=DEVICE); fake=G(z).detach()
                optC.zero_grad(set_to_none=True)
                # Minimize E[C(fake)] - E[C(real)].
                c_loss=C(fake).mean()-C(real).mean()
                if use_gp:
                    gp=gradient_penalty(C,real,fake); loss_c=c_loss+lambda_gp*gp; gps.append(gp.item())
                else:
                    # Weight clipping is the original WGAN Lipschitz constraint.
                    loss_c=c_loss
                loss_c.backward(); optC.step()
                if not use_gp:
                    for p in C.parameters(): p.data.clamp_(-0.01,0.01)
                c_losses.append(c_loss.item())
            z=torch.randn(bs,LATENT,device=DEVICE)
            optG.zero_grad(set_to_none=True); g_loss=-C(G(z)).mean()
            g_loss.backward(); optG.step(); g_losses.append(g_loss.item())
        row={"epoch":ep,"critic_loss":float(np.mean(c_losses)),"generator_loss":float(np.mean(g_losses)),
             "mean_gradient_penalty":float(np.mean(gps)) if gps else None,"seconds":time.time()-start}
        history.append(row)
        print(f"{'WGAN-GP' if use_gp else 'WGAN'} {ep}/{epochs}: C={row['critic_loss']:.3f} G={row['generator_loss']:.3f}")
        if ep%10==0:
            G.eval()
            with torch.no_grad(): preview=G(torch.randn(100,LATENT,device=DEVICE))
            show_grid(preview, f"{'WGAN-GP' if use_gp else 'WGAN'} epoch {ep}",10,(-1,1),
                      filename=f"{'wgangp' if use_gp else 'wgan'}_epoch_{ep}.png")
    name="wgangp" if use_gp else "wgan"
    torch.save(G.state_dict(),OUT_DIR/f"{name}_generator.pt")
    torch.save(C.state_dict(),OUT_DIR/f"{name}_critic.pt")
    save_json(history,f"{name}_history.json")
    return G,C,history

# Train both variants. Increase epochs for final results if lab time permits.
wgan_G,wgan_C,wgan_hist=train_wgan(use_gp=False)
wgangp_G,wgangp_C,wgangp_hist=train_wgan(use_gp=True)

@torch.no_grad()
def generate_gan_images(G,n=5000):
    G.eval(); out=[]
    for start in range(0,n, BATCH_SIZE):
        bs=min(BATCH_SIZE,n-start)
        out.append(((G(torch.randn(bs,LATENT,device=DEVICE))+1)/2).clamp(0,1).cpu())
    return torch.cat(out)

real=collect_real_images(5000)
wgan_samples=generate_gan_images(wgan_G,5000)
wgangp_samples=generate_gan_images(wgangp_G,5000)
show_grid(wgan_samples[:100],"WGAN generated samples (100)",10,filename="wgan_samples_100.png")
show_grid(wgangp_samples[:100],"WGAN-GP generated samples (100)",10,filename="wgangp_samples_100.png")
wgan_fid=calculate_fid(real,wgan_samples)
wgangp_fid=calculate_fid(real,wgangp_samples)
gan_results=[
 {"model":"WGAN","FID":wgan_fid,"training_seconds":wgan_hist[-1]["seconds"]},
 {"model":"WGAN-GP","FID":wgangp_fid,"training_seconds":wgangp_hist[-1]["seconds"]}
]
export_metrics(gan_results,"gan_fid_training_time.csv")

# %% [markdown]
# ## 2. Evaluation and deliverables
# 
# The notebook saves 100-image sample grids, generator/critic checkpoints, loss histories, FID scores and training time. Compare the two models using identical FID sample counts and the same Inception feature protocol. Do not interpret critic losses as directly comparable image-quality scores.
# 
# For the final submission, retain the epoch count and training configuration actually used. The assignment requires evidence of using the BITS GPU infrastructure; include a genuine screenshot from your Prayogshala session.

# %% [markdown]
# ## 3. Export to HTML
# 
# ```bash
# jupyter nbconvert --to html --execute PartB_WGAN_WGANGP.ipynb
# ```

