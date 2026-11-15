# UDL Assignment 2 – execution guide

## Included
- `PartA_VAE_VQVAE_PixelCNN.ipynb`: beta-VAE (beta=1,2,4,10), VQ-VAE (K=512,256,128), PixelCNN priors, PSNR/FID and codebook metrics.
- `PartB_WGAN_WGANGP.ipynb`: WGAN with weight clipping and WGAN-GP with gradient penalty, generated samples, FID and training time.

## Environment
Use the BITS WILP Prayogshala GPU environment as required by the problem statement. Recommended packages:
`torch torchvision matplotlib numpy scipy pandas pillow jupyter nbconvert`

## Suggested execution
1. Upload both notebooks to the lab GPU environment.
2. Set epoch counts after checking the available GPU time. The defaults are intentionally substantial; do not reduce them for the final run without recording the change.
3. Run Part A from top to bottom. It trains four beta-VAE runs and three VQ-VAE + PixelCNN runs.
4. Run Part B from top to bottom. It trains WGAN and WGAN-GP.
5. Keep the generated `outputs/` folder. It contains checkpoints, histories, CSV metrics and sample grids.
6. Export HTML with `jupyter nbconvert --to html --execute <notebook>.ipynb`.
7. Create Part C PDF using actual metrics and visual samples. Include exactly/at least 100 visual samples for beta-VAE beta=1, VQ-VAE K=256, WGAN and WGAN-GP, as required.
8. Capture a genuine screenshot showing use of BITS-provided infrastructure.

## Important methodological notes
- Images are loaded as RGB tensors in [0,1]. GAN inputs are linearly mapped to [-1,1] to match Tanh output.
- PSNR is calculated from per-image mean squared error with data range 1.
- FID is calculated with pretrained torchvision Inception-v3 features. The notebook uses a consistent evaluation pipeline and a default sample cap of 5,000 (1,000 for VQ-VAE prior-generated samples). State the exact sample count and feature extractor in your report.
- PixelCNN sampling is autoregressive over the 8x8 latent grid and can be slow. It is implemented as a straightforward masked-convolution baseline.
- Results depend on training duration, GPU, architecture and random seed. No score is guaranteed; train, inspect curves and report the real observed values.
