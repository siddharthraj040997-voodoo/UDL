import json, os
import base64

OUT_DIR = "outputs"

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load all JSON histories for exact numbers
w_hist = json.load(open(os.path.join(OUT_DIR, "wgan_history.json")))
gp_hist = json.load(open(os.path.join(OUT_DIR, "wgangp_history.json")))
v1_hist = json.load(open(os.path.join(OUT_DIR, "beta_vae_1_history.json")))
v2_hist = json.load(open(os.path.join(OUT_DIR, "beta_vae_2_history.json")))
v4_hist = json.load(open(os.path.join(OUT_DIR, "beta_vae_4_history.json")))
v10_hist = json.load(open(os.path.join(OUT_DIR, "beta_vae_10_history.json")))

# Charts and sample grids
b64_vae_psnr_chart = get_base64_image(os.path.join(OUT_DIR, "vae_psnr_vs_epochs.png"))
b64_vqvae_psnr_chart = get_base64_image(os.path.join(OUT_DIR, "vqvae_psnr_vs_epochs.png"))
b64_vae_rec = get_base64_image(os.path.join(OUT_DIR, "vae_reconstructions_100.png"))
b64_vae_orig = get_base64_image(os.path.join(OUT_DIR, "vae_originals_100.png"))
b64_vae_samp = get_base64_image(os.path.join(OUT_DIR, "vae_samples_100.png"))
b64_vqvae_orig = get_base64_image(os.path.join(OUT_DIR, "vqvae_originals_100.png"))
b64_vqvae_rec = get_base64_image(os.path.join(OUT_DIR, "vqvae_reconstructions_100.png"))
b64_vqvae_samp = get_base64_image(os.path.join(OUT_DIR, "vqvae_pixelcnn_samples_100.png"))
b64_wgan_samp = get_base64_image(os.path.join(OUT_DIR, "wgan_samples_100.png"))
b64_wgangp_samp = get_base64_image(os.path.join(OUT_DIR, "wgangp_samples_100.png"))

# WGAN epoch evolution snapshots (epochs 10, 50, 100)
b64_wgan_ep10 = get_base64_image(os.path.join(OUT_DIR, "wgan_epoch_10.png"))
b64_wgan_ep50 = get_base64_image(os.path.join(OUT_DIR, "wgan_epoch_50.png"))
b64_wgan_ep100 = get_base64_image(os.path.join(OUT_DIR, "wgan_epoch_100.png"))
b64_wgangp_ep10 = get_base64_image(os.path.join(OUT_DIR, "wgangp_epoch_10.png"))
b64_wgangp_ep50 = get_base64_image(os.path.join(OUT_DIR, "wgangp_epoch_50.png"))
b64_wgangp_ep100 = get_base64_image(os.path.join(OUT_DIR, "wgangp_epoch_100.png"))

# Construct SVG chart for WGAN vs WGAN-GP loss comparison
w_epochs = [r['epoch'] for r in w_hist]
w_losses = [r['critic_loss'] for r in w_hist]
gp_losses = [r['critic_loss'] for r in gp_hist]

# Normalize to SVG coordinates (width=700, height=220)
min_l, max_l = min(min(w_losses), min(gp_losses)), max(max(w_losses), max(gp_losses))
def to_svg_pt(ep, val):
    x = 50 + (ep - 1) * (620 / 99)
    y = 20 + (max_l - val) * (180 / (max_l - min_l + 1e-6))
    return f"{x:.1f},{y:.1f}"

pts_w = " ".join([to_svg_pt(ep, l) for ep, l in zip(w_epochs, w_losses)])
pts_gp = " ".join([to_svg_pt(ep, l) for ep, l in zip(w_epochs, gp_losses)])

svg_chart = f"""
<svg viewBox="0 0 720 250" style="width:100%; max-width:700px; background:#ffffff; border:1px solid #e2e8f0; border-radius:6px; margin:12px auto; display:block;">
  <!-- Grid lines -->
  <line x1="50" y1="20" x2="670" y2="20" stroke="#edf2f7" stroke-width="1"/>
  <line x1="50" y1="65" x2="670" y2="65" stroke="#edf2f7" stroke-width="1"/>
  <line x1="50" y1="110" x2="670" y2="110" stroke="#edf2f7" stroke-width="1"/>
  <line x1="50" y1="155" x2="670" y2="155" stroke="#edf2f7" stroke-width="1"/>
  <line x1="50" y1="200" x2="670" y2="200" stroke="#cbd5e0" stroke-width="1.5"/>
  <line x1="50" y1="20" x2="50" y2="200" stroke="#cbd5e0" stroke-width="1.5"/>
  <!-- Axis labels -->
  <text x="50" y="218" font-size="11" fill="#718096" text-anchor="middle">1</text>
  <text x="205" y="218" font-size="11" fill="#718096" text-anchor="middle">25</text>
  <text x="360" y="218" font-size="11" fill="#718096" text-anchor="middle">50</text>
  <text x="515" y="218" font-size="11" fill="#718096" text-anchor="middle">75</text>
  <text x="670" y="218" font-size="11" fill="#718096" text-anchor="middle">100 (Epoch)</text>
  <text x="15" y="115" font-size="11" fill="#718096" transform="rotate(-90 15,115)" text-anchor="middle">Critic Loss</text>
  <!-- Curves -->
  <polyline fill="none" stroke="#e53e3e" stroke-width="2" points="{pts_w}"/>
  <polyline fill="none" stroke="#3182ce" stroke-width="2" points="{pts_gp}"/>
  <!-- Legend -->
  <circle cx="200" cy="238" r="4" fill="#e53e3e"/>
  <text x="210" y="242" font-size="11" fill="#2d3748">WGAN Critic Loss (Weight Clipping)</text>
  <circle cx="450" cy="238" r="4" fill="#3182ce"/>
  <text x="460" y="242" font-size="11" fill="#2d3748">WGAN-GP Critic Loss (Gradient Penalty)</text>
</svg>
"""

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>UDL Assignment 2 - Comprehensive Final Submission Report</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 12mm 15mm 12mm;
    @bottom-right {{
      content: "Page " counter(page);
      font-size: 10px;
      color: #718096;
    }}
  }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    color: #1a202c;
    margin: 0;
    padding: 0;
    font-size: 12.5px;
  }}
  .header-box {{
    border-bottom: 2px solid #2b6cb0;
    padding-bottom: 10px;
    margin-bottom: 16px;
  }}
  h1 {{
    color: #1a365d;
    font-size: 21px;
    margin: 0 0 6px 0;
  }}
  .meta-grid {{
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 11.5px;
    margin-bottom: 12px;
  }}
  .team-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 6px;
    background: #ffffff;
  }}
  .team-table th, .team-table td {{
    border: 1px solid #cbd5e0;
    padding: 5px 8px;
    text-align: left;
    font-size: 11.5px;
  }}
  .team-table th {{
    background: #edf2f7;
    font-weight: 600;
  }}
  h2 {{
    color: #2b6cb0;
    font-size: 15px;
    border-bottom: 1px solid #cbd5e0;
    padding-bottom: 3px;
    margin-top: 18px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  h3 {{
    color: #2d3748;
    font-size: 13.5px;
    margin-top: 12px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}
  p, ul, ol {{
    margin: 5px 0 8px 0;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 14px 0;
    font-size: 11.5px;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e0;
    padding: 6px 8px;
    text-align: left;
  }}
  th {{
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #f7fafc;
  }}
  .highlight {{
    background-color: #ebf8ff;
    font-weight: 600;
  }}
  .metric-val {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-weight: 600;
  }}
  .image-container {{
    text-align: center;
    margin: 12px 0;
    page-break-inside: avoid;
  }}
  .image-container img {{
    max-width: 90%;
    height: auto;
    border: 1px solid #cbd5e0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  .img-grid-3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 8px 0;
    page-break-inside: avoid;
  }}
  .img-grid-3 img {{
    width: 100%;
    border: 1px solid #cbd5e0;
    border-radius: 4px;
  }}
  .caption {{
    font-size: 11px;
    color: #4a5568;
    margin-top: 4px;
    font-style: italic;
  }}
  .page-break {{
    page-break-before: always;
  }}
  .formula {{
    text-align: center;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 13.5px;
    margin: 6px 0;
    background: #f8fafc;
    padding: 5px;
    border-radius: 4px;
  }}
</style>
</head>
<body>

<div class="header-box">
  <h1>UDL Assignment 2: Part C Comparative Discussion Report</h1>
  <div class="meta-grid">
    <div style="margin-bottom: 5px;"><strong>Course:</strong> AIMLZG533 – Unsupervised Deep Learning &nbsp;|&nbsp; <strong>Group:</strong> UDL Group 22 &nbsp;|&nbsp; <strong>Dataset:</strong> CIFAR-10 (32&times;32 RGB)</div>
    <table class="team-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Student Name</th>
          <th>BITS Student ID</th>
          <th>Institutional Email</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td><strong>SIDDHARTH RAJ</strong></td>
          <td><code>2025ae05706</code></td>
          <td>2025ae05706@wilp.bits-pilani.ac.in</td>
        </tr>
        <tr>
          <td>2</td>
          <td><strong>VISHAL ANAND</strong></td>
          <td><code>2025ae05692</code></td>
          <td>2025ae05692@wilp.bits-pilani.ac.in</td>
        </tr>
        <tr>
          <td>3</td>
          <td><strong>DIVYANSH SRIVASTAVA</strong></td>
          <td><code>2025ae05690</code></td>
          <td>2025ae05690@wilp.bits-pilani.ac.in</td>
        </tr>
        <tr>
          <td>4</td>
          <td><strong>MOHAMED FHAKRUDDIN</strong></td>
          <td><code>2025ae05817</code></td>
          <td>2025ae05817@wilp.bits-pilani.ac.in</td>
        </tr>
        <tr>
          <td>5</td>
          <td><strong>GOHIL KUNAL DEEPAK VANITA</strong></td>
          <td><code>2025ae05815</code></td>
          <td>2025ae05815@wilp.bits-pilani.ac.in</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<h2>1. Infrastructure & Execution Compliance</h2>
<p>In accordance with assignment requirements, all computational workflows, environment setup, and data pipelines were validated on the <strong>BITS WILP Prayogshala</strong> infrastructure under account <code>s2025ae05706</code>. Accelerated model training across the full epoch schedule (100 epochs for WGAN/WGAN-GP, 50 epochs for VAE/VQ-VAE, 30 epochs for PixelCNN) was executed on NVIDIA CUDA GPUs. All quantitative metrics, convergence histories, checkpoints, and generated grids were saved directly to <code>./outputs</code>.</p>

<h2>2. Part A: &beta;-VAE Empirical Analysis (&beta; &isin; {{1, 2, 4, 10}})</h2>

<h3>2.1 Rate-Distortion &amp; Disentanglement Trade-Off</h3>
<p>The variational objective balances reconstruction accuracy against latent capacity via hyperparameter &beta;:</p>
<div class="formula">
  L(&theta;, &phi;; x, z, &beta;) = E<sub>q<sub>&phi;</sub>(z|x)</sub>[log p<sub>&theta;</sub>(x|z)] &minus; &beta; &middot; D<sub>KL</sub>(q<sub>&phi;</sub>(z|x) || p(z))
</div>
<p><strong>Observed Dynamics:</strong>
As &beta; increases from 1 to 10, the KL divergence term imposes stronger pressure toward the isotropic Gaussian prior N(0, I). This regularization promotes disentanglement of generative latent factors at the cost of informational bottleneck throughput. Consequently, final validation PSNR drops from <strong>18.14 dB down to 14.88 dB</strong>, while the Frechet Inception Distance (FID) increases from <strong>164.38 to 314.80</strong> due to reconstructive smoothing.</p>

<table>
  <thead>
    <tr>
      <th>Model Variant</th>
      <th>&beta; Weight</th>
      <th>Train Loss (Ep 50)</th>
      <th>Reconstruction PSNR (dB)</th>
      <th>Inception FID (&darr;)</th>
      <th>Training Time</th>
      <th>Empirical Trade-off Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr class="highlight">
      <td><strong>&beta;-VAE (&beta;=1)</strong></td>
      <td>1.0</td>
      <td class="metric-val">73.87</td>
      <td class="metric-val">18.14 dB</td>
      <td class="metric-val">164.38</td>
      <td class="metric-val">440.26 s</td>
      <td>Highest fidelity reconstructions; sharpest visual samples.</td>
    </tr>
    <tr>
      <td><strong>&beta;-VAE (&beta;=2)</strong></td>
      <td>2.0</td>
      <td class="metric-val">91.54</td>
      <td class="metric-val">17.11 dB</td>
      <td class="metric-val">215.19</td>
      <td class="metric-val">499.42 s</td>
      <td>Balanced rate-distortion; moderate high-frequency softening.</td>
    </tr>
    <tr>
      <td><strong>&beta;-VAE (&beta;=4)</strong></td>
      <td>4.0</td>
      <td class="metric-val">111.23</td>
      <td class="metric-val">16.12 dB</td>
      <td class="metric-val">268.77</td>
      <td class="metric-val">486.86 s</td>
      <td>Increased regularization; noticeable reduction in sample contrast.</td>
    </tr>
    <tr>
      <td><strong>&beta;-VAE (&beta;=10)</strong></td>
      <td>10.0</td>
      <td class="metric-val">139.03</td>
      <td class="metric-val">14.88 dB</td>
      <td class="metric-val">314.80</td>
      <td class="metric-val">481.62 s</td>
      <td>Strongest prior enforcement; pronounced reconstructive blur.</td>
    </tr>
  </tbody>
</table>

<div class="image-container">
  <h3>Figure 1: &beta;-VAE Reconstruction Quality (Validation PSNR vs. Epochs)</h3>
  {"<img src='data:image/png;base64," + b64_vae_psnr_chart + "' style='max-width:80%;' />" if b64_vae_psnr_chart else "<p>[vae_psnr_vs_epochs.png]</p>"}
  <div class="caption">Convergence curves over 50 epochs showing stable learning across all four &beta; values.</div>
</div>

<div class="page-break"></div>

<h2>3. Part A: VQ-VAE &amp; Autoregressive PixelCNN Prior</h2>

<h3>3.1 Vector Quantization Dynamics &amp; Codebook Metrics</h3>
<p>VQ-VAE discretizes continuous spatial latents onto an 8&times;8 feature grid with embedding vectors e<sub>k</sub> &isin; R<sup>64</sup> using nearest-neighbor quantization and straight-through gradient estimation.</p>

<table>
  <thead>
    <tr>
      <th>Codebook Size (K)</th>
      <th>Recon MSE Loss</th>
      <th>Validation PSNR</th>
      <th>Active Codes</th>
      <th>Utilization %</th>
      <th>Perplexity</th>
      <th>PixelCNN CE Loss</th>
      <th>Prior FID (&darr;)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>K = 512</strong></td>
      <td class="metric-val">0.00431</td>
      <td class="metric-val">24.10 dB</td>
      <td class="metric-val">199 / 512</td>
      <td class="metric-val">38.87%</td>
      <td class="metric-val">169.50</td>
      <td class="metric-val">3.600</td>
      <td class="metric-val">114.46</td>
    </tr>
    <tr class="highlight">
      <td><strong>K = 256</strong></td>
      <td class="metric-val">0.00459</td>
      <td class="metric-val">23.82 dB</td>
      <td class="metric-val">159 / 256</td>
      <td class="metric-val">62.11%</td>
      <td class="metric-val">132.22</td>
      <td class="metric-val">3.384</td>
      <td class="metric-val">117.83</td>
    </tr>
    <tr>
      <td><strong>K = 128</strong></td>
      <td class="metric-val">0.00470</td>
      <td class="metric-val">23.70 dB</td>
      <td class="metric-val">128 / 128</td>
      <td class="metric-val">100.00%</td>
      <td class="metric-val">112.48</td>
      <td class="metric-val">3.299</td>
      <td class="metric-val">116.39</td>
    </tr>
  </tbody>
</table>

<p><strong>Codebook Utilization vs. Distortion Analysis:</strong>
Decreasing K from 512 to 128 raises codebook utilization to 100%, completely eliminating code collapse. K=256 provides the optimal compromise: high active code perplexity (132.22), strong codebook utilization (62.11%), and crisp 23.82 dB reconstruction PSNR. The PixelCNN autoregressive prior trained on K=256 models the spatial index distribution effectively, achieving an FID of 117.83.</p>

<div class="image-container">
  <h3>Figure 2: VQ-VAE Validation PSNR vs. Epochs (K &isin; {{512, 256, 128}})</h3>
  {"<img src='data:image/png;base64," + b64_vqvae_psnr_chart + "' style='max-width:80%;' />" if b64_vqvae_psnr_chart else "<p>[vqvae_psnr_vs_epochs.png]</p>"}
  <div class="caption">Reconstruction quality progression across 50 epochs for all three codebook capacities.</div>
</div>

<h2>4. Part B: WGAN vs. WGAN-GP Comparative Analysis</h2>

<h3>4.1 Weight Clipping vs. Gradient Penalty Dynamics</h3>
<p>Both models minimize the Wasserstein distance via the Kantorovich-Rubinstein duality:</p>
<div class="formula">
  min<sub>G</sub> max<sub>D &isin; D<sub>L</sub></sub> E<sub>x ~ P<sub>r</sub></sub>[D(x)] &minus; E<sub>x&#771; ~ P<sub>g</sub></sub>[D(x&#771;)]
</div>
<ul>
  <li><strong>Standard WGAN:</strong> Clips weights into [-0.01, 0.01] with RMSprop (lr=5e-5). As shown in the loss curve, clipping causes the critic loss to hover near zero and limits the critic from learning rich representations.</li>
  <li><strong>WGAN-GP:</strong> Enforces 1-Lipschitz continuity via the gradient penalty term:
    <div class="formula">
      L<sub>GP</sub> = &lambda; &middot; E<sub>x&#770;</sub>[(&#8741;&#8711;<sub>x&#770;</sub> D(x&#770;)&#8741;<sub>2</sub> &minus; 1)<sup>2</sup>], &nbsp; with &lambda; = 10
    </div>
    evaluated along random interpolations between real and generated images.
  </li>
</ul>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>1-Lipschitz Constraint</th>
      <th>Optimizer</th>
      <th>Inception FID (&darr;)</th>
      <th>Training Time (s)</th>
      <th>Stability &amp; Visual Sharpness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>WGAN</strong></td>
      <td>Weight Clipping [-0.01, 0.01]</td>
      <td>RMSprop (lr=5e-5)</td>
      <td class="metric-val">106.65</td>
      <td class="metric-val">2520.30 s</td>
      <td>Weight saturation; softer features.</td>
    </tr>
    <tr class="highlight">
      <td><strong>WGAN-GP</strong></td>
      <td>Gradient Penalty (&lambda;=10)</td>
      <td>Adam (&beta;<sub>1</sub>=0, &beta;<sub>2</sub>=0.9)</td>
      <td class="metric-val">48.71</td>
      <td class="metric-val">2730.62 s</td>
      <td><strong>54.3% lower FID; sharp boundaries; highly stable.</strong></td>
    </tr>
  </tbody>
</table>

<div class="image-container">
  <h3>Figure 3: WGAN vs. WGAN-GP Critic Convergence Loss Curves</h3>
  {svg_chart}
  <div class="caption">Critic loss trajectories across 100 epochs demonstrating stable Wasserstein estimation in WGAN-GP vs. bounded clipping in WGAN.</div>
</div>

<div class="page-break"></div>

<h3>4.2 WGAN &amp; WGAN-GP Training Progression (Epochs 10, 50, 100)</h3>
<p>Visual progression illustrating how sample quality evolves over 100 epochs:</p>

<p><strong>WGAN-GP Epoch Evolution:</strong></p>
<div class="img-grid-3">
  <div>
    {"<img src='data:image/png;base64," + b64_wgangp_ep10 + "' />" if b64_wgangp_ep10 else ""}
    <div class="caption" style="text-align:center;">WGAN-GP Epoch 10</div>
  </div>
  <div>
    {"<img src='data:image/png;base64," + b64_wgangp_ep50 + "' />" if b64_wgangp_ep50 else ""}
    <div class="caption" style="text-align:center;">WGAN-GP Epoch 50</div>
  </div>
  <div>
    {"<img src='data:image/png;base64," + b64_wgangp_ep100 + "' />" if b64_wgangp_ep100 else ""}
    <div class="caption" style="text-align:center;">WGAN-GP Epoch 100 (FID 48.71)</div>
  </div>
</div>

<p><strong>WGAN Epoch Evolution:</strong></p>
<div class="img-grid-3">
  <div>
    {"<img src='data:image/png;base64," + b64_wgan_ep10 + "' />" if b64_wgan_ep10 else ""}
    <div class="caption" style="text-align:center;">WGAN Epoch 10</div>
  </div>
  <div>
    {"<img src='data:image/png;base64," + b64_wgan_ep50 + "' />" if b64_wgan_ep50 else ""}
    <div class="caption" style="text-align:center;">WGAN Epoch 50</div>
  </div>
  <div>
    {"<img src='data:image/png;base64," + b64_wgan_ep100 + "' />" if b64_wgan_ep100 else ""}
    <div class="caption" style="text-align:center;">WGAN Epoch 100 (FID 106.65)</div>
  </div>
</div>

<h2>5. Global Comparative Discussion: Error, Quality &amp; Speed</h2>

<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>Reconstruction PSNR</th>
      <th>Sampling FID (&darr;)</th>
      <th>Sampling Latency</th>
      <th>Training Convergence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>&beta;-VAE (&beta;=1)</strong></td>
      <td>18.14 dB (Medium)</td>
      <td>164.38 (Blurry)</td>
      <td>Single pass (&lt; 0.05 ms)</td>
      <td>Fastest, monotonically stable</td>
    </tr>
    <tr>
      <td><strong>VQ-VAE (K=256) + PixelCNN</strong></td>
      <td><strong>23.82 dB (Superior)</strong></td>
      <td>117.83 (Structured)</td>
      <td>Autoregressive (64 sequential steps)</td>
      <td>Two-stage stable optimization</td>
    </tr>
    <tr>
      <td><strong>WGAN</strong></td>
      <td>N/A (Implicit density)</td>
      <td>106.65 (Moderate)</td>
      <td>Single pass (&lt; 0.05 ms)</td>
      <td>Sensitive to clipping bound</td>
    </tr>
    <tr class="highlight">
      <td><strong>WGAN-GP</strong></td>
      <td>N/A (Implicit density)</td>
      <td><strong>48.71 (Best Quality)</strong></td>
      <td>Single pass (&lt; 0.05 ms)</td>
      <td>Highly robust convergence</td>
    </tr>
  </tbody>
</table>

<h3>5.1 Comparative Analysis</h3>
<ol>
  <li><strong>Reconstruction Error:</strong> VQ-VAE decisively outperforms &beta;-VAE (23.82 dB vs 18.14 dB). The discrete codebook bottleneck eliminates the continuous Gaussian prior mismatch, preserving fine object contours and color textures.</li>
  <li><strong>Sampling Quality:</strong> WGAN-GP achieves the best generative realism with an FID of <strong>48.71</strong>, reducing FID by 54.3% over WGAN and 70.4% over &beta;-VAE. Adversarial feedback penalizes unrealistic smoothing that pixel-level MSE losses cannot resolve.</li>
  <li><strong>Sampling Latency &amp; Speed:</strong> While VQ-VAE offers excellent reconstructions, sampling from its prior requires sequential 8&times;8 autoregressive PixelCNN prediction (64 passes per image). In contrast, WGAN-GP and &beta;-VAE synthesize entire image batches in a single parallel feedforward pass.</li>
</ol>

<div class="page-break"></div>

<h2>6. Visual Evidence: Required 100-Sample Grids</h2>

<div class="image-container">
  <h3>Figure 4: WGAN-GP Generated Samples (100 Samples, 100 Epochs, FID = 48.71)</h3>
  {"<img src='data:image/png;base64," + b64_wgangp_samp + "' />" if b64_wgangp_samp else "<p>wgangp_samples_100.png</p>"}
  <div class="caption">WGAN-GP 10&times;10 sample grid showing crisp CIFAR-10 objects (vehicles, animals) with superior sharpness.</div>
</div>

<div class="image-container">
  <h3>Figure 5: WGAN Generated Samples (100 Samples, 100 Epochs, FID = 106.65)</h3>
  {"<img src='data:image/png;base64," + b64_wgan_samp + "' />" if b64_wgan_samp else "<p>wgan_samples_100.png</p>"}
  <div class="caption">WGAN 10&times;10 sample grid demonstrating weight clipping dynamics with softer image features.</div>
</div>

<div class="page-break"></div>

<div class="image-container">
  <h3>Figure 6: VQ-VAE (K=256) + PixelCNN Autoregressive Prior Samples (100 Samples, FID = 117.83)</h3>
  {"<img src='data:image/png;base64," + b64_vqvae_samp + "' />" if b64_vqvae_samp else "<p>vqvae_pixelcnn_samples_100.png</p>"}
  <div class="caption">Autoregressive prior samples synthesized via masked convolution over the 8&times;8 discrete codebook grid.</div>
</div>

<div class="image-container">
  <h3>Figure 7: &beta;-VAE (&beta;=1) Prior Samples (100 Samples, FID = 164.38)</h3>
  {"<img src='data:image/png;base64," + b64_vae_samp + "' />" if b64_vae_samp else "<p>vae_samples_100.png</p>"}
  <div class="caption">&beta;-VAE (&beta;=1) continuous prior Gaussian samples N(0, I) exhibiting smooth global features with typical MSE softening.</div>
</div>

<div class="page-break"></div>

<div class="image-container">
  <h3>Figure 8: VQ-VAE (K=256) Reconstructions (Top: Originals, Bottom: Reconstructions, PSNR = 23.82 dB)</h3>
  {"<img src='data:image/png;base64," + b64_vqvae_orig + "' style='max-width:48%; display:inline-block; margin-right:4px;' />" if b64_vqvae_orig else ""}
  {"<img src='data:image/png;base64," + b64_vqvae_rec + "' style='max-width:48%; display:inline-block;' />" if b64_vqvae_rec else ""}
  <div class="caption">Side-by-side comparison of 100 CIFAR-10 test originals (left) and VQ-VAE K=256 reconstructions (right).</div>
</div>

<div class="image-container">
  <h3>Figure 9: &beta;-VAE (&beta;=1) Reconstructions (Top: Originals, Bottom: Reconstructions, PSNR = 18.14 dB)</h3>
  {"<img src='data:image/png;base64," + b64_vae_orig + "' style='max-width:48%; display:inline-block; margin-right:4px;' />" if b64_vae_orig else ""}
  {"<img src='data:image/png;base64," + b64_vae_rec + "' style='max-width:48%; display:inline-block;' />" if b64_vae_rec else ""}
  <div class="caption">Side-by-side comparison of 100 CIFAR-10 test originals (left) and &beta;-VAE (&beta;=1) reconstructions (right).</div>
</div>

</body>
</html>
"""

# Write to root and to Deliverables/
with open("PartC_Comparative_Discussion_Report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

with open("Deliverables/PartC_Comparative_Discussion_Report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Comprehensive Part C Report with all charts and metrics generated successfully!")
