import os
from pathlib import Path

# Part A and Part B Metrics Summary Report Generator
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)

# Generate HTML report that can be directly converted or printed to PDF
html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>UDL Assignment 2 - Part C Comparative Discussion Report</title>
<style>
  body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; line-height: 1.6; color: #222; margin: 40px auto; max-width: 950px; }
  h1 { color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 8px; margin-bottom: 10px; }
  h2 { color: #2b6cb0; border-bottom: 1px solid #cbd5e0; padding-bottom: 6px; margin-top: 35px; }
  h3 { color: #2d3748; margin-top: 25px; }
  table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }
  th, td { border: 1px solid #cbd5e0; padding: 10px 14px; text-align: left; }
  th { background-color: #edf2f7; color: #2d3748; font-weight: 600; }
  tr:nth-child(even) { background-color: #f7fafc; }
  .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; background: #e2e8f0; }
  .highlight { background-color: #ebf8ff; font-weight: bold; }
  .info-box { background: #f0f4f8; border-left: 4px solid #3182ce; padding: 12px 18px; margin: 20px 0; border-radius: 0 4px 4px 0; }
  .metric-val { font-family: 'Courier New', monospace; font-weight: bold; }
  img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 15px 0; }
  @media print {
    body { max-width: 100%; margin: 20mm; }
    h2 { page-break-before: always; }
    h2:first-of-type { page-break-before: avoid; }
  }
</style>
</head>
<body>

<h1>UDL Assignment 2: Part C Comparative Discussion Report</h1>
<div class="info-box">
  <p><strong>Course:</strong> AIMLZG533 – Unsupervised Deep Learning<br>
  <strong>Student Name:</strong> Siddharth Raj<br>
  <strong>BITS ID / Username:</strong> s2025ae05706<br>
  <strong>Dataset:</strong> CIFAR-10 (50,000 train images, 10,000 test images, 32&times;32 RGB)<br>
  <strong>Evaluation Protocol:</strong> Torchvision Inception-v3 features resized to 299&times;299, 5,000 samples for VAE/GAN, 1,000 for autoregressive prior.</p>
</div>

<h2>1. Infrastructure Verification (BITS WILP Prayogshala)</h2>
<p>In compliance with the assignment instructions, experimentation and environment setup were conducted on the BITS WILP Prayogshala infrastructure under user account <code>s2025ae05706</code> for course <em>AIMLZG533 – Unsupervised Deep Learning</em>. Accelerated GPU training was performed using NVIDIA CUDA infrastructure to satisfy the required 100 epochs for WGAN/WGAN-GP and 50 epochs for VAE/VQ-VAE.</p>

<h2>2. Part A: &beta;-VAE Analysis (&beta; &isin; {1, 2, 4, 10})</h2>

<h3>2.1 Rate-Distortion &amp; Disentanglement Trade-Off</h3>
<p>The &beta;-VAE objective augments standard variational autoencoding with an adjustable scalar weighting &beta; on the Kullback-Leibler (KL) divergence term:</p>
<p style="text-align: center; font-style: italic;">
  L(&theta;, &phi;; x, z, &beta;) = E<sub>q<sub>&phi;</sub>(z|x)</sub>[log p<sub>&theta;</sub>(x|z)] - &beta; &middot; D<sub>KL</sub>(q<sub>&phi;</sub>(z|x) || p(z))
</p>
<p>where p(z) = N(0, I) acts as an isotropic Gaussian prior. As &beta; increases from 1 to 10, the latent bottleneck is penalized progressively harder:</p>
<ul>
  <li><strong>&beta; = 1 (Standard VAE):</strong> Prioritizes reconstruction fidelity. The latent space retains high informational capacity, achieving the highest PSNR (17.33 dB) and the best FID (164.38).</li>
  <li><strong>&beta; = 2, 4, 10 (Strict Regularization):</strong> Forces latent factors toward mutual statistical independence (disentanglement). However, this strong pressure causes posterior collapse across redundant latent dimensions, increasing pixel blurriness and raising the FID score to 314.80 for &beta;=10.</li>
</ul>

<h3>2.2 Quantitative Results Table</h3>
<table>
  <thead>
    <tr>
      <th>Model Variant</th>
      <th>&beta; Weight</th>
      <th>Reconstruction PSNR (dB)</th>
      <th>Inception FID (&darr;)</th>
      <th>Observations</th>
    </tr>
  </thead>
  <tbody>
    <tr class="highlight">
      <td><strong>&beta;-VAE (&beta;=1)</strong></td>
      <td>1.0</td>
      <td class="metric-val">17.33</td>
      <td class="metric-val">164.38</td>
      <td>Sharpest reconstructions; best visual sample diversity.</td>
    </tr>
    <tr>
      <td><strong>&beta;-VAE (&beta;=2)</strong></td>
      <td>2.0</td>
      <td class="metric-val">16.12</td>
      <td class="metric-val">215.19</td>
      <td>Slightly softer edges; balanced KL regularization.</td>
    </tr>
    <tr>
      <td><strong>&beta;-VAE (&beta;=4)</strong></td>
      <td>4.0</td>
      <td class="metric-val">15.40</td>
      <td class="metric-val">268.77</td>
      <td>Noticeable blurring; higher sample distortion.</td>
    </tr>
    <tr>
      <td><strong>&beta;-VAE (&beta;=10)</strong></td>
      <td>10.0</td>
      <td class="metric-val">14.87</td>
      <td class="metric-val">314.80</td>
      <td>Severe regularization penalty; blurred reconstructed output.</td>
    </tr>
  </tbody>
</table>

<h2>3. Part A: VQ-VAE &amp; Autoregressive PixelCNN Prior</h2>

<h3>3.1 Vector Quantization Dynamics &amp; Codebook Utilization</h3>
<p>VQ-VAE discretizes continuous latent vectors into discrete codebook indices e<sub>k</sub> &isin; R<sup>D</sup> via nearest-neighbor mapping with straight-through gradient estimation. We evaluated codebook capacities K &isin; {512, 256, 128} with latent channel depth D = 64 across an 8&times;8 spatial feature grid.</p>

<table>
  <thead>
    <tr>
      <th>Codebook Size (K)</th>
      <th>Recon MSE Loss</th>
      <th>Validation PSNR (dB)</th>
      <th>Active Codes</th>
      <th>Utilization %</th>
      <th>Perplexity</th>
      <th>PixelCNN Prior FID (&darr;)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>K = 512</strong></td>
      <td class="metric-val">0.00431</td>
      <td class="metric-val">24.10</td>
      <td class="metric-val">199 / 512</td>
      <td class="metric-val">38.87%</td>
      <td class="metric-val">169.50</td>
      <td class="metric-val">114.46</td>
    </tr>
    <tr class="highlight">
      <td><strong>K = 256</strong></td>
      <td class="metric-val">0.00459</td>
      <td class="metric-val">23.82</td>
      <td class="metric-val">159 / 256</td>
      <td class="metric-val">62.11%</td>
      <td class="metric-val">132.22</td>
      <td class="metric-val">117.83</td>
    </tr>
    <tr>
      <td><strong>K = 128</strong></td>
      <td class="metric-val">0.00470</td>
      <td class="metric-val">23.70</td>
      <td class="metric-val">128 / 128</td>
      <td class="metric-val">100.00%</td>
      <td class="metric-val">112.48</td>
      <td class="metric-val">116.39</td>
    </tr>
  </tbody>
</table>

<p><strong>Key Insight:</strong> 
As K decreases from 512 to 128, codebook utilization increases monotonically from 38.87% to a full 100.00%, eliminating dead embeddings. Although K=512 achieves marginally superior reconstruction PSNR (24.10 dB vs 23.70 dB), K=256 achieves the best practical trade-off between expressive capacity and compact representation, making autoregressive modeling with PixelCNN robust and less susceptible to overfitting.</p>

<h2>4. Part B: WGAN vs. WGAN-GP Comparative Analysis</h2>

<h3>4.1 Weight Clipping vs. Gradient Penalty (1-Lipschitz Enforcement)</h3>
<p>To enforce the 1-Lipschitz continuity constraint required by the Kantorovich-Rubinstein duality:</p>
<ul>
  <li><strong>WGAN (Weight Clipping):</strong> Clamps critic parameters directly into [-0.01, 0.01]. This crude constraint causes critic weights to saturate at the extremes, underutilizing network capacity and restricting gradient flow across layers.</li>
  <li><strong>WGAN-GP (Gradient Penalty):</strong> Replaces hard clipping with a soft gradient penalty evaluated along straight lines between real and generated distributions:
    <div style="text-align: center; font-style: italic; margin: 10px 0;">
      L<sub>GP</sub> = E<sub>x&#770;</sub>[(&#8741;&#8711;<sub>x&#770;</sub> D(x&#770;)&#8741;<sub>2</sub> - 1)<sup>2</sup>]
    </div>
    This allows the critic to learn complex, non-trivial scoring landscapes without weight saturation.
  </li>
</ul>

<h3>4.2 Quantitative Comparison</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Constraint Mechanism</th>
      <th>Optimizer</th>
      <th>Inception FID (&darr;)</th>
      <th>Training Time (s)</th>
      <th>Convergence &amp; Stability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>WGAN</strong></td>
      <td>Weight Clipping [-0.01, 0.01]</td>
      <td>RMSprop (lr=5e-5)</td>
      <td class="metric-val">106.65</td>
      <td class="metric-val">2520.30 s</td>
      <td>Prone to capacity saturation; blurry edges.</td>
    </tr>
    <tr class="highlight">
      <td><strong>WGAN-GP</strong></td>
      <td>Gradient Penalty (&lambda;=10)</td>
      <td>Adam (&beta;<sub>1</sub>=0, &beta;<sub>2</sub>=0.9)</td>
      <td class="metric-val">48.71</td>
      <td class="metric-val">2730.62 s</td>
      <td>Highly stable; 54.3% reduction in FID; crisp details.</td>
    </tr>
  </tbody>
</table>

<h2>5. Global Comparative Discussion: Error, Quality &amp; Speed</h2>

<table>
  <thead>
    <tr>
      <th>Architecture Family</th>
      <th>Model Variant</th>
      <th>Reconstruction Quality (PSNR)</th>
      <th>Sampling Quality (FID)</th>
      <th>Inference Speed</th>
      <th>Training Stability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Continuous VAE</strong></td>
      <td>&beta;-VAE (&beta;=1)</td>
      <td>17.33 dB (Medium)</td>
      <td>164.38 (High blur)</td>
      <td>Very Fast (1 forward pass)</td>
      <td>Extremely Stable</td>
    </tr>
    <tr>
      <td><strong>Discrete VAE</strong></td>
      <td>VQ-VAE (K=256) + PixelCNN</td>
      <td><strong>23.82 dB</strong> (Very High)</td>
      <td>117.83 (Coherent structure)</td>
      <td>Slow (Sequential 8&times;8 autoregression)</td>
      <td>High (Separate stages)</td>
    </tr>
    <tr>
      <td><strong>Adversarial (Clip)</strong></td>
      <td>WGAN</td>
      <td>N/A (Implicit generator)</td>
      <td>106.65 (Moderate)</td>
      <td><strong>Fastest</strong> (Direct feedforward)</td>
      <td>Moderate (Weight clipping)</td>
    </tr>
    <tr class="highlight">
      <td><strong>Adversarial (Penalty)</strong></td>
      <td>WGAN-GP</td>
      <td>N/A (Implicit generator)</td>
      <td><strong>48.71</strong> (Superior sharpness)</td>
      <td><strong>Fastest</strong> (Direct feedforward)</td>
      <td>Extremely Stable (Lipschitz GP)</td>
    </tr>
  </tbody>
</table>

<h3>5.1 Detailed Dimension Comparisons</h3>
<ol>
  <li><strong>Reconstruction Error:</strong> VQ-VAE decisively outperforms standard &beta;-VAE (23.82 dB vs 17.33 dB). By quantizing onto a discrete codebook rather than a smooth continuous Gaussian, VQ-VAE avoids posterior collapse and preserves high-frequency edges and color contrasts.</li>
  <li><strong>Sampling Quality:</strong> WGAN-GP produces the best visual fidelity (FID 48.71 vs VQ-VAE FID 117.83 and VAE FID 164.38). Adversarial training directly penalizes visual artifacts that mean-squared-error objectives inherently smooth out.</li>
  <li><strong>Inference &amp; Sampling Speed:</strong>
    <ul>
      <li>WGAN / WGAN-GP and &beta;-VAE generate samples in a single forward pass (&lt; 0.05 ms per batch).</li>
      <li>PixelCNN sampling is sequential across all 64 latent positions (8&times;8 grid), requiring 64 sequential neural evaluations, resulting in significantly higher sampling latency.</li>
    </ul>
  </li>
</ol>

<h2>6. Summary of Required Deliverables</h2>
<ul>
  <li><strong>Deliverable 1 (Notebooks):</strong> <code>PartA_VAE_VQVAE_PixelCNN.ipynb</code> and <code>PartB_WGAN_WGANGP.ipynb</code></li>
  <li><strong>Deliverable 2 (HTML files):</strong> <code>PartA_VAE_VQVAE_PixelCNN.html</code> and <code>PartB_WGAN_WGANGP.html</code></li>
  <li><strong>Deliverable 3 (PDF Report):</strong> <code>PartC_Comparative_Discussion_Report.pdf</code> (Generated from this document)</li>
</ul>

</body>
</html>
"""

with open("PartC_Comparative_Discussion_Report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Part C HTML report created successfully: PartC_Comparative_Discussion_Report.html")
