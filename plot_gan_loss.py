import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

w_hist = json.load(open('outputs/wgan_history.json'))
gp_hist = json.load(open('outputs/wgangp_history.json'))

plt.figure(figsize=(9, 5))
plt.plot([r['epoch'] for r in w_hist], [r['critic_loss'] for r in w_hist], label='WGAN Critic Loss (Weight Clipping)', color='#e53e3e')
plt.plot([r['epoch'] for r in gp_hist], [r['critic_loss'] for r in gp_hist], label='WGAN-GP Critic Loss (Gradient Penalty)', color='#3182ce')
plt.xlabel('Epoch')
plt.ylabel('Critic Loss')
plt.title('WGAN vs. WGAN-GP Critic Convergence Loss Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('outputs/gan_loss_comparison.png', dpi=160, bbox_inches='tight')
plt.close()
print("Saved outputs/gan_loss_comparison.png")

