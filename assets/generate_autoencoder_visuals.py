import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Ellipse, Polygon
import numpy as np

# Set global style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Common colors
C_BG = '#ffffff'
C_BOX = '#f8f9fa'
C_BORDER = '#2c3e50'
C_ENCODER = '#d4e6f1' # Soft Blue
C_DECODER = '#d5f5e3' # Soft Green
C_LATENT = '#fdebd0'  # Soft Orange / Amber
C_NOISE = '#fadbd8'   # Soft Red
C_CODEBOOK = '#e8daef' # Soft Purple

def draw_arrow(ax, start, end, color='#2c3e50', lw=2, text=None, text_offset=(0, 0.2), ls='-'):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, linestyle=ls, zorder=3)
    ax.add_patch(arrow)
    if text:
        mx, my = (start[0]+end[0])/2 + text_offset[0], (start[1]+end[1])/2 + text_offset[1]
        ax.text(mx, my, text, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='#bdc3c7', boxstyle='round,pad=0.2', alpha=0.9), zorder=4)

def draw_box(ax, x, y, w, h, text, bg, fontsize=10, text_color='#1a252f', lw=1.5, edgecolor=C_BORDER):
    rect = Rectangle((x, y), w, h, facecolor=bg, edgecolor=edgecolor, lw=lw, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=3)


# ==============================================================================
# 1. CLASSICAL AUTOENCODER ARCHITECTURE
# ==============================================================================
def generate_autoencoder_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(7, 5.5, "Classical Autoencoder Bottleneck Architecture", ha='center', fontsize=15, fontweight='bold', color='#1a252f')

    # Input x
    draw_box(ax, 0.5, 1.5, 1.8, 3.0, "Input Vector\nx ∈ R^D\n(e.g., D=784)", '#eaeded', fontsize=11)

    # Encoder
    enc_poly = Polygon([(2.8, 0.8), (5.2, 1.8), (5.2, 4.2), (2.8, 5.2)], facecolor=C_ENCODER, edgecolor=C_BORDER, lw=1.5, zorder=2)
    ax.add_patch(enc_poly)
    ax.text(4.0, 3.0, "Encoder f_θ(x)\nz = σ(W_e x + b_e)", ha='center', va='center', fontsize=10, fontweight='bold', color='#1a5276')

    # Bottleneck Latent z
    draw_box(ax, 5.7, 2.0, 1.6, 2.0, "Bottleneck\nz ∈ R^d\n(d << D\ne.g., d=32)", C_LATENT, fontsize=10, edgecolor='#b9770e')

    # Decoder
    dec_poly = Polygon([(7.8, 1.8), (10.2, 0.8), (10.2, 5.2), (7.8, 4.2)], facecolor=C_DECODER, edgecolor=C_BORDER, lw=1.5, zorder=2)
    ax.add_patch(dec_poly)
    ax.text(9.0, 3.0, "Decoder g_ϕ(z)\nx̂ = σ(W_d z + b_d)", ha='center', va='center', fontsize=10, fontweight='bold', color='#196f3d')

    # Reconstruction x̂
    draw_box(ax, 10.7, 1.5, 1.8, 3.0, "Reconstruction\nx̂ ∈ R^D\n(x̂ ≈ x)", '#eaeded', fontsize=11)

    # Connecting Arrows
    draw_arrow(ax, (2.3, 3.0), (2.8, 3.0))
    draw_arrow(ax, (5.2, 3.0), (5.7, 3.0))
    draw_arrow(ax, (7.3, 3.0), (7.8, 3.0))
    draw_arrow(ax, (10.2, 3.0), (10.7, 3.0))

    # Loss Loop Backprop
    draw_arrow(ax, (11.6, 1.3), (11.6, 0.5), color='#c0392b', lw=2, ls='--')
    draw_arrow(ax, (11.6, 0.5), (1.4, 0.5), color='#c0392b', lw=2, ls='--', text="Reconstruction Loss: L(x, x̂) = 1/2 ||x - x̂||_2^2")
    draw_arrow(ax, (1.4, 0.5), (1.4, 1.3), color='#c0392b', lw=2, ls='--')

    plt.tight_layout()
    plt.savefig('assets/autoencoder_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/autoencoder_architecture.png")


# ==============================================================================
# 2. PCA VS. NON-LINEAR AUTOENCODER MANIFOLD
# ==============================================================================
def generate_pca_vs_autoencoder_manifold():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(C_BG)

    np.random.seed(42)
    # Generate non-linear curved data (S-curve / parabola)
    t = np.linspace(-3, 3, 200)
    x = t + np.random.normal(0, 0.2, 200)
    y = 0.5 * t**2 + np.random.normal(0, 0.3, 200) - 2

    # Left: PCA (Linear Subspace)
    ax1.scatter(x, y, c='#7f8c8d', alpha=0.5, s=20, label='Data points')
    # Linear PCA line
    pca_x = np.linspace(-3.5, 3.5, 100)
    pca_y = 0.3 * pca_x - 0.5
    ax1.plot(pca_x, pca_y, color='#c0392b', lw=3, label='PCA Linear Subspace (1st PC)')
    
    # Orthogonal projection lines for a few points
    for idx in [20, 60, 100, 140, 180]:
        px, py = x[idx], y[idx]
        proj_x = (px + 0.3*(py + 0.5)) / (1 + 0.3**2)
        proj_y = 0.3 * proj_x - 0.5
        ax1.plot([px, proj_x], [py, proj_y], 'r--', lw=1.2, alpha=0.8)

    ax1.set_title("Linear PCA: Rigid Orthogonal Projection", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Feature $x_1$", fontweight='bold')
    ax1.set_ylabel("Feature $x_2$", fontweight='bold')
    ax1.legend(loc='upper center', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.text(0, -3.2, "[X] Fails to capture non-linear curvature\n(High reconstruction error on curve)", 
             ha='center', fontsize=10, color='#922b21', bbox=dict(facecolor='#fdedec', edgecolor='#e6b0aa', boxstyle='round,pad=0.4'))

    # Right: Non-linear Autoencoder Manifold
    ax2.scatter(x, y, c='#7f8c8d', alpha=0.5, s=20, label='Data points')
    # Non-linear learned curve
    curve_x = np.linspace(-3.2, 3.2, 200)
    curve_y = 0.5 * curve_x**2 - 2
    ax2.plot(curve_x, curve_y, color='#27ae60', lw=3.5, label='Autoencoder Learned 1D Manifold')

    # Non-linear projection lines
    for idx in [20, 60, 100, 140, 180]:
        px, py = x[idx], y[idx]
        t_val = px
        proj_x = t_val
        proj_y = 0.5 * t_val**2 - 2
        ax2.plot([px, proj_x], [py, proj_y], 'g--', lw=1.2, alpha=0.8)

    ax2.set_title("Non-Linear Autoencoder: Flexible Curved Manifold", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Feature $x_1$", fontweight='bold')
    ax2.set_ylabel("Feature $x_2$", fontweight='bold')
    ax2.legend(loc='upper center', framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.text(0, -3.2, "[OK] Non-linear activations fit true data manifold\n(Minimum reconstruction error)", 
             ha='center', fontsize=10, color='#196f3d', bbox=dict(facecolor='#eafaf1', edgecolor='#a9dfbf', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig('assets/pca_vs_autoencoder_manifold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/pca_vs_autoencoder_manifold.png")


# ==============================================================================
# 3. REGULARIZED AUTOENCODER FAMILIES
# ==============================================================================
def generate_regularized_autoencoders():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(C_BG)

    # 1. Sparse Autoencoder
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis('off')
    ax1.set_title("Sparse Autoencoders (SAE)\n(Mechanistic Interpretability in LLMs)", fontsize=12, fontweight='bold', pad=10)
    draw_box(ax1, 0.5, 3.5, 2.2, 3.0, "Dense LLM\nActivation x\n(d=4096)", '#eaeded', fontsize=9)
    draw_box(ax1, 3.8, 1.0, 2.4, 8.0, "Overcomplete Latent z\n(Dim = 65,536)\n\n• Feature 12 (0)\n• Feature 42 (1.8: 'Golden Gate')\n• Feature 105 (0)\n• Feature 819 (2.4: 'Python')\n• Feature 999 (0)", C_LATENT, fontsize=8, edgecolor='#b9770e')
    draw_box(ax1, 7.3, 3.5, 2.2, 3.0, "Reconstructed\nActivation x̂\n(d=4096)", '#eaeded', fontsize=9)
    draw_arrow(ax1, (2.7, 5.0), (3.8, 5.0), text="Encoder\n+ L_1 Penalty")
    draw_arrow(ax1, (6.2, 5.0), (7.3, 5.0), text="Decoder")
    ax1.text(5.0, 0.3, "L = ||x - x̂||_2^2 + λ ∑ |z_i|\n(Disentangles Superposition!)", ha='center', fontsize=9, fontweight='bold', color='#7e5109')

    # 2. Denoising Autoencoder & Score Field
    ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3)
    ax2.set_title("Denoising Autoencoder (DAE)\n(Score Matching & Diffusion Basis)", fontsize=12, fontweight='bold', pad=10)
    m_x = np.linspace(-2.5, 2.5, 100)
    m_y = 0.5 * m_x
    ax2.plot(m_x, m_y, color='#2980b9', lw=3, label='Clean Data Manifold M')
    
    x_clean = np.array([0.5, 0.25])
    x_noisy = np.array([-1.2, 1.8])
    ax2.scatter([x_clean[0]], [x_clean[1]], color='#27ae60', s=100, zorder=5, label='Clean Data Point x')
    ax2.scatter([x_noisy[0]], [x_noisy[1]], color='#c0392b', s=100, zorder=5, label='Corrupted Point x̃ = x + ε')
    
    arrow = FancyArrowPatch(x_noisy, x_clean, arrowstyle='-|>', mutation_scale=20, color='#8e44ad', lw=3, zorder=4)
    ax2.add_patch(arrow)
    ax2.text(-0.2, 1.2, "Learned Vector Field:\ng(f(x̃)) - x̃ ≈ σ^2 ∇_x log p(x)\n(Points to Manifold!)", 
             fontsize=9, fontweight='bold', color='#5b2c6f', bbox=dict(facecolor='#f4ecf7', edgecolor='#d2b4de', boxstyle='round,pad=0.3'))
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # 3. Contractive Autoencoder
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis('off')
    ax3.set_title("Contractive Autoencoders (CAE)\n(Local Manifold Invariance)", fontsize=12, fontweight='bold', pad=10)
    draw_box(ax3, 1.0, 3.0, 2.5, 4.0, "Input Space\nx ∈ R^D\n\nLocal Noise ε\nAround Point x", '#eaeded', fontsize=9)
    draw_box(ax3, 6.5, 3.0, 2.5, 4.0, "Latent Space\nz = f(x)\n\nContracted / Invariant\nto Orthogonal Noise", C_ENCODER, fontsize=9, edgecolor='#2980b9')
    draw_arrow(ax3, (3.5, 5.0), (6.5, 5.0), color='#2980b9', lw=2.5, text="Jacobian Penalty\n||J_f(x)||_F^2")
    ax3.text(5.0, 1.5, "L = L_recon + λ ∑_{i,j} (∂z_i / ∂x_j)^2\n(Flattens non-manifold directions!)", 
             ha='center', fontsize=9, fontweight='bold', color='#1b4f72', bbox=dict(facecolor='#ebf5fb', edgecolor='#aed6f1', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig('assets/regularized_autoencoders.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/regularized_autoencoders.png")


# ==============================================================================
# 4. DETERMINISTIC AE VS. VAE LATENT SPACE
# ==============================================================================
def generate_vae_latent_space_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(C_BG)

    np.random.seed(101)

    # Left: Deterministic AE Latent Space
    c1 = np.random.normal(loc=[-2.5, 2.0], scale=0.3, size=(40, 2))
    c2 = np.random.normal(loc=[2.5, -2.0], scale=0.3, size=(40, 2))
    c3 = np.random.normal(loc=[2.0, 2.2], scale=0.35, size=(40, 2))

    ax1.scatter(c1[:, 0], c1[:, 1], c='#3498db', s=35, label='Class "1"')
    ax1.scatter(c2[:, 0], c2[:, 1], c='#e67e22', s=35, label='Class "7"')
    ax1.scatter(c3[:, 0], c3[:, 1], c='#2ecc71', s=35, label='Class "9"')

    dead_zone = Ellipse((0, 0), width=3.0, height=2.5, angle=20, facecolor='#fadbd8', edgecolor='#e74c3c', linestyle='--', lw=1.5, alpha=0.6)
    ax1.add_patch(dead_zone)
    ax1.text(0, 0, "[X] Uncharted 'Dead Zone'\n(Decoder outputs\ngarbled gibberish!)", ha='center', va='center', fontsize=9, fontweight='bold', color='#922b21')

    ax1.scatter([0], [0], c='#c0392b', marker='X', s=120, zorder=5)

    ax1.set_xlim(-4, 4); ax1.set_ylim(-4, 4)
    ax1.set_title("Deterministic AE: Discontinuous & Unbounded\n(Isolated Clusters with Empty Voids)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Latent Dimension $z_1$", fontweight='bold')
    ax1.set_ylabel("Latent Dimension $z_2$", fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Right: Variational AE Latent Space
    prior_circle = Circle((0, 0), radius=2.0, facecolor='#ebf5fb', edgecolor='#2980b9', linestyle='-', lw=2, alpha=0.4, label='Prior $p(z) = \\mathcal{N}(0, I)$')
    ax2.add_patch(prior_circle)

    for loc, color, label in [([-0.9, 0.8], '#3498db', 'Digit "1" $q(z|x)$'), 
                              ([0.9, -0.7], '#e67e22', 'Digit "7" $q(z|x)$'), 
                              ([0.7, 0.9], '#2ecc71', 'Digit "9" $q(z|x)$')]:
        pts = np.random.normal(loc=loc, scale=0.45, size=(40, 2))
        ax2.scatter(pts[:, 0], pts[:, 1], c=color, s=35, label=label, alpha=0.8)
        ell = Ellipse(loc, width=1.5, height=1.5, facecolor='none', edgecolor=color, lw=1.8, linestyle='--')
        ax2.add_patch(ell)

    draw_arrow(ax2, (-0.8, 0.8), (0.8, -0.7), color='#8e44ad', lw=2.5, text="Smooth Generative\nInterpolation (1 -> 7)")

    ax2.set_xlim(-4, 4); ax2.set_ylim(-4, 4)
    ax2.set_title("Variational AE (VAE): Smooth & Continuous\n(Isotropic Standard Normal Manifold $\\mathcal{N}(0, I)$)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("Latent Dimension $z_1$", fontweight='bold')
    ax2.set_ylabel("Latent Dimension $z_2$", fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('assets/vae_latent_space_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/vae_latent_space_comparison.png")


# ==============================================================================
# 5. VAE REPARAMETERIZATION TRICK
# ==============================================================================
def generate_vae_reparameterization_trick():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(C_BG)

    # 1. Non-differentiable (Stochastic Node)
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 8); ax1.axis('off')
    ax1.set_title("[X] Standard Stochastic Sampling (Breaks Backprop)", fontsize=12, fontweight='bold', color='#922b21', pad=12)
    
    draw_box(ax1, 0.5, 3.0, 1.8, 2.0, "Input x", '#eaeded', fontsize=10)
    draw_box(ax1, 2.8, 2.5, 2.2, 3.0, "Encoder\nNeural Net ϕ\n(Weights W_ϕ)", C_ENCODER, fontsize=10)
    draw_box(ax1, 5.5, 4.2, 1.5, 1.3, "Mean\nμ(x)", C_LATENT, fontsize=9)
    draw_box(ax1, 5.5, 2.5, 1.5, 1.3, "Std Dev\nσ(x)", C_LATENT, fontsize=9)
    
    stoch_circle = Circle((7.8, 4.0), radius=0.8, facecolor=C_NOISE, edgecolor='#c0392b', lw=2, zorder=2)
    ax1.add_patch(stoch_circle)
    ax1.text(7.8, 4.0, "Sample\nz ~ N(μ, σ^2)", ha='center', va='center', fontsize=8, fontweight='bold', color='#922b21', zorder=3)

    draw_box(ax1, 8.8, 3.0, 1.0, 2.0, "Decoder\ng_θ(z)", C_DECODER, fontsize=9)

    draw_arrow(ax1, (2.3, 4.0), (2.8, 4.0))
    draw_arrow(ax1, (5.0, 4.8), (5.5, 4.8))
    draw_arrow(ax1, (5.0, 3.2), (5.5, 3.2))
    draw_arrow(ax1, (7.0, 4.8), (7.2, 4.4))
    draw_arrow(ax1, (7.0, 3.2), (7.2, 3.6))
    draw_arrow(ax1, (8.6, 4.0), (8.8, 4.0))

    ax1.text(7.8, 2.0, "[!] NO GRADIENTS!\n∂z / ∂ϕ is undefined for\nstochastic sampling", ha='center', fontsize=10, fontweight='bold', color='#c0392b',
             bbox=dict(facecolor='#fdedec', edgecolor='#e6b0aa', boxstyle='round,pad=0.4'))

    # 2. Differentiable (Reparameterization Trick)
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 8); ax2.axis('off')
    ax2.set_title("[OK] Reparameterized Graph (Continuous Gradient Flow)", fontsize=12, fontweight='bold', color='#196f3d', pad=12)

    draw_box(ax2, 0.5, 3.0, 1.5, 2.0, "Input x", '#eaeded', fontsize=10)
    draw_box(ax2, 2.4, 2.5, 2.0, 3.0, "Encoder ϕ\n(Parameters)", C_ENCODER, fontsize=10)
    draw_box(ax2, 4.8, 4.5, 1.4, 1.2, "μ(x)", C_LATENT, fontsize=9)
    draw_box(ax2, 4.8, 2.3, 1.4, 1.2, "σ(x)", C_LATENT, fontsize=9)

    draw_box(ax2, 6.7, 3.0, 1.8, 2.0, "Deterministic\nz = μ + σ ⊙ ε", '#d5f5e3', fontsize=9, edgecolor='#27ae60')
    draw_box(ax2, 8.8, 3.0, 1.0, 2.0, "Decoder\ng_θ(z)", C_DECODER, fontsize=9)

    draw_box(ax2, 6.7, 6.2, 1.8, 1.1, "Auxiliary Noise\nε ~ N(0, I)", C_NOISE, fontsize=8, edgecolor='#e74c3c')

    draw_arrow(ax2, (2.0, 4.0), (2.4, 4.0))
    draw_arrow(ax2, (4.4, 4.8), (4.8, 4.8))
    draw_arrow(ax2, (4.4, 3.2), (4.8, 3.2))
    draw_arrow(ax2, (6.2, 5.1), (6.7, 4.4), text="∂z/∂μ = 1", text_offset=(0, 0.3))
    draw_arrow(ax2, (6.2, 2.9), (6.7, 3.6), text="∂z/∂σ = ε", text_offset=(0, -0.3))
    draw_arrow(ax2, (7.6, 6.2), (7.6, 5.0))
    draw_arrow(ax2, (8.5, 4.0), (8.8, 4.0))

    draw_arrow(ax2, (8.8, 2.2), (2.4, 2.2), color='#27ae60', lw=2.5, ls='--', text="Backprop flows smoothly via Chain Rule: ∂L/∂ϕ = (∂L/∂z)(∂z/∂μ)(∂μ/∂ϕ) + (∂L/∂z)(∂z/∂σ)(∂σ/∂ϕ)")

    plt.tight_layout()
    plt.savefig('assets/vae_reparameterization_trick.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/vae_reparameterization_trick.png")


# ==============================================================================
# 6. VQ-VAE CODEBOOK QUANTIZATION & STE
# ==============================================================================
def generate_vq_vae_codebook_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7); ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(7.5, 6.5, "VQ-VAE: Vector Quantized Codebook & Straight-Through Estimator (STE)", ha='center', fontsize=14, fontweight='bold')

    draw_box(ax, 0.5, 2.2, 1.8, 2.5, "Input Image x\n(256x256x3)", '#eaeded', fontsize=10)
    draw_box(ax, 2.8, 2.0, 2.2, 3.0, "CNN / ViT\nEncoder f_θ", C_ENCODER, fontsize=10)
    draw_box(ax, 5.5, 2.2, 1.8, 2.5, "Continuous\nLatent Map\nz_e(x) ∈ R^D\n(32x32xD)", C_LATENT, fontsize=9, edgecolor='#b9770e')
    draw_box(ax, 8.0, 1.2, 2.4, 4.5, "Discrete Codebook E\n[e_1: Vector 1 ∈ R^D]\n[e_2: Vector 2 ∈ R^D]\n[...]\n[e_k: Nearest Vector]\n[...]\n[e_K: Vector K ∈ R^D]", C_CODEBOOK, fontsize=8, edgecolor='#8e44ad')
    draw_box(ax, 11.0, 2.2, 1.8, 2.5, "Quantized\nLatent Map\nz_q(x) = e_k\n(32x32xD)", '#d5f5e3', fontsize=9, edgecolor='#27ae60')
    draw_box(ax, 13.2, 2.0, 1.5, 3.0, "Decoder\ng_ϕ", C_DECODER, fontsize=10)

    draw_arrow(ax, (2.3, 3.5), (2.8, 3.5))
    draw_arrow(ax, (5.0, 3.5), (5.5, 3.5))
    draw_arrow(ax, (7.3, 3.5), (8.0, 3.5), text="ArgMin ||z_e - e_j||_2")
    draw_arrow(ax, (10.4, 3.5), (11.0, 3.5), text="Replace with e_k")
    draw_arrow(ax, (12.8, 3.5), (13.2, 3.5))

    ste_arrow = FancyArrowPatch((11.9, 4.8), (6.4, 4.8), connectionstyle="arc3,rad=-0.3", arrowstyle='-|>', mutation_scale=15, color='#c0392b', lw=2.5, linestyle='--')
    ax.add_patch(ste_arrow)
    ax.text(9.15, 5.8, "Straight-Through Estimator (STE):\nGradient copied directly: ∇_{z_e} L ≈ ∇_{z_q} L", 
            ha='center', fontsize=9, fontweight='bold', color='#922b21', bbox=dict(facecolor='#fdedec', edgecolor='#e6b0aa', boxstyle='round,pad=0.3'))

    ax.text(7.5, 0.5, "Total Loss: L = L_recon(x, x̂)  +  ||sg[z_e(x)] - e||_2^2 (Codebook Loss)  +  β ||z_e(x) - sg[e]||_2^2 (Commitment Loss)",
            ha='center', fontsize=10, fontweight='bold', color='#1a252f', bbox=dict(facecolor='#f8f9fa', edgecolor='#bdc3c7', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig('assets/vq_vae_codebook_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/vq_vae_codebook_pipeline.png")


# ==============================================================================
# 7. MASKED AUTOENCODER (MAE) ASYMMETRIC PIPELINE
# ==============================================================================
def generate_mae_vision_transformer_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7.5); ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(8.0, 7.0, "Masked Autoencoder (MAE) — Asymmetric Vision Transformer Pretraining", ha='center', fontsize=14, fontweight='bold')

    draw_box(ax, 0.5, 2.5, 2.0, 2.5, "Input Image\n(224x224)\n\n196 Patches\n(14x14 Grid)", '#eaeded', fontsize=10)
    draw_box(ax, 3.2, 4.0, 2.0, 2.0, "25% Visible\nPatches (49)\n[P1, P7, P19, ...]", '#d5f5e3', fontsize=9, edgecolor='#27ae60')
    draw_box(ax, 3.2, 1.2, 2.0, 2.0, "75% Masked\nPatches (147)\n[Discarded!]", '#fadbd8', fontsize=9, edgecolor='#e74c3c')
    draw_box(ax, 5.8, 3.5, 2.4, 3.0, "Heavy ViT Encoder\n(e.g., ViT-Huge 24L)\n\nProcesses ONLY\n49 Visible Tokens!\n(16x Faster Attention!)", C_ENCODER, fontsize=9, edgecolor='#2980b9')
    draw_box(ax, 8.8, 2.2, 2.4, 3.6, "Full Token Sequence\n\n49 Latent Tokens\n+\n147 Learnable\n[MASK] Tokens\n+\nPositional Encodings", C_LATENT, fontsize=9, edgecolor='#b9770e')
    draw_box(ax, 11.8, 2.5, 2.0, 3.0, "Lightweight\nViT Decoder\n(e.g., 8 Layers)\n\n<10% Compute!", C_DECODER, fontsize=9, edgecolor='#27ae60')
    draw_box(ax, 14.2, 2.5, 1.5, 3.0, "Output\nReconstructed\nPixels", '#eaeded', fontsize=9)

    draw_arrow(ax, (2.5, 3.75), (3.2, 5.0), text="Sample 25%")
    draw_arrow(ax, (2.5, 3.75), (3.2, 2.2), text="Mask 75%")
    draw_arrow(ax, (5.2, 5.0), (5.8, 5.0))
    draw_arrow(ax, (8.2, 5.0), (8.8, 4.5))
    draw_arrow(ax, (11.2, 4.0), (11.8, 4.0))
    draw_arrow(ax, (13.8, 4.0), (14.2, 4.0))

    ax.text(8.0, 0.6, "MSE Reconstruction Loss is computed EXCLUSIVELY on the 75% Masked Patches (Normalized Pixels)\nL_MAE = (1 / |M|) ∑_{i ∈ M} ||p_i - p̂_i||_2^2",
            ha='center', fontsize=10, fontweight='bold', color='#1a252f', bbox=dict(facecolor='#fef9e7', edgecolor='#f39c12', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig('assets/mae_vision_transformer_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/mae_vision_transformer_pipeline.png")

if __name__ == '__main__':
    generate_autoencoder_architecture()
    generate_pca_vs_autoencoder_manifold()
    generate_regularized_autoencoders()
    generate_vae_latent_space_comparison()
    generate_vae_reparameterization_trick()
    generate_vq_vae_codebook_pipeline()
    generate_mae_vision_transformer_pipeline()
    print("All 7 Autoencoder publication-quality figures successfully generated in assets/!")
