"""
Analyze the actual E_end(θ₀) scaling law.

Find what power law or functional form E_end follows.
"""

import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 90)
print("ANALYZING E_end(θ₀) SCALING LAW")
print("=" * 90)

# Load fine-grid data
arr = np.loadtxt("rho_noPT_fine_grid.txt")
theta = arr[:, 0]
H = arr[:, 1]
rho = arr[:, 2]

# Get unique theta values and average rho for each
unique_theta = np.unique(theta)
rho_avg = np.array([np.mean(rho[np.abs(theta - t) < 1e-10]) for t in unique_theta])

# Filter out theta=0 (boundary)
mask = unique_theta > 1e-6
theta_clean = unique_theta[mask]
rho_clean = rho_avg[mask]

# Also exclude theta very close to pi (singularity)
mask_pi = theta_clean < 3.1
theta_clean = theta_clean[mask_pi]
rho_clean = rho_clean[mask_pi]

print(f"\nData: {len(theta_clean)} points")
print(f"θ₀ range: [{theta_clean.min():.6f}, {theta_clean.max():.6f}]")
print(f"ρ range: [{rho_clean.min():.6e}, {rho_clean.max():.6e}]")

# Parameters
H_PT = 0.7
M_phi = 1.0
t_osc = 3.0 / (2.0 * M_phi)
t_end = 10.0 / H_PT
a_osc = np.sqrt(2.0 * H_PT * t_osc)
a_end = np.sqrt(2.0 * H_PT * t_end)
redshift_cube = (a_osc / a_end) ** 3

# E_end = rho * redshift_cube (reverse the formula)
E_end = rho_clean / redshift_cube

print(f"\nRedshift factor: (a_osc/a_end)³ = {redshift_cube:.6f}")
print(f"E_end range: [{E_end.min():.6e}, {E_end.max():.6e}]")

# ============================================================================
# Test various scaling laws
# ============================================================================

models = {
    "Power θ²": (lambda t, A, n: A * t**2, [0.01, 2]),
    "Power θⁿ": (lambda t, A, n: A * t**n, [0.01, 2]),
    "Quadratic": (lambda t, a, b: a*t**2 + b*t**4, [0.01, 0.001]),
    "1-cos": (lambda t, A: A * (1 - np.cos(t)), [0.1]),
    "(1-cos)^p": (lambda t, A, p: A * (1 - np.cos(t))**p, [0.1, 1.0]),
    "(1-cos) + θ²": (lambda t, A, B: A*(1-np.cos(t)) + B*t**2, [0.1, 0.01]),
    "sinh": (lambda t, A, b: A * np.sinh(b*t), [0.0001, 1.5]),
    "Rational": (lambda t, A, B: A*t**2 / (1 + B*t**2), [0.01, 0.1]),
}

results = {}

print("\n" + "=" * 90)
print("FITTING DIFFERENT MODELS")
print("=" * 90)

for name, (func, p0) in models.items():
    try:
        popt, pcov = curve_fit(func, theta_clean, E_end, p0=p0, maxfev=10000)
        y_pred = func(theta_clean, *popt)
        ss_res = np.sum((E_end - y_pred)**2)
        ss_tot = np.sum((E_end - np.mean(E_end))**2)
        R2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((E_end - y_pred)**2))
        rel_rmse = rmse / np.mean(E_end) * 100

        results[name] = {
            'params': popt,
            'func': func,
            'R2': R2,
            'rmse': rmse,
            'rel_rmse': rel_rmse,
            'y_pred': y_pred,
        }

        print(f"\n{name}:")
        print(f"  R² = {R2:.6f}")
        print(f"  relRMSE = {rel_rmse:.2f}%")
        print(f"  Parameters: {popt}")

    except Exception as e:
        print(f"\n{name}: FAILED - {str(e)[:60]}")

# ============================================================================
# Find best model
# ============================================================================

print("\n" + "=" * 90)
print("RANKING BY R²")
print("=" * 90)

sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)

for rank, (name, data) in enumerate(sorted_results, 1):
    print(f"{rank}. {name:<20} R²={data['R2']:.6f}  relRMSE={data['rel_rmse']:.2f}%")

# ============================================================================
# Detailed analysis of best model
# ============================================================================

best_name, best_data = sorted_results[0]

print("\n" + "=" * 90)
print(f"BEST MODEL: {best_name}")
print("=" * 90)

params = best_data['params']
print(f"Parameters: {params}")

# Infer the scaling
if best_name == "Power θⁿ":
    A, n = params
    print(f"\nScaling: E_end = {A:.6e} * θ₀^{n:.4f}")
    print(f"This means E_end ∝ θ₀^{n:.4f}, NOT θ₀²!")

elif best_name == "(1-cos)^p":
    A, p = params
    print(f"\nScaling: E_end = {A:.6e} * (1-cos θ₀)^{p:.4f}")
    print(f"This suggests energy scales with (1-cos θ₀)^{p:.4f}")

elif best_name == "(1-cos) + θ²":
    A, B = params
    print(f"\nScaling: E_end = {A:.6e} * (1-cos θ₀) + {B:.6e} * θ₀²")
    print(f"Mixed scaling: linear in (1-cos) plus quadratic in θ₀²")

# ============================================================================
# Check consistency with V_theory = 1 - cos(θ₀)
# ============================================================================

print("\n" + "=" * 90)
print("COMPARISON WITH POTENTIAL V = 1 - cos(θ₀)")
print("=" * 90)

V_theory = 1 - np.cos(theta_clean)

print(f"\nE_end vs V_theory:")
print(f"{'θ₀':<10} {'E_end':<15} {'V_theory':<15} {'Ratio':<15}")
print("-" * 60)

for i in range(0, len(theta_clean), max(1, len(theta_clean)//10)):
    ratio = E_end[i] / V_theory[i]
    print(f"{theta_clean[i]:<10.4f} {E_end[i]:<15.6e} {V_theory[i]:<15.6f} {ratio:<15.6f}")

# Ratio analysis
ratio_array = E_end / V_theory
print(f"\nRatio E_end / V_theory statistics:")
print(f"  Mean: {np.mean(ratio_array):.6f}")
print(f"  Std:  {np.std(ratio_array):.6f}")
print(f"  Min:  {np.min(ratio_array):.6f}")
print(f"  Max:  {np.max(ratio_array):.6f}")

# ============================================================================
# Plot
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: E_end vs theta
ax = axes[0, 0]
ax.scatter(theta_clean, E_end, s=50, alpha=0.6, label='Data')
if best_name in results:
    ax.plot(theta_clean, best_data['y_pred'], 'r-', lw=2, label=f'Best: {best_name}')
ax.set_xlabel(r'$\theta_0$', fontsize=12)
ax.set_ylabel(r'$E_{\rm end}$', fontsize=12)
ax.set_title('Energy vs θ₀', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: log-log
ax = axes[0, 1]
ax.loglog(theta_clean, E_end, 'o-', ms=5, label='Data', alpha=0.7)
# Add power law reference
if best_name == "Power θⁿ":
    n = params[1]
    ax.loglog(theta_clean, E_end[0] * (theta_clean/theta_clean[0])**n, 'r--',
              lw=2, label=f'θ₀^{n:.2f}')
ax.set_xlabel(r'$\theta_0$ (log)', fontsize=12)
ax.set_ylabel(r'$E_{\rm end}$ (log)', fontsize=12)
ax.set_title('Log-log plot', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Plot 3: E_end / V_theory vs theta
ax = axes[1, 0]
ax.plot(theta_clean, ratio_array, 'o-', ms=5, alpha=0.7)
ax.axhline(np.mean(ratio_array), color='r', ls='--', lw=2, label=f'Mean = {np.mean(ratio_array):.4f}')
ax.fill_between(theta_clean, np.mean(ratio_array) - np.std(ratio_array),
                np.mean(ratio_array) + np.std(ratio_array), alpha=0.2, color='red')
ax.set_xlabel(r'$\theta_0$', fontsize=12)
ax.set_ylabel(r'$E_{\rm end} / V_{\rm theory}$', fontsize=12)
ax.set_title(r'Energy ratio', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Residuals
ax = axes[1, 1]
residuals = (E_end - best_data['y_pred']) / E_end * 100
ax.scatter(theta_clean, residuals, s=50, alpha=0.6)
ax.axhline(0, color='k', ls='-', lw=1)
ax.axhline(np.std(residuals), color='r', ls='--', lw=1, alpha=0.5)
ax.axhline(-np.std(residuals), color='r', ls='--', lw=1, alpha=0.5)
ax.set_xlabel(r'$\theta_0$', fontsize=12)
ax.set_ylabel('Relative residual [%]', fontsize=12)
ax.set_title(f'{best_name} residuals', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eend_scaling_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: eend_scaling_analysis.png")

print("\n" + "=" * 90)
