from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from xi_model import load_default_model


matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "axes.titlesize": 22,
    }
)


OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    model = load_default_model()

    vw = 0.5
    thetas = [0.2617993877991494, 2.356194490192345]
    beta_grid = np.array([4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40], dtype=float)
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(thetas)))

    for hstar, stem_name in [
        (2.7, "xi_vs_betaH_hstar_2p7_vw_0p5_two_theta"),
        (1.0e-4, "xi_vs_betaH_lowHstar_1e-4_vw_0p5_two_theta"),
        (3.0e-4, "xi_vs_betaH_lowHstar_3e-4_vw_0p5_two_theta"),
        (3.0e-6, "xi_vs_betaH_lowHstar_3e-6_vw_0p5_two_theta"),
    ]:
        fig, ax = plt.subplots(figsize=(7.1, 5.2), constrained_layout=True)

        for theta0, color in zip(thetas, colors):
            xi = []
            xi_lo = []
            xi_hi = []
            for beta_over_h in beta_grid:
                res = model.predict(hstar=hstar, vw=vw, theta0=theta0, beta_over_h=float(beta_over_h), clip=False)
                xi.append(res.xi)
                xi_lo.append(res.xi_lo)
                xi_hi.append(res.xi_hi)
            xi = np.array(xi)
            xi_lo = np.array(xi_lo)
            xi_hi = np.array(xi_hi)
            ax.fill_between(beta_grid, xi_lo, xi_hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(
                beta_grid,
                xi,
                color=color,
                linewidth=2.4,
                label=rf"$\theta_0={theta0:.2f}$",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\beta/H_*$")
        ax.set_ylabel(r"$\xi$")
        ax.set_xticks([4, 10, 20, 40], labels=[r"$4$", r"$10$", r"$20$", r"$40$"])
        ax.legend(loc="lower right", frameon=True)

        stem = OUT_DIR / stem_name
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
