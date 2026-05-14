#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def default_kappa_grid() -> np.ndarray:
    # Dense low-kappa coverage, explicitly including kappa=0,
    # and no very large values by default.
    zero = np.array([0.0], dtype=np.float64)
    fine = np.arange(0.01, 0.201, 0.01)
    medium = np.arange(0.22, 0.401, 0.02)
    coarse = np.arange(0.45, 0.801, 0.05)
    grid = np.unique(np.round(np.concatenate([zero, fine, medium, coarse]), 6))
    return grid


def a_of_x(x: np.ndarray, b: float) -> np.ndarray:
    return np.sqrt(1.0 + 2.0 * x / b)


def compute_F(x: np.ndarray, b: float) -> np.ndarray:
    dx = x[1] - x[0]
    a = a_of_x(x, b)
    F = np.cumsum(1.0 / a) * dx
    i0 = int(np.argmin(np.abs(x)))
    return F - F[i0]


def compute_background_RD(
    b: float,
    H_star: float,
    v_w: float,
    P_crit: float,
    x_min_factor: float,
    x_max: float,
    dx: float,
) -> dict[str, np.ndarray | float]:
    x_min = x_min_factor * b
    x = np.arange(x_min, x_max + dx, dx, dtype=np.float64)
    a = a_of_x(x, b)

    mask = a > 0.0
    x = x[mask]
    a = a[mask]
    dx = float(x[1] - x[0])

    F = compute_F(x, b)

    c_geom = (4.0 * math.pi / 3.0) * (v_w**3) / (b**4)
    K = np.exp(x) * a**3

    I = np.zeros_like(x)
    for i in range(len(x)):
        dF = np.maximum(F[i] - F[: i + 1], 0.0)
        I[i] = c_geom * a[i] ** 3 * np.sum(K[: i + 1] * dF**3) * dx

    P = np.exp(-I)

    dP = np.zeros_like(P)
    dP[1:-1] = (P[2:] - P[:-2]) / (2.0 * dx)
    dP[0] = (P[1] - P[0]) / dx
    dP[-1] = (P[-1] - P[-2]) / dx

    idx = np.where(P <= P_crit)[0]
    if len(idx) == 0:
        raise RuntimeError("Transition incomplete on this x-grid; increase x_max.")

    k = max(int(idx[0]) - 1, 0)
    k = min(k, len(x) - 2)
    xk, xk1 = x[k], x[k + 1]
    Pk, Pk1 = P[k], P[k + 1]
    if Pk == Pk1:
        x_obs = xk1
    else:
        frac = (Pk - P_crit) / (Pk - Pk1)
        x_obs = xk + frac * (xk1 - xk)

    F_obs = float(np.interp(x_obs, x, F))

    return {
        "x": x,
        "a": a,
        "F": F,
        "P": P,
        "dP": dP,
        "dx": dx,
        "x_obs": float(x_obs),
        "F_obs": F_obs,
    }


def compute_Rmax_global_over_grid(
    H_star_values: list[float],
    b_values: list[float],
    v_w: float,
    P_crit: float,
    x_min_factor: float,
    x_max: float,
    dx_ref: float,
) -> float:
    Rmax_max = 0.0
    for b in b_values:
        bg = compute_background_RD(
            b=b,
            H_star=1.0,
            v_w=v_w,
            P_crit=P_crit,
            x_min_factor=x_min_factor,
            x_max=x_max,
            dx=dx_ref,
        )
        F = np.asarray(bg["F"], dtype=np.float64)
        F_obs = float(bg["F_obs"])
        deltaF = F_obs - float(F[0])
        for H_star in H_star_values:
            beta = b * H_star
            Rmax_bH = (v_w / beta) * deltaF
            Rmax_max = max(Rmax_max, float(Rmax_bH))
    return 1.1 * Rmax_max


def compute_p_Rc_one_loop_with_grid_vectorized(
    b: float,
    H_star: float,
    Rc_grid: np.ndarray,
    v_w: float,
    mphi: float,
    kappa: float,
    P_crit: float,
    x_min_factor: float,
    x_max: float,
    dx: float,
) -> tuple[np.ndarray, float, float, float, float, float]:
    beta = b * H_star

    bg = compute_background_RD(
        b=b,
        H_star=H_star,
        v_w=v_w,
        P_crit=P_crit,
        x_min_factor=x_min_factor,
        x_max=x_max,
        dx=dx,
    )
    x = np.asarray(bg["x"], dtype=np.float64)
    a = np.asarray(bg["a"], dtype=np.float64)
    F = np.asarray(bg["F"], dtype=np.float64)
    P = np.asarray(bg["P"], dtype=np.float64)
    dP = np.asarray(bg["dP"], dtype=np.float64)
    x_obs = float(bg["x_obs"])
    F_obs = float(bg["F_obs"])

    mask_act = x <= x_obs
    x_act = x[mask_act]
    a_act = a[mask_act]
    P_act = P[mask_act]
    dP_act = dP[mask_act]
    F_act = F[mask_act]

    if len(x_act) < 2:
        raise RuntimeError("Active region too small; refine grid.")

    F_min = float(F_act[0])
    dx_n = float(x_act[1] - x_act[0])

    Gam_n = (H_star**4) * np.exp(x_act)
    pref_n = (a_act**3) * Gam_n

    p_R = np.zeros_like(Rc_grid, dtype=np.float64)

    for iR, Rc in enumerate(Rc_grid):
        if Rc <= 0.0:
            continue

        F_target = F_act + (beta / v_w) * Rc
        mask_F = (F_target <= F_obs) & (F_target >= F_min)
        if not np.any(mask_F):
            continue

        F_target_valid = F_target[mask_F]
        x_n_valid = x_act[mask_F]
        pref_n_valid = pref_n[mask_F]

        x_c = np.interp(F_target_valid, F_act, x_act)
        mask_c = np.isfinite(x_c) & (x_c > x_n_valid)
        if not np.any(mask_c):
            continue

        x_c_sel = x_c[mask_c]
        pref_sel = pref_n_valid[mask_c]

        a_c = np.interp(x_c_sel, x_act, a_act)
        P_c = np.interp(x_c_sel, x_act, P_act)
        dP_c = np.interp(x_c_sel, x_act, dP_act)

        W_vec = pref_sel * a_c * P_c * (-dP_c)
        p_R[iR] = np.sum(W_vec) * dx_n

    dR = float(Rc_grid[1] - Rc_grid[0])
    p_R = np.maximum(p_R, 0.0)
    norm = float(np.sum(p_R * dR))
    if norm <= 0.0:
        raise RuntimeError("p(R_c) is zero everywhere; check parameters/grid.")
    p_R_norm = p_R / norm

    Rc_mean_all = float(np.sum(p_R_norm * Rc_grid * dR))

    # New definition requested by user:
    # R_min = kappa / mphi, no extra v_w factor.
    R_min = kappa / mphi
    mask_BM = Rc_grid >= R_min
    mask_BM_inv = mask_BM & (Rc_grid > 0.0)
    if not np.any(mask_BM):
        f_BM = 0.0
        G_BM_raw = 0.0
        Rc_mean_BM = 0.0
    else:
        f_BM = float(np.sum(p_R_norm[mask_BM] * dR))
        if np.any(mask_BM_inv):
            G_BM_raw = float(np.sum(p_R_norm[mask_BM_inv] * (1.0 / Rc_grid[mask_BM_inv]) * dR))
        else:
            G_BM_raw = 0.0
        if f_BM > 0.0:
            Rc_mean_BM = float(np.sum(p_R_norm[mask_BM] * Rc_grid[mask_BM] * dR) / f_BM)
        else:
            Rc_mean_BM = 0.0

    return p_R_norm, f_BM, G_BM_raw, float(R_min), Rc_mean_all, Rc_mean_BM


def compute_geometry_point(
    *,
    v_w: float,
    kappa: float,
    H_star: float,
    b: float,
    Rmax_global: float,
    mphi: float,
    P_crit: float,
    x_min_factor: float,
    x_max: float,
    dx: float,
    N_R: int,
) -> dict[str, float]:
    Rc_grid = np.linspace(0.0, float(Rmax_global), int(N_R), dtype=np.float64)
    x_max_try = float(x_max)
    last_err: RuntimeError | None = None
    for _ in range(8):
        try:
            p_Rc, f_BM, G_BM_raw, R_min, Rc_mean_all, Rc_mean_BM = compute_p_Rc_one_loop_with_grid_vectorized(
                b=float(b),
                H_star=float(H_star),
                Rc_grid=Rc_grid,
                v_w=float(v_w),
                mphi=float(mphi),
                kappa=float(kappa),
                P_crit=float(P_crit),
                x_min_factor=float(x_min_factor),
                x_max=x_max_try,
                dx=float(dx),
            )
            break
        except RuntimeError as err:
            last_err = err
            if "Transition incomplete on this x-grid" not in str(err):
                raise
            x_max_try *= 1.5
    else:
        raise RuntimeError(
            f"Transition incomplete for geometry point even after enlarging x_max to {x_max_try:g}"
        ) from last_err
    A_BM = float(G_BM_raw / f_BM) if f_BM > 0.0 else 0.0
    return {
        "b": float(b),
        "H_star": float(H_star),
        "f_BM": float(f_BM),
        "A_BM": float(A_BM),
        "G_BM": float(G_BM_raw),
        "BM_count": 0.0,
        "R_min": float(R_min),
        "Rc_mean_kappa0": float(Rc_mean_all),
        "Rc_mean_BM": float(Rc_mean_BM),
        "Rmax_global": float(Rmax_global),
        "x_max_used": float(x_max_try),
    }


def generate_one_file(
    v_w: float,
    kappa: float,
    H_star_values: list[float],
    b_values: list[float],
    Rmax_global: float,
    mphi: float,
    P_crit: float,
    x_min_factor: float,
    x_max: float,
    dx_final: float,
    N_R_final: int,
    output_dir: str,
    overwrite: bool,
) -> dict[str, str | float]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_name = f"BM_geometry_RD_kappa_{kappa:.3f}_vw{v_w:0.1f}_oneloop.json"
    out_path = outdir / out_name
    if out_path.exists() and not overwrite:
        return {"status": "skipped", "path": str(out_path), "v_w": v_w, "kappa": kappa}

    Rc_global = np.linspace(0.0, Rmax_global, N_R_final, dtype=np.float64)
    bm_table: dict[str, dict[str, dict[str, float | int | str]]] = {}

    for H_star in H_star_values:
        H_key = f"{H_star:.2f}"
        bm_table[H_key] = {}
        for b in b_values:
            p_Rc, f_BM, G_BM_raw, R_min, Rc_mean_all, Rc_mean_BM = compute_p_Rc_one_loop_with_grid_vectorized(
                b=b,
                H_star=H_star,
                Rc_grid=Rc_global,
                v_w=v_w,
                mphi=mphi,
                kappa=kappa,
                P_crit=P_crit,
                x_min_factor=x_min_factor,
                x_max=x_max,
                dx=dx_final,
            )
            A_BM = float(G_BM_raw / f_BM) if f_BM > 0.0 else 0.0
            b_key = f"{b:.2f}"
            bm_table[H_key][b_key] = {
                "b": float(b),
                "H_star": float(H_star),
                "f_BM": float(f_BM),
                "A_BM": float(A_BM),
                "G_BM": float(G_BM_raw),
                "BM_count": 0,
                "R_min": float(R_min),
                "Rc_mean_kappa0": float(Rc_mean_all),
                "Rc_mean_BM": float(Rc_mean_BM),
                "Rmax_global": float(Rmax_global),
                "kappa_definition": "R_min = kappa / mphi",
                "Rc_mean_kappa0_definition": "<Rc> over normalized p(Rc), i.e. kappa=0 / full-support mean",
                "Rc_mean_BM_definition": "<Rc> over the BM-selected support Rc >= R_min",
            }

    out_path.write_text(json.dumps(bm_table, indent=2))
    return {"status": "written", "path": str(out_path), "v_w": v_w, "kappa": kappa}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate compact BM geometry-bank JSON files using the one-loop vectorized "
            "construction, with the BM cutoff defined by R_min = kappa / mphi."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "geom_bank_compact_nomvw",
        help="Directory where the regenerated geometry bank will be written.",
    )
    parser.add_argument(
        "--vw-values",
        type=str,
        default="0.3,0.5,0.7,0.9,1.0",
        help="Comma-separated v_w values.",
    )
    parser.add_argument(
        "--H-values",
        type=str,
        default="0.05,0.10,0.20,0.50,1.00,1.50,2.00",
        help="Comma-separated H* values.",
    )
    parser.add_argument(
        "--b-values",
        type=str,
        default="1,2,3,4,5,6,8,10,12,16,20,25,32,40",
        help="Comma-separated beta/H* values.",
    )
    parser.add_argument(
        "--kappa-values",
        type=str,
        default="",
        help="Optional explicit comma-separated kappa list. Overrides the default compact grid.",
    )
    parser.add_argument("--kappa-min", type=float, default=0.01, help="Used only for informational logging.")
    parser.add_argument("--kappa-max", type=float, default=0.80, help="Used only for informational logging.")
    parser.add_argument("--kappa-step", type=float, default=0.01, help="Used only for informational logging.")
    parser.add_argument("--mphi", type=float, default=1.0)
    parser.add_argument("--p-crit", type=float, default=1.0e-2)
    parser.add_argument("--dx", type=float, default=0.005)
    parser.add_argument("--nr", type=int, default=1600, help="Number of Rc grid points.")
    parser.add_argument("--x-max", type=float, default=20.0)
    parser.add_argument("--x-min-factor", type=float, default=-0.49)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker count across (v_w, kappa) tasks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite files if they already exist.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    vw_values = parse_csv_floats(args.vw_values)
    H_star_values = parse_csv_floats(args.H_values)
    b_values = parse_csv_floats(args.b_values)
    if args.kappa_values.strip():
        kappa_values = np.array(parse_csv_floats(args.kappa_values), dtype=np.float64)
    else:
        kappa_values = default_kappa_grid()

    print("Generating compact geometry bank")
    print(f"  output_dir   = {args.output_dir}")
    print(f"  vw_values    = {vw_values}")
    print(f"  H_values     = {H_star_values}")
    print(f"  b_values     = {b_values}")
    print(f"  kappa_count  = {len(kappa_values)}")
    print(f"  kappa_range  = [{kappa_values.min():.3f}, {kappa_values.max():.3f}]")
    print(f"  workers      = {args.workers}")
    print(f"  cutoff       = R_min = kappa / mphi")

    rmax_by_vw: dict[float, float] = {}
    for v_w in vw_values:
        rmax = compute_Rmax_global_over_grid(
            H_star_values=H_star_values,
            b_values=b_values,
            v_w=v_w,
            P_crit=args.p_crit,
            x_min_factor=args.x_min_factor,
            x_max=args.x_max,
            dx_ref=min(0.01, args.dx),
        )
        rmax_by_vw[v_w] = rmax
        print(f"  Rmax_global(vw={v_w:.1f}) = {rmax:.6f}")

    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for v_w in vw_values:
            for kappa in kappa_values:
                jobs.append(
                    ex.submit(
                        generate_one_file,
                        float(v_w),
                        float(kappa),
                        H_star_values,
                        b_values,
                        float(rmax_by_vw[v_w]),
                        float(args.mphi),
                        float(args.p_crit),
                        float(args.x_min_factor),
                        float(args.x_max),
                        float(args.dx),
                        int(args.nr),
                        str(args.output_dir),
                        bool(args.overwrite),
                    )
                )

        for fut in as_completed(jobs):
            res = fut.result()
            print(f"[{res['status']}] vw={res['v_w']:.1f}, kappa={res['kappa']:.3f} -> {res['path']}")

    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "vw_values": vw_values,
        "H_star_values": H_star_values,
        "b_values": b_values,
        "kappa_values": [float(x) for x in kappa_values],
        "mphi": float(args.mphi),
        "P_crit": float(args.p_crit),
        "dx": float(args.dx),
        "N_R": int(args.nr),
        "x_max": float(args.x_max),
        "x_min_factor": float(args.x_min_factor),
        "workers": int(args.workers),
        "kappa_definition": "R_min = kappa / mphi",
        "rmax_by_vw": {f"{k:.1f}": float(v) for k, v in rmax_by_vw.items()},
    }
    summary_path = Path(args.output_dir) / "geom_compact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
