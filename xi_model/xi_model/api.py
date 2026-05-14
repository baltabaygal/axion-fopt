from __future__ import annotations

import argparse
import csv
import json
import math
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import _geom_compact, _percolation

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
GEOM_DIR = DATA_DIR / "geometry_bank"
LATTICE_DIR = DATA_DIR / "lattice_tables"
GSTAR_LOOKUP_PATH = DATA_DIR / "gstar_lookup.json"
KAPPA_PLATEAU_REL_TOL = 3.0e-2
DOMAIN_EPS = 1.0e-9
MPL_REDUCED_EV = 2.435e27


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


_GSTAR_LOOKUP = _load_json(GSTAR_LOOKUP_PATH)
_T_DOF_EV = np.array(_GSTAR_LOOKUP["temperature_ev"], dtype=float)
_GSTAR_ENERGY = np.array(_GSTAR_LOOKUP["g_energy"], dtype=float)
_GSTAR_ENTROPY = np.array(_GSTAR_LOOKUP["g_entropy"], dtype=float)


def _x_theta(theta0: float) -> float:
    return float(np.log(np.e / (np.cos(float(theta0) / 2.0) ** 2)))


def _potential(theta0: float) -> float:
    return float(1.0 - np.cos(float(theta0)))


def _clip_value(value: float, grid: np.ndarray) -> tuple[float, str | None]:
    lo = float(grid.min())
    hi = float(grid.max())
    if value < lo:
        return lo, f"clipped from {value:g} to lower domain edge {lo:g}"
    if value > hi:
        return hi, f"clipped from {value:g} to upper domain edge {hi:g}"
    return float(value), None


def _interp_linear_extrap(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x = float(x)
    if len(xs) < 2:
        return float(ys[0])
    if x <= float(xs[0]):
        x0, x1 = float(xs[0]), float(xs[1])
        y0, y1 = float(ys[0]), float(ys[1])
    elif x >= float(xs[-1]):
        x0, x1 = float(xs[-2]), float(xs[-1])
        y0, y1 = float(ys[-2]), float(ys[-1])
    else:
        return float(np.interp(x, xs, ys))
    slope = (y1 - y0) / (x1 - x0)
    return float(y0 + slope * (x - x0))


def _interp_linear_extrap_array(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.interp(x, xs, ys)
    if len(xs) < 2:
        return np.full_like(x, float(ys[0]), dtype=float)
    left = x <= float(xs[0])
    if np.any(left):
        x0, x1 = float(xs[0]), float(xs[1])
        y0, y1 = float(ys[0]), float(ys[1])
        slope = (y1 - y0) / (x1 - x0)
        out[left] = y0 + slope * (x[left] - x0)
    right = x >= float(xs[-1])
    if np.any(right):
        x0, x1 = float(xs[-2]), float(xs[-1])
        y0, y1 = float(ys[-2]), float(ys[-1])
        slope = (y1 - y0) / (x1 - x0)
        out[right] = y0 + slope * (x[right] - x0)
    return out.astype(float, copy=False)


def _interp_log_temperature_dof(temp_ev: float, values: np.ndarray) -> float:
    temp_ev = float(max(temp_ev, float(_T_DOF_EV[0])))
    return float(np.interp(np.log(temp_ev), np.log(_T_DOF_EV), np.asarray(values, dtype=float)))


def _gstar_energy(temp_ev: float) -> float:
    return _interp_log_temperature_dof(temp_ev, _GSTAR_ENERGY)


def _gstar_entropy(temp_ev: float) -> float:
    return _interp_log_temperature_dof(temp_ev, _GSTAR_ENTROPY)


def _temperature_from_hubble_rd(hubble_ev: float) -> float:
    temp_ev = float(math.sqrt(hubble_ev * MPL_REDUCED_EV) * (90.0 / (math.pi**2 * 100.0)) ** 0.25)
    for _ in range(32):
        g_eff = _gstar_energy(temp_ev)
        updated = float(math.sqrt(hubble_ev * MPL_REDUCED_EV) * (90.0 / (math.pi**2 * g_eff)) ** 0.25)
        if abs(updated - temp_ev) <= 1.0e-10 * max(1.0, temp_ev):
            temp_ev = updated
            break
        temp_ev = updated
    return temp_ev


@dataclass
class PredictResult:
    xi: float
    xi_err: float
    xi_lo: float
    xi_hi: float
    xi_dm: float
    tp: float
    fanh_theta: float
    rho_weight: float
    kappa: float
    kappa_raw: float
    kappa_pilot: float
    kappa_plateau_applied: bool
    kappa_plateau_hstar: float
    geometry_vw: float
    geometry_hstar: float
    geometry_generated_kappas: list[float]
    geometry_generated_betas: list[float]
    f_bm: float
    g_bm: float
    lambda_theta: float
    a0_theta: float
    s_vw: float
    p: float
    q: float
    xi_dm_mode: str
    include_gstar: bool
    mphi_ev: float | None
    gstar_thermo_factor: float
    t_pt_ev: float | None
    t_osc_ev: float | None
    gstar_pt: float | None
    gstar_osc: float | None
    gsstar_pt: float | None
    gsstar_osc: float | None
    prediction_status: str
    status_flags: list[str]
    clipped_inputs: dict[str, float]
    in_domain: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class XiModel:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.geom_dir = self.data_dir / "geometry_bank"
        self._geometry_generation_lock = threading.Lock()
        self.summary = _load_json(self.data_dir / "summary.json")
        self.local = _load_json(self.data_dir / "local_summary.json")
        self._load_fit_grid()
        self._load_percolation_kernel()
        self._load_pilot_kappa()
        self._load_geometry()
        self._load_geom_generator()
        self.q = float(self.summary["q"])
        self._load_fanh_baseline()

    def _load_fit_grid(self) -> None:
        path = self.data_dir / "fit_points.csv"
        rows = []
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        theta = sorted({float(r["theta"]) for r in rows})
        vw = sorted({float(r["v_w"]) for r in rows})
        hstar = sorted({float(r["H"]) for r in rows})
        beta = sorted({float(r["beta_over_H"]) for r in rows})
        self.theta_grid = np.array(theta, dtype=float)
        self.vw_grid = np.array(vw, dtype=float)
        self.hstar_grid = np.array(hstar, dtype=float)
        self.beta_grid = np.array(beta, dtype=float)
        shape = (len(theta), len(vw), len(hstar), len(beta))
        xi_dm = np.zeros(shape, dtype=float)
        xi_dm_field = "xi_dm_broken_powerlaw" if "xi_dm_broken_powerlaw" in rows[0] else "xi_dm_H2H02"
        self.xi_dm_grid_source = xi_dm_field
        index_t = {v: i for i, v in enumerate(theta)}
        index_v = {v: i for i, v in enumerate(vw)}
        index_h = {v: i for i, v in enumerate(hstar)}
        index_b = {v: i for i, v in enumerate(beta)}
        for r in rows:
            it = index_t[float(r["theta"])]
            iv = index_v[float(r["v_w"])]
            ih = index_h[float(r["H"])]
            ib = index_b[float(r["beta_over_H"])]
            xi_dm[it, iv, ih, ib] = float(r[xi_dm_field])
        points = (self.theta_grid, self.vw_grid, self.hstar_grid, self.beta_grid)
        self.xi_dm_interp = RegularGridInterpolator(points, xi_dm, bounds_error=False, fill_value=None)

    def _load_percolation_kernel(self) -> None:
        self._tperc_cache = _percolation.PercolationCache()

    def _load_pilot_kappa(self) -> None:
        path = self.data_dir / "lattice_tables" / "pilot_kappa.csv"
        rows = np.genfromtxt(path, delimiter=",", names=True)
        theta = np.array(sorted({float(v) for v in rows["theta0"]}), dtype=float)
        vw = np.array(sorted({float(v) for v in rows["vw"]}), dtype=float)
        hstar = np.array(sorted({float(v) for v in rows["Hb"]}), dtype=float)
        arr = np.zeros((len(theta), len(vw), len(hstar)), dtype=float)
        plateau_h = np.zeros((len(theta), len(vw)), dtype=float)
        plateau_kappa = np.zeros((len(theta), len(vw)), dtype=float)
        it = {v: i for i, v in enumerate(theta)}
        iv = {v: i for i, v in enumerate(vw)}
        ih = {v: i for i, v in enumerate(hstar)}
        for row in rows:
            arr[it[float(row["theta0"])], iv[float(row["vw"])], ih[float(row["Hb"])]] = float(row["kappa"])
        for i in range(len(theta)):
            for j in range(len(vw)):
                curve = arr[i, j, :]
                baseline = float(curve[0])
                rel = np.abs(curve / max(abs(baseline), 1.0e-12) - 1.0)
                plateau_end = 0
                for idx, is_flat in enumerate(rel <= KAPPA_PLATEAU_REL_TOL):
                    if is_flat:
                        plateau_end = idx
                    else:
                        break
                plateau_h[i, j] = float(hstar[plateau_end])
                plateau_kappa[i, j] = float(np.mean(curve[: plateau_end + 1]))
        self.kappa_theta_grid = theta
        self.kappa_vw_grid = vw
        self.kappa_hstar_grid = hstar
        self.kappa_plateau_h_interp = RegularGridInterpolator((theta, vw), plateau_h, bounds_error=False, fill_value=None)
        self.kappa_plateau_value_interp = RegularGridInterpolator((theta, vw), plateau_kappa, bounds_error=False, fill_value=None)
        self.kappa_interp = RegularGridInterpolator((theta, vw, hstar), arr, bounds_error=False, fill_value=None)

    def _load_geometry(self) -> None:
        files = sorted(self.geom_dir.glob("BM_geometry_RD_kappa_*_vw*_oneloop.json"))
        if not files:
            raise FileNotFoundError(f"No geometry JSON files found in {self.geom_dir}")
        payloads = [(path, _load_json(path)) for path in files]
        vws = sorted({float(p.stem.split("_")[5].replace("vw", "")) for p in files})
        self.geom_vw_grid = np.array(vws, dtype=float)
        self.geom_models: dict[float, dict[str, Any]] = {}
        global_h = None
        global_beta = None
        for vw in vws:
            vw_payloads = []
            for path, payload in payloads:
                parts = path.stem.split("_")
                if abs(float(parts[5].replace("vw", "")) - vw) < 1.0e-9:
                    vw_payloads.append((float(parts[4]), payload))
            kappas_vw = np.array(sorted(k for k, _ in vw_payloads), dtype=float)
            hstar_vw = np.array(sorted({float(k) for _, payload in vw_payloads for k in payload.keys()}), dtype=float)
            beta_sets = []
            for _, payload in vw_payloads:
                local_betas = {
                    float(b)
                    for _, hpayload in payload.items()
                    for b in hpayload.keys()
                }
                beta_sets.append(local_betas)
            beta_common = set.intersection(*beta_sets) if beta_sets else set()
            beta_vw = np.array(sorted(beta_common), dtype=float)
            f_arr = np.full((len(hstar_vw), len(kappas_vw), len(beta_vw)), np.nan, dtype=float)
            g_arr = np.full_like(f_arr, np.nan)
            ik = {k: i for i, k in enumerate(kappas_vw)}
            ih = {h: i for i, h in enumerate(hstar_vw)}
            ib = {b: i for i, b in enumerate(beta_vw)}
            for kappa, payload in vw_payloads:
                for hkey, hpayload in payload.items():
                    h = float(hkey)
                    for bkey, bpayload in hpayload.items():
                        b = float(bkey)
                        if b not in ib:
                            continue
                        f_arr[ih[h], ik[kappa], ib[b]] = float(bpayload["f_BM"])
                        g_arr[ih[h], ik[kappa], ib[b]] = float(bpayload["G_BM"])
            if np.isnan(f_arr).any() or np.isnan(g_arr).any():
                raise RuntimeError(f"Geometry bank for vw={vw:g} is not rectangular in (H_star, kappa, beta/H*); regenerate missing slices before loading.")
            self.geom_models[vw] = {
                "h_grid": hstar_vw,
                "kappa_grid": kappas_vw,
                "beta_grid": beta_vw,
                "f_interp": RegularGridInterpolator((hstar_vw, kappas_vw, beta_vw), f_arr, bounds_error=False, fill_value=None),
                "g_interp": RegularGridInterpolator((hstar_vw, kappas_vw, beta_vw), g_arr, bounds_error=False, fill_value=None),
            }
            if global_h is None:
                global_h = hstar_vw
            if global_beta is None:
                global_beta = beta_vw
        self.geom_h_grid = np.array(global_h, dtype=float)
        self.geom_beta_grid = np.array(global_beta, dtype=float)

    def _persist_geom_summary(self) -> None:
        summary_path = self.geom_dir / "geom_compact_summary.json"
        if self.geom_summary is not None:
            summary_path.write_text(json.dumps(self.geom_summary, indent=2, sort_keys=True))

    def _load_geom_generator(self) -> None:
        self._geom_generator_module = _geom_compact
        summary_path = self.geom_dir / "geom_compact_summary.json"
        self.geom_summary = _load_json(summary_path) if summary_path.exists() else None

    def _ensure_geometry_support(self, vw: float, kappa_needed: float) -> list[float]:
        geom_vw = float(np.clip(vw, float(self.geom_vw_grid.min()), float(self.geom_vw_grid.max())))
        geom_model = self.geom_models[geom_vw]
        kappa_grid = np.asarray(geom_model["kappa_grid"], dtype=float)
        if kappa_needed <= float(kappa_grid.max()) + 1.0e-12:
            return []
        if self.geom_summary is None:
            return []
        if not any(abs(geom_vw - float(v)) < 1.0e-9 for v in self.geom_summary.get("vw_values", [])):
            return []

        # For one prediction we only need the requested kappa slice, not the
        # whole staircase of intermediate files up to that kappa.
        target = round(float(kappa_needed), 3)
        targets = [target]
        if not targets:
            return []

        generated: list[float] = []
        with self._geometry_generation_lock:
            # Another thread/process may have generated files since the first check.
            current_max = float(np.asarray(self.geom_models[geom_vw]["kappa_grid"], dtype=float).max())
            if kappa_needed <= current_max + 1.0e-12:
                return []
            for kappa in targets:
                out_name = f"BM_geometry_RD_kappa_{kappa:.3f}_vw{geom_vw:0.1f}_oneloop.json"
                out_path = self.geom_dir / out_name
                if out_path.exists():
                    continue
                self._geom_generator_module.generate_one_file(
                    float(geom_vw),
                    float(kappa),
                    [float(v) for v in self.geom_summary["H_star_values"]],
                    [float(v) for v in self.geom_summary["b_values"]],
                    float(self.geom_summary["rmax_by_vw"][f"{geom_vw:.1f}"]),
                    float(self.geom_summary["mphi"]),
                    float(self.geom_summary["P_crit"]),
                    float(self.geom_summary["x_min_factor"]),
                    float(self.geom_summary["x_max"]),
                    float(self.geom_summary["dx"]),
                    int(self.geom_summary["N_R"]),
                    str(self.geom_dir),
                    False,
                )
                generated.append(float(kappa))
            if generated:
                self._load_geometry()
        return generated

    def _ensure_geometry_beta_support(self, beta_needed: float) -> list[float]:
        if beta_needed <= float(self.geom_beta_grid.max()) + 1.0e-12:
            return []
        if self.geom_summary is None:
            return []
        target_beta = round(float(beta_needed), 6)
        existing_betas = sorted({float(v) for v in self.geom_summary.get("b_values", [])} | {float(v) for v in self.geom_beta_grid})
        if any(abs(target_beta - b) < 1.0e-9 for b in existing_betas):
            return []
        new_betas = sorted(existing_betas + [target_beta])
        vws = [float(v) for v in self.geom_summary.get("vw_values", list(self.geom_vw_grid))]
        generated: list[float] = []
        with self._geometry_generation_lock:
            current_max = float(self.geom_beta_grid.max())
            if beta_needed <= current_max + 1.0e-12:
                return []
            for geom_vw in vws:
                rmax = float(self.geom_summary["rmax_by_vw"][f"{geom_vw:.1f}"])
                kappas_for_vw = [float(v) for v in np.asarray(self.geom_models[geom_vw]["kappa_grid"], dtype=float)]
                for kappa in kappas_for_vw:
                    self._geom_generator_module.generate_one_file(
                        float(geom_vw),
                        float(kappa),
                        [float(v) for v in self.geom_summary["H_star_values"]],
                        new_betas,
                        rmax,
                        float(self.geom_summary["mphi"]),
                        float(self.geom_summary["P_crit"]),
                        float(self.geom_summary["x_min_factor"]),
                        float(self.geom_summary["x_max"]),
                        float(self.geom_summary["dx"]),
                        int(self.geom_summary["N_R"]),
                        str(self.geom_dir),
                        True,
                    )
            self.geom_summary["b_values"] = new_betas
            self._persist_geom_summary()
            generated.append(target_beta)
            self._load_geometry()
        return generated

    def _geometry_vw_bracket(self, vw: float) -> tuple[float, float]:
        grid = np.asarray(self.geom_vw_grid, dtype=float)
        if vw <= float(grid[0]):
            return float(grid[0]), float(grid[0])
        if vw >= float(grid[-1]):
            return float(grid[-1]), float(grid[-1])
        idx = int(np.searchsorted(grid, vw))
        if abs(float(grid[idx]) - vw) < 1.0e-12:
            return float(grid[idx]), float(grid[idx])
        return float(grid[idx - 1]), float(grid[idx])

    def _eval_geometry_direct(self, vw_node: float, hstar: float, kappa_raw: float, beta_over_h: float) -> dict[str, float]:
        if self.geom_summary is None:
            raise RuntimeError("Geometry summary unavailable for direct geometry evaluation")
        kappa_eval = round(float(kappa_raw), 3)
        h_bank = np.asarray(self.geom_h_grid, dtype=float)
        h_eval = float(np.clip(hstar, float(h_bank.min()), float(h_bank.max())))
        h_lo = max([h for h in h_bank if h <= h_eval], default=float(h_bank.min()))
        h_hi = min([h for h in h_bank if h >= h_eval], default=float(h_bank.max()))

        def _one_point(h_node: float) -> dict[str, float]:
            return self._geom_generator_module.compute_geometry_point(
                v_w=float(vw_node),
                kappa=float(kappa_eval),
                H_star=float(h_node),
                b=float(beta_over_h),
                Rmax_global=float(self.geom_summary["rmax_by_vw"][f"{vw_node:.1f}"]),
                mphi=float(self.geom_summary["mphi"]),
                P_crit=float(self.geom_summary["P_crit"]),
                x_min_factor=float(self.geom_summary["x_min_factor"]),
                x_max=float(self.geom_summary["x_max"]),
                dx=float(self.geom_summary["dx"]),
                N_R=int(self.geom_summary["N_R"]),
            )

        if abs(h_hi - h_lo) < 1.0e-12:
            cell = _one_point(h_lo)
            f_bm = float(cell["f_BM"])
            g_bm = float(cell["G_BM"])
        else:
            c_lo = _one_point(h_lo)
            c_hi = _one_point(h_hi)
            w = (h_eval - h_lo) / (h_hi - h_lo)
            f_bm = (1.0 - w) * float(c_lo["f_BM"]) + w * float(c_hi["f_BM"])
            g_bm = (1.0 - w) * float(c_lo["G_BM"]) + w * float(c_hi["G_BM"])
        return {
            "f_bm": float(f_bm),
            "g_bm": float(g_bm),
            "kappa": float(kappa_eval),
            "geom_hstar": h_eval,
            "geom_beta": float(beta_over_h),
            "kappa_min": float(kappa_eval),
            "kappa_max": float(kappa_eval),
            "generated_kappa": float(kappa_eval) if not any(abs(kappa_eval - k) < 1.0e-9 for k in self.geom_models[vw_node]["kappa_grid"]) else None,
            "generated_beta": float(beta_over_h) if beta_over_h > float(self.geom_models[vw_node]["beta_grid"].max()) + 1.0e-12 else None,
        }

    def _load_fanh_baseline(self) -> None:
        self.fanh_fit_summary = _load_json(self.data_dir / "effective_ftilde_by_vw_summary.json")
        self.fanh_fit_a_inf = float(self.fanh_fit_summary["f_inf"]["A_inf"])
        self.fanh_fit_gamma_inf = float(self.fanh_fit_summary["f_inf"]["gamma_inf"])
        self.fanh_fit_c0 = float(self.fanh_fit_summary["C_law"]["C0"])
        self.fanh_fit_c_p = float(self.fanh_fit_summary["C_law"]["p"])
        self.fanh_fit_r0 = float(self.fanh_fit_summary["r_law"]["r0"])
        self.fanh_fit_r1 = float(self.fanh_fit_summary["r_law"]["r1"])

    @property
    def domain(self) -> dict[str, tuple[float, float]]:
        return {
            "theta0": (float(self.theta_grid.min()), float(self.theta_grid.max())),
            "vw": (float(self.vw_grid.min()), float(self.vw_grid.max())),
            "hstar": (float(self.hstar_grid.min()), float(self.hstar_grid.max())),
            "beta_over_h": (float(self.beta_grid.min()), float(self.beta_grid.max())),
        }

    def _effective_a0(self, theta0: float) -> float:
        xs = np.array(sorted(float(k) for k in self.summary["A0_theta"].keys()), dtype=float)
        ys = np.array([float(self.summary["A0_theta"][f"{x:0.6f}"]) for x in xs], dtype=float)
        return _interp_linear_extrap(theta0, xs, ys)

    def _effective_s(self, vw: float) -> float:
        xs = np.array(sorted(float(k) for k in self.summary["s_by_vw"].keys()), dtype=float)
        ys = np.array([float(self.summary["s_by_vw"][f"{x:0.1f}"]) for x in xs], dtype=float)
        return _interp_linear_extrap(vw, xs, ys)

    def _effective_sigma_a0(self, theta0: float) -> float:
        params = self.local["params"]
        xs = np.array(sorted(float(k.split("_", 1)[1]) for k in params if k.startswith("A0_")), dtype=float)
        ys = np.array([float(params[f"A0_{x:0.6f}"]["sigma"]) for x in xs], dtype=float)
        return max(0.0, _interp_linear_extrap(theta0, xs, ys))

    def _effective_sigma_s(self, vw: float) -> float:
        params = self.local["params"]
        xs = np.array(sorted(float(k.split("_", 1)[1]) for k in params if k.startswith("s_")), dtype=float)
        ys = np.array([float(params[f"s_{x:0.1f}"]["sigma"]) for x in xs], dtype=float)
        return max(0.0, _interp_linear_extrap(vw, xs, ys))

    def _lambda_theta(self, theta0: float, lambda_A: float | None = None, lambda_gamma: float | None = None) -> float:
        lamA = float(self.summary["lambda_law"]["A"] if lambda_A is None else lambda_A)
        gam = float(self.summary["lambda_law"]["gamma"] if lambda_gamma is None else lambda_gamma)
        return float(lamA * (_x_theta(theta0) ** gam))

    def _fanh_theta(self, theta0: float) -> float:
        A_f = 0.37293632597
        gamma_f = 1.2033986788
        return float(A_f * (_x_theta(theta0) ** gamma_f))

    def _fanh_inf_theta(self, theta0: float) -> float:
        return float(self.fanh_fit_a_inf * (_x_theta(theta0) ** self.fanh_fit_gamma_inf))

    def _fanh_c_vw(self, vw: float) -> float:
        return float(self.fanh_fit_c0 * (float(vw) ** self.fanh_fit_c_p))

    def _fanh_r_vw(self, vw: float) -> float:
        return float(self.fanh_fit_r0 + self.fanh_fit_r1 * float(vw))

    def _fanh_pt_broken_powerlaw(self, theta0: float, tp: float, vw: float) -> float:
        f0 = self._fanh_theta(theta0)
        finf = self._fanh_inf_theta(theta0)
        r = self._fanh_r_vw(vw)
        c = self._fanh_c_vw(vw)
        inside = (c * finf) ** r + (f0 ** r) / (((2.0 / 3.0) * tp) ** (1.5 * r))
        return float(max(inside, 1.0e-300) ** (1.0 / r))

    def _xi_dm_from_ftilde(self, theta0: float, tp: float, vw: float) -> float:
        f0 = self._fanh_theta(theta0)
        ftilde = self._fanh_pt_broken_powerlaw(theta0, tp, vw)
        return float((ftilde / f0) * (((2.0 * tp) / 3.0) ** 1.5))

    def predict_theta_batch(
        self,
        *,
        theta0_array: np.ndarray,
        hstar: float,
        vw: float,
        beta_over_h: float,
        clip: bool = False,
        xi_dm_mode: str = "broken_powerlaw_ftilde",
    ) -> dict[str, np.ndarray | float | bool]:
        theta = np.asarray(theta0_array, dtype=float)
        if theta.ndim != 1:
            raise ValueError("theta0_array must be a 1D array")
        if theta.size == 0:
            return {
                "theta0": theta.copy(),
                "xi": np.array([], dtype=float),
                "fanh_theta": np.array([], dtype=float),
                "rho_weight": np.array([], dtype=float),
                "tp": float("nan"),
                "batched": True,
            }
        if xi_dm_mode != "broken_powerlaw_ftilde":
            raise ValueError("predict_theta_batch currently supports only xi_dm_mode='broken_powerlaw_ftilde'")

        # This fast path is intended for fixed (H_*, v_w, beta/H_*) scans over theta.
        # If any point falls outside the in-bank geometry regime, fall back to scalar
        # predict() so behavior remains correct.
        vals0, _, _ = self._prepare_inputs(
            hstar=hstar,
            vw=vw,
            theta0=float(theta[0]),
            beta_over_h=beta_over_h,
            clip=clip,
        )
        tp = self._tp_rd(vals0["hstar"], vals0["beta_over_h"], vals0["vw"])
        geom_vw = float(np.clip(vals0["vw"], float(self.geom_vw_grid.min()), float(self.geom_vw_grid.max())))
        vw_lo, vw_hi = self._geometry_vw_bracket(geom_vw)
        s_eff = float(self._effective_s(vals0["vw"]))

        theta_eval = theta.copy()
        if clip:
            theta_eval = np.clip(theta_eval, 0.0, math.pi)

        plateau_point = np.column_stack([theta_eval, np.full(theta_eval.shape, vals0["vw"], dtype=float)])
        plateau_hstar = np.asarray(self.kappa_plateau_h_interp(plateau_point), dtype=float)
        plateau_kappa = np.asarray(self.kappa_plateau_value_interp(plateau_point), dtype=float)
        kappa_interp_points = np.column_stack(
            [theta_eval, np.full(theta_eval.shape, vals0["vw"], dtype=float), np.full(theta_eval.shape, vals0["hstar"], dtype=float)]
        )
        kappa_pilot_interp = np.asarray(self.kappa_interp(kappa_interp_points), dtype=float)
        kappa_pilot = np.where(vals0["hstar"] <= plateau_hstar, plateau_kappa, kappa_pilot_interp)
        kappa_raw = s_eff * kappa_pilot

        evals: list[dict[str, np.ndarray | float]] = []
        for vw_node in sorted({vw_lo, vw_hi}):
            geom_model = self.geom_models[vw_node]
            kappa_grid = np.asarray(geom_model["kappa_grid"], dtype=float)
            beta_grid = np.asarray(geom_model["beta_grid"], dtype=float)
            use_direct = (
                vals0["beta_over_h"] > float(beta_grid.max()) + 1.0e-12
                or np.any(kappa_raw > float(kappa_grid.max()) + 1.0e-12)
            )
            if use_direct:
                xi_list = []
                fanh_list = []
                rho_list = []
                for th in theta_eval:
                    res = self.predict(
                        hstar=vals0["hstar"],
                        vw=vals0["vw"],
                        theta0=float(th),
                        beta_over_h=vals0["beta_over_h"],
                        clip=clip,
                        xi_dm_mode=xi_dm_mode,
                    )
                    xi_list.append(res.xi)
                    fanh_list.append(res.fanh_theta)
                    rho_list.append(res.rho_weight)
                return {
                    "theta0": theta_eval,
                    "xi": np.array(xi_list, dtype=float),
                    "fanh_theta": np.array(fanh_list, dtype=float),
                    "rho_weight": np.array(rho_list, dtype=float),
                    "tp": float(tp),
                    "batched": False,
                }
            h_grid = np.asarray(geom_model["h_grid"], dtype=float)
            kappa_node = np.clip(kappa_raw, float(kappa_grid.min()), float(kappa_grid.max()))
            geom_hstar = float(np.clip(vals0["hstar"], float(h_grid.min()), float(h_grid.max())))
            geom_beta = float(np.clip(vals0["beta_over_h"], float(beta_grid.min()), float(beta_grid.max())))
            gpoint = np.column_stack(
                [
                    np.full(theta_eval.shape, geom_hstar, dtype=float),
                    kappa_node.astype(float),
                    np.full(theta_eval.shape, geom_beta, dtype=float),
                ]
            )
            evals.append(
                {
                    "f_bm": np.asarray(geom_model["f_interp"](gpoint), dtype=float),
                    "g_bm": np.asarray(geom_model["g_interp"](gpoint), dtype=float),
                }
            )

        if vw_lo == vw_hi:
            f_bm = np.asarray(evals[0]["f_bm"], dtype=float)
            g_bm = np.asarray(evals[0]["g_bm"], dtype=float)
        else:
            w = (geom_vw - vw_lo) / (vw_hi - vw_lo)
            f_bm = (1.0 - w) * np.asarray(evals[0]["f_bm"], dtype=float) + w * np.asarray(evals[1]["f_bm"], dtype=float)
            g_bm = (1.0 - w) * np.asarray(evals[0]["g_bm"], dtype=float) + w * np.asarray(evals[1]["g_bm"], dtype=float)

        f_bm = np.clip(f_bm, 1.0e-12, 1.0 - 1.0e-12)
        g_bm = np.maximum(g_bm, 0.0)

        x = np.log(np.e / (np.cos(theta_eval / 2.0) ** 2))
        A_f = 0.37293632597
        gamma_f = 1.2033986788
        f0 = A_f * (x**gamma_f)
        finf = self.fanh_fit_a_inf * (x**self.fanh_fit_gamma_inf)
        r = self._fanh_r_vw(vals0["vw"])
        c = self._fanh_c_vw(vals0["vw"])
        inside = (c * finf) ** r + (f0**r) / (((2.0 / 3.0) * tp) ** (1.5 * r))
        ftilde = np.maximum(inside, 1.0e-300) ** (1.0 / r)
        xi_dm = (ftilde / f0) * (((2.0 * tp) / 3.0) ** 1.5)

        lamA = float(self.summary["lambda_law"]["A"])
        gam = float(self.summary["lambda_law"]["gamma"])
        lam = lamA * (x**gam)
        xs_a0 = np.array(sorted(float(k) for k in self.summary["A0_theta"].keys()), dtype=float)
        ys_a0 = np.array([float(self.summary["A0_theta"][f"{xv:0.6f}"]) for xv in xs_a0], dtype=float)
        a0 = _interp_linear_extrap_array(theta_eval, xs_a0, ys_a0)
        p_eff = float(self.summary["p"])
        potential = 1.0 - np.cos(theta_eval)
        rho_weight = f0 * potential
        xi = xi_dm * (((1.0 - f_bm) ** lam) + a0 * (vals0["vw"] ** p_eff) * f_bm * g_bm)
        return {
            "theta0": theta_eval,
            "xi": np.asarray(xi, dtype=float),
            "fanh_theta": np.asarray(f0, dtype=float),
            "rho_weight": np.asarray(rho_weight, dtype=float),
            "tp": float(tp),
            "batched": True,
        }

    def _tp_rd(self, hstar: float, beta_over_h: float, vw: float) -> float:
        return float(self._tperc_cache.get(hstar, beta_over_h, vw))

    def _gstar_thermo_correction(self, *, hstar: float, mphi_ev: float) -> dict[str, float]:
        if mphi_ev <= 0.0:
            raise ValueError(f"mphi_ev={mphi_ev:g} must be positive")
        if hstar <= 0.0:
            raise ValueError(f"hstar={hstar:g} must be positive")
        hubble_osc_ev = mphi_ev / 3.0
        hubble_pt_ev = hstar * mphi_ev
        t_osc_ev = _temperature_from_hubble_rd(hubble_osc_ev)
        t_pt_ev = _temperature_from_hubble_rd(hubble_pt_ev)
        g_osc = _gstar_energy(t_osc_ev)
        g_pt = _gstar_energy(t_pt_ev)
        gs_osc = _gstar_entropy(t_osc_ev)
        gs_pt = _gstar_entropy(t_pt_ev)
        factor = float((gs_osc / gs_pt) * ((g_pt / g_osc) ** 0.75))
        return {
            "factor": factor,
            "t_pt_ev": float(t_pt_ev),
            "t_osc_ev": float(t_osc_ev),
            "gstar_pt": float(g_pt),
            "gstar_osc": float(g_osc),
            "gsstar_pt": float(gs_pt),
            "gsstar_osc": float(gs_osc),
        }

    def _eval_core(self, theta0: float, vw: float, hstar: float, beta_over_h: float, *, p: float | None = None,
                   lambda_A: float | None = None, lambda_gamma: float | None = None,
                   a0_theta: float | None = None, s_vw: float | None = None,
                   xi_dm_mode: str = "broken_powerlaw_ftilde",
                   mphi_ev: float | None = None,
                   include_gstar: bool = False) -> dict[str, float]:
        point = np.array([[theta0, vw, hstar, beta_over_h]], dtype=float)
        tp = self._tp_rd(hstar, beta_over_h, vw)
        if xi_dm_mode == "frozen_grid":
            xi_dm = float(self.xi_dm_interp(point)[0])
        elif xi_dm_mode == "broken_powerlaw_ftilde":
            xi_dm = self._xi_dm_from_ftilde(theta0, tp, vw)
        else:
            raise ValueError(f"unknown xi_dm_mode={xi_dm_mode!r}")
        s_eff = float(self._effective_s(vw) if s_vw is None else s_vw)
        plateau_point = np.array([[theta0, vw]], dtype=float)
        plateau_hstar = float(self.kappa_plateau_h_interp(plateau_point)[0])
        plateau_kappa = float(self.kappa_plateau_value_interp(plateau_point)[0])
        kappa_plateau_applied = bool(hstar <= plateau_hstar)
        if kappa_plateau_applied:
            kappa_pilot = plateau_kappa
        else:
            kappa_pilot = float(self.kappa_interp(np.array([[theta0, vw, hstar]], dtype=float))[0])
        kappa_raw = float(s_eff * kappa_pilot)
        geom_vw = float(np.clip(vw, float(self.geom_vw_grid.min()), float(self.geom_vw_grid.max())))
        vw_lo, vw_hi = self._geometry_vw_bracket(geom_vw)
        generated_kappas_all: list[float] = []
        generated_betas_all: list[float] = []
        evals: list[dict[str, float]] = []
        for vw_node in sorted({vw_lo, vw_hi}):
            geom_model = self.geom_models[vw_node]
            kappa_grid = np.asarray(geom_model["kappa_grid"], dtype=float)
            beta_grid = np.asarray(geom_model["beta_grid"], dtype=float)
            use_direct = (
                beta_over_h > float(beta_grid.max()) + 1.0e-12
                or kappa_raw > float(kappa_grid.max()) + 1.0e-12
            )
            if use_direct:
                direct = self._eval_geometry_direct(vw_node, hstar, kappa_raw, beta_over_h)
                if direct["generated_kappa"] is not None:
                    generated_kappas_all.append(float(direct["generated_kappa"]))
                if direct["generated_beta"] is not None:
                    generated_betas_all.append(float(direct["generated_beta"]))
                evals.append(
                    {
                        "vw": vw_node,
                        "f_bm": direct["f_bm"],
                        "g_bm": direct["g_bm"],
                        "kappa": direct["kappa"],
                        "geom_hstar": direct["geom_hstar"],
                        "geom_beta": direct["geom_beta"],
                        "kappa_min": direct["kappa_min"],
                        "kappa_max": direct["kappa_max"],
                    }
                )
            else:
                h_grid = np.asarray(geom_model["h_grid"], dtype=float)
                kappa_node = float(np.clip(kappa_raw, float(kappa_grid.min()), float(kappa_grid.max())))
                geom_hstar = float(np.clip(hstar, float(h_grid.min()), float(h_grid.max())))
                geom_beta = float(np.clip(beta_over_h, float(beta_grid.min()), float(beta_grid.max())))
                gpoint = np.array([[geom_hstar, kappa_node, geom_beta]], dtype=float)
                evals.append(
                    {
                        "vw": vw_node,
                        "f_bm": float(geom_model["f_interp"](gpoint)[0]),
                        "g_bm": float(geom_model["g_interp"](gpoint)[0]),
                        "kappa": kappa_node,
                        "geom_hstar": geom_hstar,
                        "geom_beta": geom_beta,
                        "kappa_min": float(kappa_grid.min()),
                        "kappa_max": float(kappa_grid.max()),
                    }
                )
        geometry_generated_kappas = sorted({float(v) for v in generated_kappas_all})
        geometry_generated_betas = sorted({float(v) for v in generated_betas_all})
        if vw_lo == vw_hi:
            interp = evals[0]
        else:
            w = (geom_vw - vw_lo) / (vw_hi - vw_lo)
            interp = {
                "f_bm": (1.0 - w) * evals[0]["f_bm"] + w * evals[1]["f_bm"],
                "g_bm": (1.0 - w) * evals[0]["g_bm"] + w * evals[1]["g_bm"],
                "kappa": (1.0 - w) * evals[0]["kappa"] + w * evals[1]["kappa"],
                "geom_hstar": (1.0 - w) * evals[0]["geom_hstar"] + w * evals[1]["geom_hstar"],
                "geom_beta": (1.0 - w) * evals[0]["geom_beta"] + w * evals[1]["geom_beta"],
                "kappa_min": min(evals[0]["kappa_min"], evals[1]["kappa_min"]),
                "kappa_max": max(evals[0]["kappa_max"], evals[1]["kappa_max"]),
            }
        kappa = float(interp["kappa"])
        geom_hstar = float(interp["geom_hstar"])
        geom_beta = float(interp["geom_beta"])
        f_bm = float(interp["f_bm"])
        g_bm = float(interp["g_bm"])
        f_bm = float(np.clip(f_bm, 1.0e-12, 1.0 - 1.0e-12))
        g_bm = float(max(g_bm, 0.0))
        lam = self._lambda_theta(theta0, lambda_A=lambda_A, lambda_gamma=lambda_gamma)
        a0 = float(self._effective_a0(theta0) if a0_theta is None else a0_theta)
        p_eff = float(self.summary["p"] if p is None else p)
        fanh_theta = self._fanh_theta(theta0)
        rho_weight = float(fanh_theta * _potential(theta0))
        xi_core = float(xi_dm * (((1.0 - f_bm) ** lam) + a0 * (vw ** p_eff) * f_bm * g_bm))
        if include_gstar:
            if mphi_ev is None:
                raise ValueError("mphi_ev is required when include_gstar=True")
            thermo = self._gstar_thermo_correction(hstar=hstar, mphi_ev=mphi_ev)
        else:
            thermo = {
                "factor": 1.0,
                "t_pt_ev": float("nan"),
                "t_osc_ev": float("nan"),
                "gstar_pt": float("nan"),
                "gstar_osc": float("nan"),
                "gsstar_pt": float("nan"),
                "gsstar_osc": float("nan"),
            }
        xi = float(xi_core * thermo["factor"])
        return {
            "xi": xi,
            "xi_core": xi_core,
            "xi_dm": xi_dm,
            "tp": tp,
            "fanh_theta": fanh_theta,
            "rho_weight": rho_weight,
            "s_vw": s_eff,
            "kappa_pilot": kappa_pilot,
            "kappa": kappa,
            "kappa_raw": kappa_raw,
            "kappa_plateau_applied": kappa_plateau_applied,
            "kappa_plateau_hstar": plateau_hstar,
            "geometry_vw": geom_vw,
            "geometry_hstar": geom_hstar,
            "geometry_beta": geom_beta,
            "geometry_kappa_min": float(interp["kappa_min"]),
            "geometry_kappa_max": float(interp["kappa_max"]),
            "geometry_generated_betas": geometry_generated_betas,
            "geometry_generated_kappas": geometry_generated_kappas,
            "f_bm": f_bm,
            "g_bm": g_bm,
            "lambda_theta": lam,
            "a0_theta": a0,
            "p": p_eff,
            "xi_dm_mode": xi_dm_mode,
            "include_gstar": bool(include_gstar),
            "mphi_ev": None if mphi_ev is None else float(mphi_ev),
            "gstar_thermo_factor": float(thermo["factor"]),
            "t_pt_ev": float(thermo["t_pt_ev"]),
            "t_osc_ev": float(thermo["t_osc_ev"]),
            "gstar_pt": float(thermo["gstar_pt"]),
            "gstar_osc": float(thermo["gstar_osc"]),
            "gsstar_pt": float(thermo["gsstar_pt"]),
            "gsstar_osc": float(thermo["gsstar_osc"]),
        }

    def _prepare_inputs(self, *, hstar: float, vw: float, theta0: float, beta_over_h: float, clip: bool) -> tuple[dict[str, float], bool, list[str]]:
        warnings: list[str] = []
        in_domain = True
        if clip:
            if beta_over_h <= 0.0:
                raise ValueError(f"beta_over_h={beta_over_h:g} must be positive")
            if not (0.0 - DOMAIN_EPS <= theta0 <= math.pi + DOMAIN_EPS):
                raise ValueError(f"theta0={theta0:g} outside physical domain [0, pi]")
            theta0_new = float(theta0)
            theta_lo = float(self.theta_grid.min())
            theta_hi = float(self.theta_grid.max())
            if not (theta_lo - DOMAIN_EPS <= theta0 <= theta_hi + DOMAIN_EPS):
                warnings.append(
                    f"theta0={theta0:g} outside fitted A0/pilot-kappa domain [{theta_lo:g}, {theta_hi:g}]; "
                    "analytic theta laws will be used and A0(theta) will extrapolate linearly"
                )
                in_domain = False
            if vw <= 0.0:
                raise ValueError(f"vw={vw:g} must be positive")
            vw_new = float(vw)
            vw_lo = float(self.vw_grid.min())
            vw_hi = float(self.vw_grid.max())
            if not (vw_lo - DOMAIN_EPS <= vw <= vw_hi + DOMAIN_EPS):
                warnings.append(
                    f"vw={vw:g} outside fitted s(vw) domain [{vw_lo:g}, {vw_hi:g}]; "
                    "compact vw laws will be extrapolated and geometry may use a boundary vw"
                )
                in_domain = False
            h_lo = float(self.hstar_grid.min())
            h_hi = float(self.hstar_grid.max())
            if hstar < h_lo:
                h_new = float(hstar)
                warnings.append(
                    f"hstar={hstar:g} below validated baseline xi_DM grid domain [{h_lo:g}, {h_hi:g}]; "
                    "t_p is computed directly from the RD kernel, while frozen-grid xi_DM will extrapolate "
                    "and low-H kappa plateau logic may be applied"
                )
                in_domain = False
            else:
                h_new, msg = _clip_value(hstar, self.hstar_grid)
                if msg:
                    warnings.append(f"hstar {msg}")
                    in_domain = False
            b_lo = float(self.beta_grid.min())
            b_hi = float(self.beta_grid.max())
            b_new = float(beta_over_h)
            if not (b_lo - DOMAIN_EPS <= beta_over_h <= b_hi + DOMAIN_EPS):
                warnings.append(
                    f"beta_over_h={beta_over_h:g} outside fitted beta domain [{b_lo:g}, {b_hi:g}]; "
                    "t_p is computed directly from the RD kernel and geometry / frozen-grid sectors will continue beyond the tabulated beta range"
                )
                in_domain = False
            return {"theta0": theta0_new, "vw": vw_new, "hstar": h_new, "beta_over_h": b_new}, in_domain, warnings
        if hstar <= 0.0:
            raise ValueError(f"hstar={hstar:g} must be positive")
        if vw <= 0.0:
            raise ValueError(f"vw={vw:g} must be positive")
        if beta_over_h <= 0.0:
            raise ValueError(f"beta_over_h={beta_over_h:g} must be positive")
        if not (0.0 - DOMAIN_EPS <= theta0 <= math.pi + DOMAIN_EPS):
            raise ValueError(f"theta0={theta0:g} outside physical domain [0, pi]")
        b_lo = float(self.beta_grid.min())
        b_hi = float(self.beta_grid.max())
        if not (b_lo - DOMAIN_EPS <= beta_over_h <= b_hi + DOMAIN_EPS):
            warnings.append(
                f"beta_over_h={beta_over_h:g} outside fitted beta domain [{b_lo:g}, {b_hi:g}]; "
                "t_p is computed directly from the RD kernel and geometry / frozen-grid sectors will continue beyond the tabulated beta range"
            )
            in_domain = False
        return {"theta0": theta0, "vw": vw, "hstar": hstar, "beta_over_h": beta_over_h}, in_domain, warnings

    def predict(
        self,
        *,
        hstar: float,
        vw: float,
        theta0: float,
        beta_over_h: float,
        clip: bool = False,
        xi_dm_mode: str = "broken_powerlaw_ftilde",
        mphi_ev: float | None = None,
        include_gstar: bool = False,
    ) -> PredictResult:
        vals, in_domain, warnings = self._prepare_inputs(hstar=hstar, vw=vw, theta0=theta0, beta_over_h=beta_over_h, clip=clip)
        base = self._eval_core(**vals, xi_dm_mode=xi_dm_mode, mphi_ev=mphi_ev, include_gstar=include_gstar)
        status_flags: list[str] = []
        if base["kappa_plateau_applied"]:
            status_flags.append("continued_low_h_kappa_plateau")
            warnings.append(
                f"kappa_pilot frozen to low-H plateau for hstar={vals['hstar']:g} "
                f"(plateau edge {base['kappa_plateau_hstar']:g})"
            )
        if base["geometry_hstar"] != vals["hstar"]:
            if vals["hstar"] < float(self.geom_h_grid.min()):
                status_flags.append("continued_low_h_geometry_boundary")
                warnings.append(
                    f"geometry evaluated at low-H boundary hstar={base['geometry_hstar']:g} "
                    f"because geometry bank starts at {float(self.geom_h_grid.min()):g}"
                )
            elif vals["hstar"] > float(self.geom_h_grid.max()):
                status_flags.append("continued_high_h_geometry_boundary")
                warnings.append(
                    f"geometry evaluated at high-H boundary hstar={base['geometry_hstar']:g} "
                    f"because geometry bank ends at {float(self.geom_h_grid.max()):g}"
                )
        if base["geometry_vw"] != vals["vw"]:
            status_flags.append("continued_vw_geometry_boundary")
            warnings.append(
                f"geometry evaluated at vw={base['geometry_vw']:g} because geometry bank support is "
                f"[{float(self.geom_vw_grid.min()):g}, {float(self.geom_vw_grid.max()):g}]"
            )
        if vals["beta_over_h"] > float(self.beta_grid.max()) + DOMAIN_EPS:
            status_flags.append("continued_high_beta")
            warnings.append(
                f"beta_over_h={vals['beta_over_h']:g} exceeds fitted beta grid upper edge {float(self.beta_grid.max()):g}; "
                "RD-kernel t_p is used directly and interpolated sectors are being continued beyond the tabulated beta range"
            )
        elif vals["beta_over_h"] < float(self.beta_grid.min()) - DOMAIN_EPS:
            status_flags.append("continued_low_beta")
            warnings.append(
                f"beta_over_h={vals['beta_over_h']:g} below fitted beta grid lower edge {float(self.beta_grid.min()):g}; "
                "interpolated sectors are being continued below the tabulated beta range"
            )
        if base["geometry_generated_kappas"]:
            status_flags.append("generated_geometry")
            kappas = ", ".join(f"{k:.3f}" for k in base["geometry_generated_kappas"])
            warnings.append(f"generated geometry-bank kappa slices on demand: {kappas}")
        if base["geometry_generated_betas"]:
            status_flags.append("generated_geometry")
            betas = ", ".join(f"{b:.3f}" for b in base["geometry_generated_betas"])
            warnings.append(f"generated geometry-bank beta slices on demand: {betas}")
        if base["kappa"] != base["kappa_raw"]:
            status_flags.append("continued_kappa_clipped_to_geometry")
            warnings.append(
                f"kappa clipped from {base['kappa_raw']:g} to geometry-bank support "
                f"[{base['geometry_kappa_min']:g}, {base['geometry_kappa_max']:g}]"
            )
        if not (
            float(self.theta_grid.min()) - DOMAIN_EPS <= vals["theta0"] <= float(self.theta_grid.max()) + DOMAIN_EPS
            and float(self.vw_grid.min()) - DOMAIN_EPS <= vals["vw"] <= float(self.vw_grid.max()) + DOMAIN_EPS
            and float(self.hstar_grid.min()) - DOMAIN_EPS <= vals["hstar"] <= float(self.hstar_grid.max()) + DOMAIN_EPS
            and float(self.beta_grid.min()) - DOMAIN_EPS <= vals["beta_over_h"] <= float(self.beta_grid.max()) + DOMAIN_EPS
        ):
            status_flags.append("continued_outside_fit_domain")

        if not status_flags:
            prediction_status = "validated"
        elif any(flag.startswith("continued_") for flag in status_flags):
            prediction_status = "continued"
        else:
            prediction_status = "validated_with_generated_geometry"

        params = self.local["params"]
        sigma_p = float(params["p"]["sigma"])
        sigma_lamA = float(params["lambda_A"]["sigma"])
        sigma_lamg = float(params["lambda_gamma"]["sigma"])
        sigma_s = self._effective_sigma_s(vals["vw"])
        sigma_a0 = self._effective_sigma_a0(vals["theta0"])

        curves = [base["xi"]]
        for sign in (-1.0, 1.0):
            curves.append(
                self._eval_core(
                    **vals,
                    p=base["p"] + sign * sigma_p,
                    xi_dm_mode=xi_dm_mode,
                    mphi_ev=mphi_ev,
                    include_gstar=include_gstar,
                )["xi"]
            )
            curves.append(
                self._eval_core(
                    **vals,
                    lambda_A=float(self.summary["lambda_law"]["A"]) + sign * sigma_lamA,
                    xi_dm_mode=xi_dm_mode,
                    mphi_ev=mphi_ev,
                    include_gstar=include_gstar,
                )["xi"]
            )
            curves.append(
                self._eval_core(
                    **vals,
                    lambda_gamma=float(self.summary["lambda_law"]["gamma"]) + sign * sigma_lamg,
                    xi_dm_mode=xi_dm_mode,
                    mphi_ev=mphi_ev,
                    include_gstar=include_gstar,
                )["xi"]
            )
            curves.append(
                self._eval_core(
                    **vals,
                    s_vw=max(1.0e-8, base["s_vw"] + sign * sigma_s),
                    xi_dm_mode=xi_dm_mode,
                    mphi_ev=mphi_ev,
                    include_gstar=include_gstar,
                )["xi"]
            )
            curves.append(
                self._eval_core(
                    **vals,
                    a0_theta=max(1.0e-8, base["a0_theta"] + sign * sigma_a0),
                    xi_dm_mode=xi_dm_mode,
                    mphi_ev=mphi_ev,
                    include_gstar=include_gstar,
                )["xi"]
            )

        xi_lo = float(np.min(curves))
        xi_hi = float(np.max(curves))
        xi_err = float(max(base["xi"] - xi_lo, xi_hi - base["xi"]))
        return PredictResult(
            xi=base["xi"],
            xi_err=xi_err,
            xi_lo=xi_lo,
            xi_hi=xi_hi,
            xi_dm=base["xi_dm"],
            tp=base["tp"],
            fanh_theta=base["fanh_theta"],
            rho_weight=base["rho_weight"],
            kappa=base["kappa"],
            kappa_raw=base["kappa_raw"],
            kappa_pilot=base["kappa_pilot"],
            kappa_plateau_applied=base["kappa_plateau_applied"],
            kappa_plateau_hstar=base["kappa_plateau_hstar"],
            geometry_vw=base["geometry_vw"],
            geometry_hstar=base["geometry_hstar"],
            geometry_generated_kappas=base["geometry_generated_kappas"],
            geometry_generated_betas=base["geometry_generated_betas"],
            f_bm=base["f_bm"],
            g_bm=base["g_bm"],
            lambda_theta=base["lambda_theta"],
            a0_theta=base["a0_theta"],
            s_vw=base["s_vw"],
            p=base["p"],
            q=self.q,
            xi_dm_mode=base["xi_dm_mode"],
            include_gstar=base["include_gstar"],
            mphi_ev=base["mphi_ev"],
            gstar_thermo_factor=base["gstar_thermo_factor"],
            t_pt_ev=None if math.isnan(base["t_pt_ev"]) else base["t_pt_ev"],
            t_osc_ev=None if math.isnan(base["t_osc_ev"]) else base["t_osc_ev"],
            gstar_pt=None if math.isnan(base["gstar_pt"]) else base["gstar_pt"],
            gstar_osc=None if math.isnan(base["gstar_osc"]) else base["gstar_osc"],
            gsstar_pt=None if math.isnan(base["gsstar_pt"]) else base["gsstar_pt"],
            gsstar_osc=None if math.isnan(base["gsstar_osc"]) else base["gsstar_osc"],
            prediction_status=prediction_status,
            status_flags=status_flags,
            clipped_inputs=vals,
            in_domain=in_domain,
            warnings=warnings,
        )


@lru_cache(maxsize=1)
def load_default_model() -> XiModel:
    return XiModel()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the accepted compact xi model.")
    p.add_argument("--hstar", type=float, required=True, help="H_*/M_phi")
    p.add_argument("--vw", type=float, required=True, help="Wall speed v_w")
    p.add_argument("--theta0", type=float, required=True, help="Initial angle theta_0")
    p.add_argument("--betaH", type=float, required=True, help="beta/H_*")
    p.add_argument("--clip", action="store_true", help="Clip out-of-domain inputs instead of failing")
    p.add_argument(
        "--xi-dm-mode",
        choices=["frozen_grid", "broken_powerlaw_ftilde"],
        default="broken_powerlaw_ftilde",
        help="Homogeneous baseline source for xi_DM",
    )
    p.add_argument("--mphi-ev", type=float, default=None, help="Physical axion mass in eV for the optional g_* correction")
    p.add_argument("--include-gstar", action="store_true", help="Include the thermodynamic g_*, g_{*s} abundance correction")
    p.add_argument("--pretty", action="store_true", help="Pretty-print a compact text summary")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    model = load_default_model()
    result = model.predict(
        hstar=ns.hstar,
        vw=ns.vw,
        theta0=ns.theta0,
        beta_over_h=ns.betaH,
        clip=ns.clip,
        xi_dm_mode=ns.xi_dm_mode,
        mphi_ev=ns.mphi_ev,
        include_gstar=ns.include_gstar,
    )
    if ns.pretty:
        print(f"xi      = {result.xi:.8g}")
        print(f"xi_lo   = {result.xi_lo:.8g}")
        print(f"xi_hi   = {result.xi_hi:.8g}")
        print(f"xi_err  = {result.xi_err:.8g}")
        print(f"xi_dm   = {result.xi_dm:.8g}")
        print(f"tp      = {result.tp:.8g}")
        print(f"kappa   = {result.kappa:.8g}")
        print(f"kappa_raw = {result.kappa_raw:.8g}")
        print(f"f_bm    = {result.f_bm:.8g}")
        print(f"g_bm    = {result.g_bm:.8g}")
        print(f"xi_dm_mode = {result.xi_dm_mode}")
        if result.include_gstar:
            print(f"gstar_factor = {result.gstar_thermo_factor:.8g}")
            print(f"T_PT [eV]    = {result.t_pt_ev:.8g}")
            print(f"T_osc [eV]   = {result.t_osc_ev:.8g}")
        print(f"in_domain = {result.in_domain}")
        if result.warnings:
            print("warnings:")
            for w in result.warnings:
                print(f"  - {w}")
    else:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0
