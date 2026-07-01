"""XiModel — semi-analytic enhancement factor xi for axion dark matter from a
first-order phase transition (manuscript Eq. for xi).

    xi = xi_dm * [ (1 - eps_BM)^lambda + A0 (gamma_w v_w)^p_mink (a_p/a_*)^{-2} G_BM ]

with
    xi_dm  = [ 1 + (C0 * F_inf/F_anh * tau^{3/2})^r ]^{1/r},   r = r0 + r1 v_w,
    tau    = (2/3) t_p,          t_p = percolation time (radiation domination),
    eps_BM = f_BM(kappa),        G_BM = eps_BM <1/(M_phi R_c)>,
    kappa  = s v_w * kappa_pilot(theta,v_w,H_*)^{q_k},
    (a_p/a_*)^{-2} = [1 + (sqrt3/2) H_* R_c / v_w]^{-2}.

Geometry (f_BM, G_BM, R_c) is computed on-the-fly by integrating a single
expanding bubble with an adaptive comoving support R_max (essential at low H_*);
the anharmonic factors F_anh, F_inf use the manuscript log-form.  Parameters come
from a global fit to lattice simulations (data/summary.json).

Public reference implementation for https://github.com/baltabaygal/axion-fopt .
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import _geom_compact, _percolation

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


@dataclass
class PredictResult:
    """Result of :meth:`XiModel.predict`."""
    xi: float
    xi_dm: float
    f_bm: float
    g_bm: float
    rc_mean: float
    kappa: float
    tp: float
    theta0: float
    vw: float
    hstar: float
    beta_over_h: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class XiModel:
    """Semi-analytic model for the abundance enhancement factor xi.

    Parameters
    ----------
    data_dir : path, optional
        Directory containing ``summary.json``, ``geometry_bank/`` and
        ``lattice_tables/``.  Defaults to the packaged data.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.geom_dir = self.data_dir / "geometry_bank"
        self.lattice_dir = self.data_dir / "lattice_tables"

        P = _load_json(self.data_dir / "summary.json")
        self.summary = P
        self.C0 = float(P["xi_dm"]["C0"])
        self.r0 = float(P["xi_dm"]["r0"])
        self.r1 = float(P["xi_dm"]["r1"])
        self.a_exp = float(P["xi_dm"].get("a", 1.5))
        self.xi_floor = float(P["xi_dm"].get("xi_floor", 1.0))
        self.lam = float(P["lambda"])          # (1 - eps_BM)^lambda ; constant
        self.A0 = float(P["A0"])               # bubble-misalignment amplitude ; constant
        self.p_mink = float(P["velocity"]["p_mink"])
        self.s = float(P["kappa"]["s"])        # kappa = s v_w kappa_pilot^{q_k}
        self.q_k = float(P["kappa"]["q_k"])
        self.kappa_eps = float(P["kappa"].get("analytic_eps", 2.23299))
        self.fanh_eps = float(P["fanh"]["eps0"])
        self.fanh_alpha = float(P["fanh"]["alpha0"])
        self.finf_eps = float(P["finf"]["eps_inf"])
        self.finf_alpha = float(P["finf"]["alpha_inf"])

        self._geom = _geom_compact
        self.geom_summary = _load_json(self.geom_dir / "geom_compact_summary.json")
        self._load_geometry_bank()
        self._load_kappa_analytic()
        self._otf_cache: dict[tuple, tuple[float, float, float]] = {}

    # ── anharmonic factors (manuscript log-form, normalised to 1 at theta=0) ──
    def _fanh(self, theta0: float) -> float:
        xr = min(abs(theta0) / math.pi, 1.0 - 1e-12)
        return math.log(math.e / max(1.0 - xr ** self.fanh_eps, 1e-12)) ** self.fanh_alpha

    def _finf(self, theta0: float) -> float:
        xr = min(abs(theta0) / math.pi, 1.0 - 1e-12)
        return math.log(math.e / max(1.0 - xr ** self.finf_eps, 1e-12)) ** self.finf_alpha

    # ── percolation time t_p (radiation domination) ──────────────────────────
    def _tp_rd(self, hstar: float, beta_over_h: float, vw: float) -> float:
        return float(_percolation.t_perc_RD(float(hstar), float(beta_over_h), float(vw)))

    # ── analytic minimum-collision-radius pilot kappa ────────────────────────
    def _load_kappa_analytic(self) -> None:
        """kappa_pilot = A(v_w,H_b) * xlog(theta0)^{beta(v_w,H_b)},
        xlog = ln(e/(1-(theta0/pi)^eps)); A,beta fit per (v_w,H_b) slice of
        pilot_kappa.csv by log-log regression, then interpolated on (v_w,H_b)."""
        eps = self.kappa_eps
        rows = np.genfromtxt(self.lattice_dir / "pilot_kappa.csv", delimiter=",", names=True)
        vws = np.array(sorted({float(v) for v in rows["vw"]}), dtype=float)
        hbs = np.array(sorted({float(v) for v in rows["Hb"]}), dtype=float)
        A_arr = np.full((len(vws), len(hbs)), np.nan)
        B_arr = np.full((len(vws), len(hbs)), np.nan)

        def _xl(t: np.ndarray) -> np.ndarray:
            xr = np.minimum(np.abs(np.asarray(t, dtype=float)) / np.pi, 1.0 - 1e-12)
            return np.log(np.e / np.maximum(1.0 - xr ** eps, 1e-12))

        for i, vw in enumerate(vws):
            for j, hb in enumerate(hbs):
                m = (np.abs(rows["vw"] - vw) < 1e-6) & (np.abs(rows["Hb"] - hb) < 1e-9)
                th = rows["theta0"][m].astype(float)
                kp = rows["kappa"][m].astype(float)
                if len(th) < 6:
                    continue
                o = np.argsort(th)
                b, lnA = np.polyfit(np.log(_xl(th[o])), np.log(kp[o]), 1)
                A_arr[i, j] = math.exp(lnA)
                B_arr[i, j] = b
        for arr in (A_arr, B_arr):
            for i in range(len(vws)):
                nan = np.isnan(arr[i])
                if nan.any() and (~nan).any():
                    arr[i, nan] = np.interp(hbs[nan], hbs[~nan], arr[i, ~nan])
        self._k_vws, self._k_hbs = vws, hbs
        self._kA = RegularGridInterpolator((vws, hbs), A_arr, bounds_error=False, fill_value=None)
        self._kB = RegularGridInterpolator((vws, hbs), B_arr, bounds_error=False, fill_value=None)

    def _kappa_pilot_analytic(self, theta0: float, vw: float, hstar: float) -> float:
        vc = float(np.clip(vw, self._k_vws[0], self._k_vws[-1]))
        hc = float(np.clip(hstar, self._k_hbs[0], self._k_hbs[-1]))
        A = float(self._kA([[vc, hc]])[0])
        B = float(self._kB([[vc, hc]])[0])
        xr = min(abs(theta0) / math.pi, 1.0 - 1e-12)
        xl = math.log(math.e / max(1.0 - xr ** self.kappa_eps, 1e-12))
        return float(A * max(xl, 1e-12) ** B)

    # ── geometry bank (fallback above the simulated H* ceiling) ───────────────
    def _load_geometry_bank(self) -> None:
        files = sorted(self.geom_dir.glob("BM_geometry_RD_kappa_*_vw*_oneloop.json"))
        vws = sorted({float(p.stem.split("_")[5].replace("vw", "")) for p in files})
        self.geom_vw_grid = np.array(vws, dtype=float)
        self.geom_models: dict[float, dict[str, Any]] = {}
        payloads = [(p, _load_json(p)) for p in files]
        for vw in vws:
            vwp = [(float(p.stem.split("_")[4]), pl) for p, pl in payloads
                   if abs(float(p.stem.split("_")[5].replace("vw", "")) - vw) < 1e-9]
            kappas = np.array(sorted(k for k, _ in vwp), dtype=float)
            hset = set.intersection(*[{float(k) for k in pl.keys()} for _, pl in vwp])
            hs = np.array(sorted(hset), dtype=float)
            bset = set.intersection(*[{float(b) for _, hp in pl.items() for b in hp} for _, pl in vwp])
            bs = np.array(sorted(bset), dtype=float)
            f = np.full((len(hs), len(kappas), len(bs)), np.nan)
            g = np.full_like(f, np.nan)
            ik = {k: i for i, k in enumerate(kappas)}
            ih = {h: i for i, h in enumerate(hs)}
            ib = {b: i for i, b in enumerate(bs)}
            for kap, pl in vwp:
                for hk, hp in pl.items():
                    h = float(hk)
                    if h not in ih:
                        continue
                    for bk, cell in hp.items():
                        b = float(bk)
                        if b not in ib:
                            continue
                        f[ih[h], ik[kap], ib[b]] = float(cell["f_BM"])
                        g[ih[h], ik[kap], ib[b]] = float(cell["G_BM"])
            self.geom_models[vw] = {
                "h_grid": hs, "kappa_grid": kappas, "beta_grid": bs,
                "f_interp": RegularGridInterpolator((hs, kappas, bs), f, bounds_error=False, fill_value=None),
                "g_interp": RegularGridInterpolator((hs, kappas, bs), g, bounds_error=False, fill_value=None),
            }

    def _nearest_geom_vw(self, vw: float) -> float:
        g = self.geom_vw_grid
        return float(g[int(np.argmin(np.abs(g - vw)))])

    def _adaptive_rmax(self, vw: float, hstar: float, beta_over_h: float) -> tuple[float, int]:
        """Comoving support (R_max, N_R) for the single-bubble geometry integral.
        R_max must grow ~1/H_* at low H_*, else f_BM/G_BM/R_c are truncated."""
        gs = self.geom_summary
        base_rmax = float(gs["rmax_by_vw"][f"{self._nearest_geom_vw(vw):.1f}"])
        base_nr = int(gs["N_R"])
        beta_phys = float(beta_over_h * hstar)
        if hstar <= 0.0 or beta_phys <= 0.0:
            return base_rmax, base_nr
        try:
            bg = self._geom.compute_background_RD(
                b=float(beta_over_h), H_star=float(hstar), v_w=float(vw),
                P_crit=float(gs["P_crit"]), x_min_factor=float(gs["x_min_factor"]),
                x_max=float(gs["x_max"]), dx=float(gs["dx"]))
            Farr = np.asarray(bg["F"], dtype=float)
            dyn = (float(vw) / beta_phys) * (float(bg["F_obs"]) - float(Farr[0]))
        except Exception:
            return base_rmax, base_nr
        rmax = max(base_rmax, 1.2 * dyn)
        if rmax <= base_rmax:
            return base_rmax, base_nr
        ratio = max(1.0, rmax / max(base_rmax, 1e-12))
        nr = int(min(40000, max(base_nr, int(base_nr * min(ratio, 20.0)))))
        return float(rmax), nr

    def _geometry(self, vw: float, hstar: float, kappa: float, beta_over_h: float) -> tuple[float, float, float]:
        """Return (f_BM, G_BM, R_c) via on-the-fly single-bubble integration with
        adaptive support.  Above the simulated H* ceiling, fall back to the bank."""
        gm = self.geom_models[self._nearest_geom_vw(vw)]
        h_ceiling = float(np.asarray(gm["h_grid"]).max())
        key = (round(vw, 4), round(hstar, 10), round(float(kappa), 5), round(beta_over_h, 4))
        if key in self._otf_cache:
            return self._otf_cache[key]
        h_eval = min(hstar, h_ceiling)
        try:
            rmax, nr = self._adaptive_rmax(vw, h_eval, beta_over_h)
            gs = self.geom_summary
            cell = self._geom.compute_geometry_point(
                v_w=float(vw), kappa=float(kappa), H_star=float(h_eval), b=float(beta_over_h),
                Rmax_global=float(rmax), mphi=float(gs["mphi"]), P_crit=float(gs["P_crit"]),
                x_min_factor=float(gs["x_min_factor"]), x_max=float(gs["x_max"]),
                dx=float(gs["dx"]), N_R=int(nr))
            f_bm = float(np.clip(cell["f_BM"], 1e-12, 1.0 - 1e-12))
            g_bm = float(max(cell["G_BM"], 0.0))
            rc = float(cell["Rc_mean_kappa0"])
        except Exception:
            kg = np.asarray(gm["kappa_grid"]); bg = np.asarray(gm["beta_grid"])
            pt = np.array([[h_ceiling, float(np.clip(kappa, kg[0], kg[-1])),
                            float(np.clip(beta_over_h, bg[0], bg[-1]))]])
            f_bm = float(np.clip(gm["f_interp"](pt)[0], 1e-12, 1.0 - 1e-12))
            g_bm = float(max(gm["g_interp"](pt)[0], 0.0))
            rc = g_bm / max(f_bm, 1e-12)
        self._otf_cache[key] = (f_bm, g_bm, rc)
        return f_bm, g_bm, rc

    # ── public prediction ────────────────────────────────────────────────────
    def predict(self, hstar: float, vw: float, theta0: float, beta_over_h: float,
                clip: bool = True) -> PredictResult:
        """Enhancement factor xi = rho_PT / rho_noPT at late times.

        Parameters
        ----------
        hstar : H_*/M_phi;   vw : bubble wall velocity in (0,1);
        theta0 : initial misalignment angle in [0, pi);   beta_over_h : beta / H_*;
        clip : clip xi to its physical floor (>= xi_floor).
        """
        theta0 = abs(float(theta0))
        tp = self._tp_rd(hstar, beta_over_h, vw)
        tau = (2.0 / 3.0) * tp

        f0 = self._fanh(theta0)
        fi = self._finf(theta0)
        r = self.r0 + self.r1 * vw
        z = max(self.C0 * fi / f0 * max(tau, 1e-30) ** self.a_exp, 1e-30)
        xi_dm = max(self.xi_floor ** r + z ** r, 1e-300) ** (1.0 / r)

        kp = self._kappa_pilot_analytic(theta0, vw, hstar)
        kappa = self.s * vw * max(kp, 1e-30) ** self.q_k

        f_bm, g_bm, rc = self._geometry(vw, hstar, kappa, beta_over_h)

        vwg = min(vw, 0.9999)
        gamma_w_vw = vwg / math.sqrt(max(1.0 - vwg ** 2, 1e-8))
        a_ratio = 1.0 + (math.sqrt(3.0) / 2.0) * hstar * rc / max(vw, 1e-9)
        v_factor = (gamma_w_vw ** self.p_mink) / a_ratio ** 2

        additive = self.A0 * v_factor * g_bm
        xi = xi_dm * max((1.0 - f_bm) ** self.lam + additive, 1e-20)
        if clip:
            xi = max(xi, self.xi_floor)
        return PredictResult(xi=float(xi), xi_dm=float(xi_dm), f_bm=f_bm, g_bm=g_bm,
                             rc_mean=rc, kappa=float(kappa), tp=float(tp), theta0=theta0,
                             vw=float(vw), hstar=float(hstar), beta_over_h=float(beta_over_h))


def load_default_model() -> XiModel:
    """Load :class:`XiModel` with the packaged fit and geometry data."""
    return XiModel()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: print xi for given parameters."""
    ap = argparse.ArgumentParser(description="Predict the axion abundance enhancement factor xi.")
    ap.add_argument("--hstar", type=float, required=True, help="H_*/M_phi")
    ap.add_argument("--vw", type=float, required=True, help="bubble wall velocity")
    ap.add_argument("--theta0", type=float, required=True, help="initial misalignment angle [0, pi)")
    ap.add_argument("--beta-over-h", type=float, required=True, help="beta / H_*")
    ap.add_argument("--no-clip", action="store_true", help="do not clip xi to its floor")
    args = ap.parse_args(argv)
    model = load_default_model()
    res = model.predict(hstar=args.hstar, vw=args.vw, theta0=args.theta0,
                        beta_over_h=args.beta_over_h, clip=not args.no_clip)
    print(json.dumps(res.to_dict(), indent=2))
    return 0
