import json
import math
import time
from pathlib import Path

import numpy as np
from axion_sim import create_simulation_pt_at_zero

# Requested subset
H_VALUES = [0.05, 0.1, 1.5]
BETA_OVER_H_VALUES = [4, 10, 20, 40]
VW_VALUES = [0.3, 0.4, 0.7]
THETA_VALUES = [math.pi / 32.0, math.pi / 6.0]

# Realization policy
N_REAL_DEFAULT = 5
N_REAL_BY_H = {1.5: 1}  # H*=1.5 uses only one realization

# Simulation controls
NGRID = 64
NTR_PT = 100_000
NTR_NOPT = 1_000
END_FACTOR = 50.0


def n_real_for_H(H):
    for k, v in N_REAL_BY_H.items():
        if abs(H - k) < 1e-12:
            return v
    return N_REAL_DEFAULT


def format_eta(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run_sim(H_PT, beta_over_H, theta0, v_w, with_pt, seed):
    np.random.seed(seed)
    if with_pt:
        beta = beta_over_H * H_PT
        enable_nucleation = True
        mb = 0.0
        ntr = NTR_PT
    else:
        beta = 0.0
        enable_nucleation = False
        mb = 1.0
        ntr = NTR_NOPT

    sim, tau_f = create_simulation_pt_at_zero(
        H_PT=H_PT,
        beta=beta,
        start_time=0.0,
        end_time=END_FACTOR / H_PT,
        nucleation_mode='standard',
        Ngrid=NGRID,
        num_tracers=ntr,
        enable_nucleation=enable_nucleation,
        mb=mb,
        theta0_initial=theta0,
        v_bubble=v_w,
        spatial_hash_cells=8,
        energy_save_interval=10,
        checkpoint_interval=0,
    )

    t0 = time.perf_counter()
    res = sim.run_simulation(tau_f, progress_bar=False, save_interval=10)
    wall = time.perf_counter() - t0
    final_total = float(res['energy_history'][-1]['total_energy'])
    bubbles = int(sim.bubble_manager.count)
    return final_total, wall, bubbles


def main():
    out_dir = Path('/Users/baltabay/Desktop/miniclusters/axion-fopt/sim_core/cosine_sweep_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build PT task list with mixed realizations
    pt_tasks = []
    for theta0 in THETA_VALUES:
        for H in H_VALUES:
            n_real = n_real_for_H(H)
            for bH in BETA_OVER_H_VALUES:
                for vw in VW_VALUES:
                    for r in range(n_real):
                        pt_tasks.append((theta0, H, bH, vw, r))

    # one noPT per (theta0, H)
    nopt_tasks = [(theta0, H) for theta0 in THETA_VALUES for H in H_VALUES]

    total_jobs = len(nopt_tasks) + len(pt_tasks)

    print('=' * 86)
    print('Subset cosine sweep with mixed realizations')
    print(f'H*: {H_VALUES}')
    print(f'beta/H: {BETA_OVER_H_VALUES}')
    print(f'v_w: {VW_VALUES}')
    print('theta0: [pi/32, pi/6]')
    print(f'Realizations: default={N_REAL_DEFAULT}, override={N_REAL_BY_H}')
    print(f'Jobs: noPT={len(nopt_tasks)}, PT={len(pt_tasks)}, total={total_jobs}')
    print('=' * 86)

    run_rows = []
    noPT = {}

    completed = 0
    wall_start = time.perf_counter()

    # noPT phase
    for theta0, H in nopt_tasks:
        completed += 1
        avg = (time.perf_counter() - wall_start) / max(1, completed - 1)
        eta = avg * (total_jobs - completed + 1) if completed > 1 else 0.0
        print(f"\n[{completed}/{total_jobs}] noPT theta0={theta0:.10f}, H*={H} | ETA {format_eta(eta)}")

        seed = int(1_000_000 * theta0) + int(10_000 * H) + 777
        e0, wall, bubbles = run_sim(H, 0.0, theta0, VW_VALUES[0], with_pt=False, seed=seed)
        noPT[(theta0, H)] = e0
        print(f"  done noPT: rho={e0:.9e}, wall={wall:.2f}s")

    # PT phase
    for theta0, H, bH, vw, r in pt_tasks:
        completed += 1
        elapsed = time.perf_counter() - wall_start
        avg = elapsed / max(1, completed - 1)
        eta = avg * (total_jobs - completed + 1)
        print(
            f"\n[{completed}/{total_jobs}] PT theta0={theta0:.10f}, H*={H}, beta/H={bH}, v_w={vw}, r={r} | "
            f"elapsed {format_eta(elapsed)} | ETA {format_eta(eta)}"
        )

        seed = int(1_000_000 * theta0) + int(100_000 * H) + int(1_000 * bH) + int(10 * vw) + r
        ept, wall, bubbles = run_sim(H, bH, theta0, vw, with_pt=True, seed=seed)
        rho0 = noPT[(theta0, H)]
        ratio = ept / rho0 if rho0 != 0 else float('nan')

        print(f"  done PT: rho={ept:.9e}, ratio={ratio:.6f}, bubbles={bubbles}, wall={wall:.2f}s")

        run_rows.append({
            'theta0': theta0,
            'H_PT': H,
            'beta_over_H': bH,
            'beta': bH * H,
            'v_w': vw,
            'realization': r,
            'seed': seed,
            'rho_PT': ept,
            'rho_noPT': rho0,
            'ratio': ratio,
            'num_bubbles': bubbles,
            'wall_seconds': wall,
        })

    # summarize
    summary = []
    for theta0 in THETA_VALUES:
        for H in H_VALUES:
            for bH in BETA_OVER_H_VALUES:
                for vw in VW_VALUES:
                    sub = [x for x in run_rows
                           if abs(x['theta0'] - theta0) < 1e-12
                           and abs(x['H_PT'] - H) < 1e-12
                           and abs(x['beta_over_H'] - bH) < 1e-12
                           and abs(x['v_w'] - vw) < 1e-12]
                    arr = np.array([x['ratio'] for x in sub], dtype=float)
                    summary.append({
                        'theta0': theta0,
                        'H_PT': H,
                        'beta_over_H': bH,
                        'beta': bH * H,
                        'v_w': vw,
                        'rho_noPT': float(noPT[(theta0, H)]),
                        'mean_ratio': float(np.mean(arr)),
                        'std_ratio': float(np.std(arr, ddof=0)),
                        'N_samples': int(arr.size),
                    })

    ts = time.strftime('%Y%m%d_%H%M%S')
    jpath = out_dir / f'subset_mixed_realizations_{ts}.json'
    tpath = out_dir / f'subset_mixed_realizations_{ts}.tsv'

    payload = {
        'config': {
            'potential': 'cosine',
            'H_values': H_VALUES,
            'beta_over_H_values': BETA_OVER_H_VALUES,
            'v_w_values': VW_VALUES,
            'theta_values': THETA_VALUES,
            'Ngrid': NGRID,
            'end_time_factor': END_FACTOR,
            'n_real_default': N_REAL_DEFAULT,
            'n_real_by_H': N_REAL_BY_H,
        },
        'noPT': {f"theta={k[0]:.12f},H={k[1]}": v for k, v in noPT.items()},
        'per_run': run_rows,
        'summary': summary,
    }

    with open(jpath, 'w') as f:
        json.dump(payload, f, indent=2)

    with open(tpath, 'w') as f:
        f.write('theta0\tH_PT\tbeta_over_H\tbeta\tv_w\trho_noPT\tmean_ratio\tstd_ratio\tN_samples\n')
        for s in summary:
            f.write(
                f"{s['theta0']:.12f}\t{s['H_PT']}\t{s['beta_over_H']}\t{s['beta']}\t{s['v_w']}\t"
                f"{s['rho_noPT']:.12e}\t{s['mean_ratio']:.12e}\t{s['std_ratio']:.12e}\t{s['N_samples']}\n"
            )

    total_elapsed = time.perf_counter() - wall_start
    print('\n' + '=' * 86)
    print(f'Done. total elapsed: {format_eta(total_elapsed)}')
    print(f'JSON: {jpath}')
    print(f'TSV : {tpath}')
    print('=' * 86)


if __name__ == '__main__':
    main()
