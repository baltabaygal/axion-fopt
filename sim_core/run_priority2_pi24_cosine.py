import json
import math
import time
from pathlib import Path

import numpy as np
from axion_sim import create_simulation_pt_at_zero

THETA0 = math.pi / 24.0
H_VALUES = [0.05, 0.1]
BETA_OVER_H_VALUES = [4, 8, 16, 40]
VW_VALUES = [0.7, 0.9]
N_REAL = 5

NGRID = 64
NTR_PT = 100_000
NTR_NOPT = 1_000
END_FACTOR = 50.0


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
    return final_total, wall, int(sim.bubble_manager.count)


def main():
    out_dir = Path('/Users/baltabay/Desktop/miniclusters/axion-fopt/sim_core/cosine_sweep_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 76)
    print('Priority-2 cosine sweep (theta0=pi/24)')
    print(f'H*: {H_VALUES}')
    print(f'beta/H: {BETA_OVER_H_VALUES}')
    print(f'v_w: {VW_VALUES}')
    print(f'PT realizations per point: {N_REAL}')
    print('noPT baselines: single run per H*')
    print('=' * 76)

    # One noPT per H* (use v_w=0.7 placeholder, irrelevant when nucleation off)
    noPT = {}
    for i, H in enumerate(H_VALUES):
        seed = 7770 + i
        print(f"\\n[noPT] H*={H}, seed={seed}")
        e0, wall, nb = run_sim(H, 0.0, THETA0, v_w=VW_VALUES[0], with_pt=False, seed=seed)
        noPT[H] = {
            'final_total_energy': e0,
            'wall_seconds': wall,
            'num_bubbles': nb,
            'seed': seed,
        }
        print(f"  rho_noPT={e0:.9e}, wall={wall:.2f}s")

    per_run = []
    tasks = [(H, bH, vw, r) for H in H_VALUES for bH in BETA_OVER_H_VALUES for vw in VW_VALUES for r in range(N_REAL)]

    for idx, (H, bH, vw, r) in enumerate(tasks, 1):
        seed = int(100000*H) + int(1000*bH) + int(10*vw) + r
        print(f"\\n[{idx}/{len(tasks)}] H*={H}, beta/H={bH}, v_w={vw}, r={r}, seed={seed}")
        ept, wall, nb = run_sim(H, bH, THETA0, v_w=vw, with_pt=True, seed=seed)
        rho0 = noPT[H]['final_total_energy']
        ratio = ept / rho0 if rho0 != 0 else float('nan')
        print(f"  rho_PT={ept:.9e}, ratio={ratio:.6f}, bubbles={nb}, wall={wall:.2f}s")

        per_run.append({
            'theta0': THETA0,
            'H_PT': H,
            'beta_over_H': bH,
            'beta': bH * H,
            'v_w': vw,
            'realization': r,
            'seed': seed,
            'rho_PT': ept,
            'rho_noPT': rho0,
            'ratio': ratio,
            'num_bubbles': nb,
            'wall_seconds': wall,
        })

    # aggregate
    summary = []
    for H in H_VALUES:
        for bH in BETA_OVER_H_VALUES:
            for vw in VW_VALUES:
                arr = np.array([x['ratio'] for x in per_run if x['H_PT']==H and x['beta_over_H']==bH and x['v_w']==vw], dtype=float)
                summary.append({
                    'theta0': THETA0,
                    'H_PT': H,
                    'beta_over_H': bH,
                    'beta': bH * H,
                    'v_w': vw,
                    'mean_ratio': float(np.mean(arr)),
                    'std_ratio': float(np.std(arr, ddof=0)),
                    'N_samples': int(arr.size),
                    'rho_noPT': float(noPT[H]['final_total_energy']),
                })

    payload = {
        'config': {
            'potential': 'cosine',
            'theta0': THETA0,
            'H_values': H_VALUES,
            'beta_over_H_values': BETA_OVER_H_VALUES,
            'v_w_values': VW_VALUES,
            'n_realizations': N_REAL,
            'Ngrid': NGRID,
            'num_tracers_pt': NTR_PT,
            'num_tracers_noPT': NTR_NOPT,
            'end_time_factor': END_FACTOR,
        },
        'noPT_per_H': noPT,
        'per_run': per_run,
        'summary': summary,
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    jpath = out_dir / f'priority2_pi24_cosine_{ts}.json'
    tpath = out_dir / f'priority2_pi24_cosine_{ts}.tsv'

    with open(jpath, 'w') as f:
        json.dump(payload, f, indent=2)

    with open(tpath, 'w') as f:
        f.write('theta0\tH_PT\tbeta_over_H\tbeta\tv_w\trho_noPT\tmean_ratio\tstd_ratio\tN_samples\n')
        for s in summary:
            f.write(
                f"{s['theta0']:.12f}\t{s['H_PT']}\t{s['beta_over_H']}\t{s['beta']}\t{s['v_w']}\t"
                f"{s['rho_noPT']:.12e}\t{s['mean_ratio']:.12e}\t{s['std_ratio']:.12e}\t{s['N_samples']}\n"
            )

    print('\n' + '='*76)
    print('SUMMARY TABLE')
    print('theta0\tH_PT\tbeta_over_H\tbeta\tv_w\trho_noPT\tmean_ratio\tstd_ratio\tN')
    for s in summary:
        print(
            f"{s['theta0']:.12f}\t{s['H_PT']}\t{s['beta_over_H']}\t{s['beta']}\t{s['v_w']}\t"
            f"{s['rho_noPT']:.6e}\t{s['mean_ratio']:.6f}\t{s['std_ratio']:.6f}\t{s['N_samples']}"
        )
    print('='*76)
    print(f'JSON: {jpath}')
    print(f'TSV : {tpath}')


if __name__ == '__main__':
    main()
