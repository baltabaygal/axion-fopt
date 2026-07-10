import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Requested grid (small set)
THETA_VALUES = [math.pi / 64.0, math.pi / 128.0, math.pi / 256.0]
H_VALUES = [0.05, 0.1]
BETA_OVER_H_VALUES = [4, 8, 10, 20, 40]
VW_VALUES = [0.5, 0.9]
N_REAL = 5

# Simulation settings
NGRID = 64
NTR_PT = 100_000
NTR_NOPT = 1_000
END_FACTOR = 50.0


def fmt_eta(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_one(H_PT, beta_over_H, theta0, v_w, with_pt, seed, numba_threads=1):
    # Keep each worker single-threaded inside Numba to avoid oversubscription.
    import numba as nb
    from axion_sim import create_simulation_pt_at_zero

    nb.set_num_threads(numba_threads)
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


def build_tasks():
    nopt_tasks = [(th, H) for th in THETA_VALUES for H in H_VALUES]
    pt_tasks = [
        (th, H, bH, vw, r)
        for th in THETA_VALUES
        for H in H_VALUES
        for bH in BETA_OVER_H_VALUES
        for vw in VW_VALUES
        for r in range(N_REAL)
    ]
    return nopt_tasks, pt_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 4), help='process workers')
    parser.add_argument('--numba-threads-per-worker', type=int, default=1, help='Numba threads inside each worker')
    args = parser.parse_args()

    out_dir = Path('/Users/baltabay/Desktop/miniclusters/axion-fopt/sim_core/cosine_sweep_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    nopt_tasks, pt_tasks = build_tasks()
    total = len(nopt_tasks) + len(pt_tasks)

    print('=' * 96)
    print('Small-theta reduced-grid sweep (cosine potential, parallel PT)')
    print(f'theta0: {THETA_VALUES}')
    print(f'H*: {H_VALUES}')
    print(f'beta/H: {BETA_OVER_H_VALUES}')
    print(f'v_w: {VW_VALUES}')
    print(f'N_real: {N_REAL}, Ngrid: {NGRID}, end_time: 50/H*')
    print(f'Workers: {args.workers}, numba_threads_per_worker: {args.numba_threads_per_worker}')
    print(f'Jobs: noPT={len(nopt_tasks)}, PT={len(pt_tasks)}, total={total}')
    print('=' * 96)

    start_all = time.perf_counter()
    done = 0

    # noPT (serial, tiny set)
    noPT = {}
    for th, H in nopt_tasks:
        done += 1
        elapsed = time.perf_counter() - start_all
        avg = elapsed / max(1, done - 1)
        eta = avg * (total - done + 1) if done > 1 else 0
        print(f"\n[{done}/{total}] noPT theta0={th:.8g}, H*={H} | elapsed={fmt_eta(elapsed)} ETA={fmt_eta(eta)}")
        seed = int(th * 1e9) + int(H * 1e6) + 777
        e0, wall, _ = run_one(H, 0.0, th, VW_VALUES[0], with_pt=False, seed=seed, numba_threads=1)
        noPT[(th, H)] = e0
        print(f"  done: rho_noPT={e0:.9e}, wall={wall:.2f}s")

    # PT (parallel)
    per_run = []
    futures = {}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for th, H, bH, vw, r in pt_tasks:
            seed = int(th * 1e9) + int(H * 1e6) + int(bH * 1e4) + int(vw * 100) + r
            fut = ex.submit(run_one, H, bH, th, vw, True, seed, args.numba_threads_per_worker)
            futures[fut] = (th, H, bH, vw, r, seed)

        for fut in as_completed(futures):
            th, H, bH, vw, r, seed = futures[fut]
            ept, wall, bubbles = fut.result()

            done += 1
            elapsed = time.perf_counter() - start_all
            avg = elapsed / done
            eta = avg * (total - done)

            rho0 = noPT[(th, H)]
            ratio = ept / rho0 if rho0 != 0 else float('nan')

            print(
                f"[{done}/{total}] PT theta0={th:.8g}, H*={H}, beta/H={bH}, v_w={vw}, r={r} "
                f"| ratio={ratio:.6f}, bubbles={bubbles}, run={wall:.2f}s | ETA={fmt_eta(eta)}"
            )

            per_run.append({
                'theta0': th,
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

    summary = []
    for th in THETA_VALUES:
        for H in H_VALUES:
            for bH in BETA_OVER_H_VALUES:
                for vw in VW_VALUES:
                    arr = np.array([
                        x['ratio'] for x in per_run
                        if x['theta0'] == th and x['H_PT'] == H and x['beta_over_H'] == bH and x['v_w'] == vw
                    ], dtype=float)
                    summary.append({
                        'theta0': th,
                        'H_PT': H,
                        'beta_over_H': bH,
                        'beta': bH * H,
                        'v_w': vw,
                        'rho_noPT': noPT[(th, H)],
                        'mean_ratio': float(np.mean(arr)),
                        'std_ratio': float(np.std(arr, ddof=0)),
                        'N_samples': int(arr.size),
                    })

    ts = time.strftime('%Y%m%d_%H%M%S')
    jpath = out_dir / f'smalltheta_reduced_grid_parallel_{ts}.json'
    tpath = out_dir / f'smalltheta_reduced_grid_parallel_{ts}.tsv'
    with open(jpath, 'w') as f:
        json.dump({'config': {
            'theta_values': THETA_VALUES,
            'H_values': H_VALUES,
            'beta_over_H_values': BETA_OVER_H_VALUES,
            'v_w_values': VW_VALUES,
            'N_real': N_REAL,
            'Ngrid': NGRID,
            'end_time_factor': END_FACTOR,
            'workers': args.workers,
            'numba_threads_per_worker': args.numba_threads_per_worker,
        }, 'per_run': per_run, 'summary': summary}, f, indent=2)

    with open(tpath, 'w') as f:
        f.write('theta0\tH_PT\tbeta_over_H\tbeta\tv_w\trho_noPT\tmean_ratio\tstd_ratio\tN_samples\n')
        for s in summary:
            f.write(
                f"{s['theta0']:.12e}\t{s['H_PT']}\t{s['beta_over_H']}\t{s['beta']}\t{s['v_w']}\t"
                f"{s['rho_noPT']:.12e}\t{s['mean_ratio']:.12e}\t{s['std_ratio']:.12e}\t{s['N_samples']}\n"
            )

    total_elapsed = time.perf_counter() - start_all
    print('\n' + '=' * 96)
    print(f'DONE in {fmt_eta(total_elapsed)}')
    print(f'JSON: {jpath}')
    print(f'TSV : {tpath}')
    print('=' * 96)


if __name__ == '__main__':
    main()
