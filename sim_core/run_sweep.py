"""
Serial parameter sweep over (H_PT, β/H, θ_0, realization).

Runs one simulation at a time, saves the final energy to a JSON file,
then moves to the next parameter point.  Representative energy-evolution
plots are generated in a second pass after the sweep completes.

Usage
-----
    python run_sweep.py

Edit the parameter lists at the bottom of this file before running.
Results are written to the directory specified by `output_dir`.
"""

import json
import os
from dataclasses import asdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from axion_sim import create_simulation_pt_at_zero


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_single_sim(params):
    """
    Run one simulation and return a result dict.

    Parameters
    ----------
    params : tuple
        (H_PT, beta_over_H, theta0, with_pt, realization)

    Returns
    -------
    dict with keys: status, H_PT, beta_over_H, theta0_initial, with_pt,
                    realization, final_time, kinetic, gradient, potential,
                    total, scale_factor, num_bubbles, config.
    On failure: status = 'ERROR' and keys error, traceback.
    """
    H_PT, beta_over_H, theta0, with_pt, realization = params

    try:
        if with_pt:
            beta              = beta_over_H * H_PT
            enable_nucleation = True
            mb_value          = 0.0
        else:
            beta              = 0.0
            enable_nucleation = False
            mb_value          = 1.0

        sim, tau_final = create_simulation_pt_at_zero(
            H_PT=H_PT,
            beta=beta,
            start_time=0.0,
            end_time=10.0 / H_PT,
            nucleation_mode='standard',
            Ngrid=64,
            num_tracers=100_000 if enable_nucleation else 1_000,
            enable_nucleation=enable_nucleation,
            mb=mb_value,
            theta0_initial=theta0,
            v_bubble=0.6,
            spatial_hash_cells=8,
            energy_save_interval=10,
            checkpoint_interval=0,
        )

        results = sim.run_simulation(tau_final, progress_bar=False,
                                     save_interval=10)

        if results['times_cosmic'] and results['energy_history']:
            final_time   = results['config'].m0 * results['times_cosmic'][-1]
            final_energy = results['energy_history'][-1]
            config_dict  = asdict(sim.config)

            return _to_native({
                'H_PT':           H_PT,
                'beta_over_H':    beta_over_H,
                'theta0_initial': theta0,
                'with_pt':        with_pt,
                'realization':    realization,
                'final_time':     final_time,
                'kinetic':        final_energy['kinetic_energy'],
                'gradient':       final_energy['gradient_energy'],
                'potential':      final_energy['potential_energy'],
                'total':          final_energy['total_energy'],
                'scale_factor':   final_energy['scale_factor'],
                'num_bubbles':    sim.bubble_manager.count,
                'status':         'SUCCESS',
                'config':         config_dict,
            })

        return {'status': 'FAILED', 'H_PT': H_PT, 'beta_over_H': beta_over_H,
                'theta0_initial': theta0, 'with_pt': with_pt,
                'realization': realization}

    except Exception as e:
        import traceback
        return {'status': 'ERROR', 'H_PT': H_PT, 'beta_over_H': beta_over_H,
                'theta0_initial': theta0, 'with_pt': with_pt,
                'realization': realization,
                'error': str(e), 'traceback': traceback.format_exc()}


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Sweep manager
# ---------------------------------------------------------------------------

class SerialParameterSweep:
    """Manage a serial sweep and persist results to disk."""

    def __init__(self, output_dir: str = 'sweep_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'final_energies'), exist_ok=True)

    def run(self, H_PT_values, beta_over_H_values, theta0_values,
            num_realizations):
        """
        Run the full parameter sweep serially.

        Returns
        -------
        all_results : list of successful result dicts
        failed_tasks : list of (H_PT, beta_over_H, theta0, with_pt, realization)
        """
        tasks = []
        for H_PT in H_PT_values:
            for theta0 in theta0_values:
                tasks.append((H_PT, 0.0, theta0, False, 0))

        for H_PT in H_PT_values:
            for beta_over_H in beta_over_H_values:
                for theta0 in theta0_values:
                    for real in range(num_realizations):
                        tasks.append((H_PT, beta_over_H, theta0, True, real))

        n_no_pt = len(H_PT_values) * len(theta0_values)
        n_pt    = len(H_PT_values) * len(beta_over_H_values) * len(theta0_values) * num_realizations

        print("=" * 70)
        print("Serial Parameter Sweep")
        print(f"  H_PT values    : {len(H_PT_values)}")
        print(f"  beta/H values  : {len(beta_over_H_values)}")
        print(f"  theta0 values  : {len(theta0_values)}")
        print(f"  Realizations   : {num_realizations}")
        print(f"  Total sims     : {len(tasks)}  ({n_no_pt} no-PT + {n_pt} with-PT)")
        print("=" * 70)

        all_results  = []
        failed_tasks = []

        for idx, task in enumerate(tasks, 1):
            H_PT, beta_over_H, theta0, with_pt, real = task
            label = f"[{idx}/{len(tasks)}] H_PT={H_PT:.3f} β/H={beta_over_H:.2f} θ₀={theta0:.3f} PT={with_pt} r={real}"
            print(f"\n{label}")

            result = run_single_sim(task)

            if result['status'] == 'SUCCESS':
                all_results.append(result)
                fname = (f'pt_H{H_PT:.3f}_b{beta_over_H:.2f}_t{theta0:.3f}_r{real}.json'
                         if with_pt else
                         f'nopt_H{H_PT:.3f}_t{theta0:.3f}.json')
                path = os.path.join(self.output_dir, 'final_energies', fname)
                with open(path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  OK  total_energy = {result['total']:.6e}")
            else:
                failed_tasks.append(task)
                print(f"  FAIL  {result.get('error', result['status'])}")
                if 'traceback' in result:
                    print(result['traceback'])

        summary = {
            'successful': len(all_results),
            'failed':     len(failed_tasks),
            'failed_tasks': [
                {'H_PT': t[0], 'beta_over_H': t[1], 'theta0': t[2],
                 'with_pt': t[3], 'realization': t[4]}
                for t in failed_tasks
            ],
            'sweep_parameters': {
                'H_PT_values':      [float(x) for x in H_PT_values],
                'beta_over_H_values': [float(x) for x in beta_over_H_values],
                'theta0_values':    [float(x) for x in theta0_values],
                'num_realizations': num_realizations,
            },
        }
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Done.  {len(all_results)}/{len(tasks)} succeeded.")
        print(f"{'='*70}")
        return all_results, failed_tasks

    def make_plots(self, H_PT_values, beta_over_H_values, theta0_values):
        """
        Generate representative energy-evolution plots for a subset of
        (H_PT, theta0) combinations after the main sweep.
        """
        print("\nGenerating representative plots...")

        plot_thetas = (theta0_values if len(theta0_values) <= 3
                       else [theta0_values[0],
                              theta0_values[len(theta0_values) // 2],
                              theta0_values[-1]])

        for H_PT in H_PT_values:
            for theta0 in plot_thetas:
                print(f"  H_PT={H_PT:.3f}  θ₀={theta0:.3f}")

                sim_no_pt, tau_f = create_simulation_pt_at_zero(
                    H_PT=H_PT, beta=0.0,
                    start_time=1.0 / (2 * H_PT), end_time=100.0 / H_PT,
                    Ngrid=16, num_tracers=1_000,
                    enable_nucleation=False, mb=1.0,
                    theta0_initial=theta0,
                    energy_save_interval=10, checkpoint_interval=0,
                )
                res_no_pt = sim_no_pt.run_simulation(tau_f, save_interval=10)

                for beta_over_H in beta_over_H_values:
                    sim_pt, tau_f = create_simulation_pt_at_zero(
                        H_PT=H_PT, beta=beta_over_H * H_PT,
                        start_time=1.0 / (2 * H_PT), end_time=100.0 / H_PT,
                        Ngrid=16, num_tracers=100_000,
                        enable_nucleation=True, mb=0.0,
                        theta0_initial=theta0,
                        energy_save_interval=10, checkpoint_interval=0,
                    )
                    res_pt = sim_pt.run_simulation(tau_f, save_interval=10)
                    self._plot_energy(res_pt, res_no_pt, H_PT, beta_over_H, theta0)

    def _plot_energy(self, res_pt, res_no_pt, H_PT, beta_over_H, theta0):
        """Single energy-evolution comparison plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        def extract(res):
            cfg = res['config']
            eh  = res['energy_history']
            t   = cfg.m0 * np.array([e['t_cosmic'] for e in eh])
            a   = np.array([e['scale_factor'] for e in eh])
            K   = np.array([e['kinetic_energy'] for e in eh])
            G   = np.array([e['gradient_energy'] for e in eh])
            V   = np.array([e['potential_energy'] for e in eh])
            return t, a, K, G, V

        t_pt, a_pt, K_pt, G_pt, V_pt = extract(res_pt)
        t_0,  a_0,  K_0,  G_0,  V_0  = extract(res_no_pt)

        ax.plot(t_pt, G_pt * a_pt**3, color='C1', lw=2,
                label='gradient (with PT)')
        ax.plot(t_pt, (K_pt + V_pt) * a_pt**3, color='C2', lw=2,
                label='kinetic+potential (with PT)')
        ax.plot(t_0,  (K_0 + V_0) * a_0**3,   color='C4', lw=2, ls='--',
                label='kinetic+potential (no PT)')
        ax.axvline(0, color='red', ls=':', lw=1.5, alpha=0.7)

        ax.set_xlabel(r'$m_0\, t$', fontsize=13)
        ax.set_ylabel(r'$\rho\, a^3$', fontsize=13)
        ax.set_title(fr'$H_*={H_PT:.2f}$, $\beta/H={beta_over_H:.1f}$, $\theta_0={theta0:.2f}$',
                     fontsize=13)
        ax.set_xscale('symlog', linthresh=0.1)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
        plt.tight_layout()

        fname = f'H{H_PT:.3f}_b{beta_over_H:.2f}_t{theta0:.3f}.png'
        plt.savefig(os.path.join(self.output_dir, 'plots', fname),
                    dpi=150, bbox_inches='tight')
        plt.close()


# ---------------------------------------------------------------------------
# Entry point — edit parameters here before running
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    H_PT_values = [0.07, 0.7, 1.3, 1.9]

    theta0_values = [
        np.pi / 11,
        np.pi / 7  + np.pi / 12,
        np.pi / 3  + np.pi / 13,
        np.pi / 2.2 + np.pi / 12.9,
        2 * np.pi / 3.5 + np.pi / 11,
        5 * np.pi / 6   + np.pi / 10,
    ]

    beta_over_H_values = [4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40]
    num_realizations   = 5

    sweep = SerialParameterSweep(output_dir='sweep_results')

    all_results, failed = sweep.run(
        H_PT_values, beta_over_H_values, theta0_values, num_realizations)

    sweep.make_plots(H_PT_values, beta_over_H_values, theta0_values)
