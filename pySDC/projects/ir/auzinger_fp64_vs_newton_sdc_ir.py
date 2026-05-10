import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.ir.sweepers import generic_implicit_newton_sdc_ir


METHODS = ('plain-fp64', 'gmres-ir')


def make_plain_description(dt, num_nodes, restol, maxiter):
    return {
        'problem_class': auzinger,
        'problem_params': {
            'newton_maxiter': 20,
            'newton_tol': 1e-14,
            'float_precision': np.dtype('float64'),
        },
        'sweeper_class': generic_implicit,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'initial_guess': 'spread',
            'float_precision': np.dtype('float64'),
        },
        'level_params': {
            'restol': restol,
            'dt': np.float64(dt),
            'residual_type': 'full_abs',
        },
        'step_params': {
            'maxiter': maxiter,
        },
    }


def make_gmres_ir_description(dt, num_nodes, outer_tol, outer_maxiter):
    return {
        'problem_class': auzinger,
        'problem_params': {
            'newton_maxiter': 20,
            'newton_tol': 1e-14,
            'float_precision': np.dtype('float64'),
        },
        'sweeper_class': generic_implicit_newton_sdc_ir,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'initial_guess': 'spread',
            'float_precision': np.dtype('float64'),
            'inner_float_precision': np.dtype('float32'),
            'inner_solver': 'sdc',
            'adaptive_inner': True,
            'inner_eta_scale': 0.01,
            'inner_eta_power': 1.5,
            'inner_tol_floor': 1e-7,
            'inner_tol_ceiling': 1e-2,
            'inner_maxiter': 20,
            'gmres_maxiter': 20,
            'gmres_restart': None,
            'inner_tol': 1e-8,
        },
        'level_params': {
            'restol': outer_tol,
            'dt': np.float64(dt),
            'residual_type': 'full_abs',
        },
        'step_params': {
            'maxiter': outer_maxiter,
        },
    }


def _run_method(description, t0, t_end, steps, method):
    step = Step(description=description)
    level = step.levels[0]
    problem = level.prob
    dt = float(level.dt)

    u_current = problem.u_exact(t0)
    total_outer = 0
    total_inner = 0
    final_residual = None
    converged = True
    endpoint_times = np.empty(steps + 1, dtype=np.float64)
    endpoint_errors = np.empty(steps + 1, dtype=np.float64)

    endpoint_times[0] = float(t0)
    endpoint_errors[0] = 0.0

    start = perf_counter()
    for step_index in range(steps):
        current_time = t0 + step_index * dt

        step.reset_step()
        step.init_step(u_current)
        level.status.time = current_time

        if hasattr(level.sweep, 'reset_ir_stats'):
            level.sweep.reset_ir_stats()

        level.sweep.predict()
        level.sweep.compute_residual()

        step.status.iter = 0
        while step.status.iter < step.params.maxiter and level.status.residual > level.params.restol:
            level.sweep.update_nodes()
            level.sweep.compute_residual()
            step.status.iter += 1

        if level.status.residual > level.params.restol:
            converged = False

        level.sweep.compute_end_point()
        u_current = problem.dtype_u(level.uend)
        total_outer += step.status.iter
        final_residual = float(level.status.residual)

        if hasattr(level.sweep, 'total_inner_iterations'):
            total_inner += level.sweep.total_inner_iterations
            level.sweep.total_inner_iterations = 0

        step_time = current_time + dt
        endpoint_times[step_index + 1] = step_time
        endpoint_errors[step_index + 1] = np.linalg.norm(
            np.array(u_current, dtype=np.float64) - np.array(problem.u_exact(step_time), dtype=np.float64),
            np.inf,
        )

    elapsed = perf_counter() - start
    exact = np.array(problem.u_exact(t_end), dtype=np.float64)
    uend = np.array(u_current, dtype=np.float64)
    err = np.linalg.norm(uend - exact, np.inf)

    return {
        'method': method,
        'elapsed': elapsed,
        'avg_step_time': elapsed / steps,
        'endpoint_error': float(err),
        'final_residual': final_residual,
        'total_outer_iterations': total_outer,
        'total_inner_iterations': total_inner,
        'converged': converged,
        'endpoint_times': endpoint_times,
        'endpoint_errors': endpoint_errors,
    }


def benchmark_configuration(method, t0, t_end, steps, num_nodes, tol, maxiter, repeats=2):
    dt = (t_end - t0) / steps
    best = None
    for _ in range(repeats):
        description = (
            make_plain_description(dt=dt, num_nodes=num_nodes, restol=tol, maxiter=maxiter)
            if method == 'plain-fp64'
            else make_gmres_ir_description(dt=dt, num_nodes=num_nodes, outer_tol=tol, outer_maxiter=maxiter)
        )
        result = _run_method(description, t0, t_end, steps, method)
        if best is None or result['elapsed'] < best['elapsed']:
            best = result
    return best


def create_time_error_plot(output_path, benchmarks, t0, t_end):
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    styles = {
        'plain-fp64': {'color': 'tab:blue', 'marker': 'o'},
        'gmres-ir': {'color': 'tab:pink', 'marker': 'P'},
    }

    for entry in benchmarks:
        ax.semilogy(
            entry['endpoint_times'][1:],
            np.maximum(entry['endpoint_errors'][1:], np.finfo(np.float64).tiny),
            linewidth=1.4,
            label=entry['method'],
            **styles[entry['method']],
        )

    ax.set_xlabel('Time')
    ax.set_ylabel('Endpoint error per step')
    ax.set_title(f'Auzinger fp64 vs gmres-ir on [{t0:g}, {t_end:.6g}]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_results(benchmarks, t0, t_end, steps, num_nodes, target_tol, maxiter):
    print('Auzinger benchmark: fp64 generic_implicit vs Newton-SDC gmres-ir')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={num_nodes}, '
        f'target_tol={target_tol}, maxiter={maxiter}'
    )
    print('method        avg_step_time[s]   total_time[s]      end_error       residual        outer_it   inner_it   status')
    for entry in benchmarks:
        print(
            f"{entry['method']:<12}"
            f"{entry['avg_step_time']:>16.6e}   "
            f"{entry['elapsed']:>13.6e}   "
            f"{entry['endpoint_error']:>12.6e}   "
            f"{entry['final_residual']:>12.6e}   "
            f"{entry['total_outer_iterations']:>8d}   "
            f"{entry['total_inner_iterations']:>8d}   "
            f"{'ok' if entry['converged'] else 'stalled'}"
        )

    plain = next(entry for entry in benchmarks if entry['method'] == 'plain-fp64')
    gmres = next(entry for entry in benchmarks if entry['method'] == 'gmres-ir')
    speedup = plain['elapsed'] / gmres['elapsed'] if gmres['elapsed'] > 0.0 else np.inf
    print(f'plain-fp64 endpoint error = {plain["endpoint_error"]:.6e}')
    print(f'gmres-ir endpoint error   = {gmres["endpoint_error"]:.6e}')
    print(f'gmres-ir speedup vs plain = {speedup:.3f}x')


def main(t0=0.0, t_end=4.0 * np.pi, steps=200, num_nodes=3, target_tol=1e-10, maxiter=50):
    benchmarks = [
        benchmark_configuration(
            method=method,
            t0=t0,
            t_end=t_end,
            steps=steps,
            num_nodes=num_nodes,
            tol=target_tol,
            maxiter=maxiter,
            repeats=2,
        )
        for method in METHODS
    ]

    images_dir = Path(__file__).resolve().parents[3] / 'images'
    output_path = images_dir / 'auzinger_fp64_vs_gmres_ir_time_error.png'
    create_time_error_plot(output_path, benchmarks, t0, t_end)
    print_results(benchmarks, t0, t_end, steps, num_nodes, target_tol, maxiter)
    print(f'Wrote {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Compare fp64 generic_implicit against default gmres-ir on Auzinger.')
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=4.0 * np.pi, help='Final time')
    parser.add_argument('--steps', type=int, default=200, help='Number of time steps')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=3, help='Number of collocation nodes')
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-10, help='Target stopping tolerance')
    parser.add_argument('--maxiter', type=int, default=50, help='Maximum outer iterations per step')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(t0=args.t0, t_end=args.t_end, steps=args.steps, num_nodes=args.num_nodes, target_tol=args.target_tol, maxiter=args.maxiter)
