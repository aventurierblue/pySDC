import argparse
import gc
from pathlib import Path
from time import perf_counter
import tracemalloc

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.ir.scalar_sdc_benchmark import collect_reachable_arrays, get_snapshot_bytes
from pySDC.projects.ir.sweepers import generic_implicit_newton_sdc_ir


METHODS = ('plain-fp64', 'sdc-ir')


def make_plain_description(dt, num_nodes, restol, maxiter, nvars, eps, radius):
    return {
        'problem_class': allencahn_fullyimplicit,
        'problem_params': {
            'nvars': nvars,
            'nu': 2,
            'eps': eps,
            'newton_maxiter': 100,
            'newton_tol': 1e-9,
            'lin_tol': 1e-9,
            'lin_maxiter': 200,
            'radius': radius,
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


def make_gmres_ir_description(
    dt,
    num_nodes,
    outer_tol,
    outer_maxiter,
    nvars,
    eps,
    radius,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
):
    return {
        'problem_class': allencahn_fullyimplicit,
        'problem_params': {
            'nvars': nvars,
            'nu': 2,
            'eps': eps,
            'newton_maxiter': 100,
            'newton_tol': 1e-9,
            'lin_tol': 1e-6,
            'lin_maxiter': 200,
            'radius': radius,
            'float_precision': np.dtype('float64'),
        },
        'sweeper_class': generic_implicit_newton_sdc_ir,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'inner_QI': inner_qi,
            'initial_guess': 'spread',
            'float_precision': np.dtype('float64'),
            'inner_float_precision': np.dtype('float32'),
            'inner_solver': inner_solver,
            'adaptive_inner': True,
            'inner_eta_scale': 0.01,
            'inner_eta_power': 1.5,
            'inner_tol_floor': 1e-5,
            'inner_tol_ceiling': 1e-2,
            'inner_maxiter': 20,
            'gmres_maxiter': 20,
            'gmres_restart': 20,
            'inner_tol': 1e-8,
            'cache_inner_step': cache_inner_step,
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


def make_sdc_ir_description(dt, num_nodes, outer_tol, outer_maxiter, nvars, eps, radius, inner_qi='LU', cache_inner_step=False):
    description = make_gmres_ir_description(
        dt=dt,
        num_nodes=num_nodes,
        outer_tol=outer_tol,
        outer_maxiter=outer_maxiter,
        nvars=nvars,
        eps=eps,
        radius=radius,
        inner_solver='sdc',
        inner_qi=inner_qi,
        cache_inner_step=cache_inner_step,
    )
    description['sweeper_params'].update(
        {
            'adaptive_inner': False,
            'inner_tol_floor': None,
            'inner_tol_ceiling': None,
            'inner_tol': 1e-8,
            'inner_maxiter': 20,
        }
    )
    return description


def _run_method(description, t0, t_end, steps, method):
    step = Step(description=description)
    level = step.levels[0]
    problem = level.prob
    dt = float(level.dt)

    u_current = problem.u_exact(t0)
    reference_start = problem.u_exact(t0)
    reference_cache = {float(t0): np.array(reference_start, dtype=np.float64)}
    total_outer = 0
    total_inner = 0
    final_residual = None
    converged = True
    endpoint_times = np.empty(steps + 1, dtype=np.float64)
    endpoint_errors = np.empty(steps + 1, dtype=np.float64)
    endpoint_solutions = [np.array(u_current, dtype=np.float64)]

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
        endpoint_solutions.append(np.array(u_current, dtype=np.float64))

    elapsed = perf_counter() - start

    # Keep expensive reference solves out of the timed region.
    for step_index in range(1, steps + 1):
        step_time = endpoint_times[step_index]
        if step_time not in reference_cache:
            reference_cache[step_time] = np.array(problem.u_exact(step_time, u_init=reference_start, t_init=t0), dtype=np.float64)
        endpoint_errors[step_index] = np.linalg.norm(endpoint_solutions[step_index] - reference_cache[step_time], np.inf)

    if float(t_end) not in reference_cache:
        reference_cache[float(t_end)] = np.array(problem.u_exact(t_end, u_init=reference_start, t_init=t0), dtype=np.float64)
    exact = reference_cache[float(t_end)]
    uend = endpoint_solutions[-1]
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


def _initialize_step_state(step, t0=0.0):
    level = step.levels[0]
    problem = level.prob

    step.reset_step()
    step.init_step(problem.u_exact(t0))
    level.status.time = t0
    if hasattr(level.sweep, 'reset_ir_stats'):
        level.sweep.reset_ir_stats()
    level.sweep.predict()
    level.sweep.compute_residual()
    level.sweep.update_nodes()
    level.sweep.compute_end_point()

    return step


def audit_algorithm_state_memory(description):
    step = _initialize_step_state(Step(description=description))
    arrays = collect_reachable_arrays(step)

    bytes_by_dtype = {}
    total_bytes = 0
    for array in arrays:
        nbytes = int(array.nbytes)
        total_bytes += nbytes
        bytes_by_dtype[str(array.dtype)] = bytes_by_dtype.get(str(array.dtype), 0) + nbytes

    return {
        'memory_verified_bytes': total_bytes,
        'memory_verified_array_count': len(arrays),
        'memory_verified_float64_bytes': bytes_by_dtype.get('float64', 0),
        'memory_verified_float32_bytes': bytes_by_dtype.get('float32', 0),
    }


def profile_algorithm_memory(description):
    warm_step = _initialize_step_state(Step(description=description))
    del warm_step
    gc.collect()

    tracemalloc.start(25)
    before = tracemalloc.take_snapshot()
    baseline_current = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()

    step = _initialize_step_state(Step(description=description))
    gc.collect()

    after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        'memory_state_bytes': max(0, get_snapshot_bytes(after) - get_snapshot_bytes(before)),
        'memory_peak_bytes': max(0, peak - baseline_current),
        'memory_current_bytes': max(0, current - baseline_current),
    }
    if hasattr(np.lib, 'tracemalloc_domain'):
        result['memory_numpy_bytes'] = max(
            0,
            get_snapshot_bytes(after, np.lib.tracemalloc_domain) - get_snapshot_bytes(before, np.lib.tracemalloc_domain),
        )
    else:
        result['memory_numpy_bytes'] = None

    return result


def benchmark_configuration(
    method,
    t0,
    t_end,
    steps,
    num_nodes,
    tol,
    maxiter,
    nvars,
    eps,
    radius,
    repeats=2,
):
    dt = (t_end - t0) / steps
    best = None
    memory_description = None
    for _ in range(repeats):
        description = (
            make_plain_description(dt=dt, num_nodes=num_nodes, restol=tol, maxiter=maxiter, nvars=nvars, eps=eps, radius=radius)
            if method == 'plain-fp64'
            else make_sdc_ir_description(
                dt=dt,
                num_nodes=num_nodes,
                outer_tol=tol,
                outer_maxiter=maxiter,
                nvars=nvars,
                eps=eps,
                radius=radius,
            )
        )
        memory_description = description
        result = _run_method(description, t0, t_end, steps, method)
        if best is None or result['elapsed'] < best['elapsed']:
            best = result

    best.update(audit_algorithm_state_memory(memory_description))
    best.update(profile_algorithm_memory(memory_description))
    best['memory_traced_bytes'] = best['memory_numpy_bytes'] if best['memory_numpy_bytes'] is not None else best['memory_state_bytes']
    best['memory_bytes'] = best['memory_verified_bytes']
    return best


def create_time_error_plot(output_path, benchmarks, t0, t_end):
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    styles = {
        'plain-fp64': {'color': 'tab:blue', 'marker': 'o'},
        'sdc-ir': {'color': 'tab:pink', 'marker': 'P'},
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
    ax.set_title(f'Allen-Cahn fp64 vs sdc-ir on [{t0:g}, {t_end:.6g}]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_memory_time_plot(output_path, benchmarks, target_tol):
    labels = [entry['method'] for entry in benchmarks]
    times = [entry['elapsed'] * 1e3 for entry in benchmarks]
    memories = [entry['memory_bytes'] / 1024.0 for entry in benchmarks]
    errors = [entry['endpoint_error'] for entry in benchmarks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    colors = ['tab:blue', 'tab:pink']

    axes[0].bar(labels, times, color=colors)
    axes[0].set_ylabel('Time to solution [ms]')
    axes[0].set_title('Runtime')

    axes[1].bar(labels, memories, color=colors)
    axes[1].set_ylabel('Verified retained NumPy state [KiB]')
    axes[1].set_title('Memory')

    fig.suptitle(f'Allen-Cahn comparison at target tolerance {target_tol:g}')
    text_lines = [f"{entry['method']}: error={err:.2e}" for entry, err in zip(benchmarks, errors, strict=True)]
    fig.text(0.5, -0.02, ' | '.join(text_lines), ha='center', va='top', fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_results(benchmarks, t0, t_end, steps, num_nodes, target_tol, maxiter, nvars):
    print('Allen-Cahn benchmark: fp64 generic_implicit vs Newton-SDC sdc-ir')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={num_nodes}, '
        f'nvars={nvars}, target_tol={target_tol}, maxiter={maxiter}'
    )
    print('method        avg_step_time[s]   total_time[s]      end_error       residual        outer_it   inner_it   mem[KiB]   peak[KiB]   status')
    for entry in benchmarks:
        print(
            f"{entry['method']:<12}"
            f"{entry['avg_step_time']:>16.6e}   "
            f"{entry['elapsed']:>13.6e}   "
            f"{entry['endpoint_error']:>12.6e}   "
            f"{entry['final_residual']:>12.6e}   "
            f"{entry['total_outer_iterations']:>8d}   "
            f"{entry['total_inner_iterations']:>8d}   "
            f"{entry['memory_bytes'] / 1024.0:>8.1f}   "
            f"{entry['memory_peak_bytes'] / 1024.0:>9.1f}   "
            f"{'ok' if entry['converged'] else 'stalled'}"
        )

    plain = next(entry for entry in benchmarks if entry['method'] == 'plain-fp64')
    sdc = next(entry for entry in benchmarks if entry['method'] == 'sdc-ir')
    speedup = plain['elapsed'] / sdc['elapsed'] if sdc['elapsed'] > 0.0 else np.inf
    print(f'plain-fp64 endpoint error = {plain["endpoint_error"]:.6e}')
    print(f'sdc-ir endpoint error     = {sdc["endpoint_error"]:.6e}')
    print(f'sdc-ir speedup vs plain   = {speedup:.3f}x')
    print(
        f'plain-fp64 memory         = {plain["memory_bytes"] / 1024.0:.3f} KiB retained, '
        f'{plain["memory_peak_bytes"] / 1024.0:.3f} KiB peak traced'
    )
    print(
        f'sdc-ir memory             = {sdc["memory_bytes"] / 1024.0:.3f} KiB retained, '
        f'{sdc["memory_peak_bytes"] / 1024.0:.3f} KiB peak traced'
    )


def main(
    t0=0.0,
    t_end=0.032,
    steps=32,
    num_nodes=3,
    target_tol=1e-8,
    maxiter=50,
    nvars=(64, 64),
    eps=0.04,
    radius=0.25,
):
    benchmarks = [
        benchmark_configuration(
            method=method,
            t0=t0,
            t_end=t_end,
            steps=steps,
            num_nodes=num_nodes,
            tol=target_tol,
            maxiter=maxiter,
            nvars=nvars,
            eps=eps,
            radius=radius,
            repeats=2,
        )
        for method in METHODS
    ]

    images_dir = Path(__file__).resolve().parents[3] / 'images'
    output_path = images_dir / 'allencahn_fp64_vs_sdc_ir_time_error.png'
    memory_output_path = images_dir / 'allencahn_fp64_vs_sdc_ir_memory_time.png'
    create_time_error_plot(output_path, benchmarks, t0, t_end)
    create_memory_time_plot(memory_output_path, benchmarks, target_tol)
    print_results(benchmarks, t0, t_end, steps, num_nodes, target_tol, maxiter, nvars)
    print(f'Wrote {output_path}')
    print(f'Wrote {memory_output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Compare fp64 generic_implicit against mixed-precision Newton-SDC IR on 2D Allen-Cahn.')
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=0.005, help='Final time')
    parser.add_argument('--steps', type=int, default=5, help='Number of time steps')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=3, help='Number of collocation nodes')
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-8, help='Target stopping tolerance')
    parser.add_argument('--maxiter', type=int, default=50, help='Maximum outer iterations per step')
    parser.add_argument('--nvars', nargs=2, type=int, default=(64, 64), help='Spatial resolution')
    parser.add_argument('--eps', type=float, default=0.04, help='Allen-Cahn epsilon')
    parser.add_argument('--radius', type=float, default=0.25, help='Initial circle radius')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        t0=args.t0,
        t_end=args.t_end,
        steps=args.steps,
        num_nodes=args.num_nodes,
        target_tol=args.target_tol,
        maxiter=args.maxiter,
        nvars=tuple(args.nvars),
        eps=args.eps,
        radius=args.radius,
    )
