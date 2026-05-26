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
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.ir.scalar_sdc_benchmark import collect_reachable_arrays, get_snapshot_bytes
from pySDC.projects.ir.sweepers import generic_implicit_newton_sdc, generic_implicit_newton_sdc_ir


METHODS = ('newton-sdc-fp64', 'newton-sdc-ir')
DEFAULT_INNER_ETA = 1e-3


def make_plain_description(dt, num_nodes, restol, maxiter, nvars, nu, lambda0):
    return {
        'problem_class': generalized_fisher,
        'problem_params': {
            'nvars': nvars,
            'nu': nu,
            'lambda0': lambda0,
            'newton_maxiter': 60,
            'newton_tol': 1e-9,
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
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
):
    return {
        'problem_class': generalized_fisher,
        'problem_params': {
            'nvars': nvars,
            'nu': nu,
            'lambda0': lambda0,
            'newton_maxiter': 60,
            'newton_tol': 1e-9,
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
            'inner_eta': DEFAULT_INNER_ETA,
            'inner_tol_floor': None if inner_solver == 'sdc' else 5e-5,
            'inner_maxiter': 1000,
            'gmres_maxiter': 100,
            'gmres_restart': 100,
            'inner_tol': 1e-7,
            'preconditioner_solver': 'splu',
            'gmres_warm_start': True,
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


def make_newton_sdc_ir_description(
    dt,
    num_nodes,
    outer_tol,
    outer_maxiter,
    nvars,
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
):
    return make_gmres_ir_description(
        dt=dt,
        num_nodes=num_nodes,
        outer_tol=outer_tol,
        outer_maxiter=outer_maxiter,
        nvars=nvars,
        nu=nu,
        lambda0=lambda0,
        inner_solver=inner_solver,
        inner_qi=inner_qi,
        cache_inner_step=cache_inner_step,
    )


def make_newton_sdc_fp64_description(
    dt,
    num_nodes,
    outer_tol,
    outer_maxiter,
    nvars,
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
):
    description = make_newton_sdc_ir_description(
        dt=dt,
        num_nodes=num_nodes,
        outer_tol=outer_tol,
        outer_maxiter=outer_maxiter,
        nvars=nvars,
        nu=nu,
        lambda0=lambda0,
        inner_solver=inner_solver,
        inner_qi=inner_qi,
        cache_inner_step=cache_inner_step,
    )
    description['sweeper_class'] = generic_implicit_newton_sdc
    description['sweeper_params'] = dict(description['sweeper_params'])
    description['sweeper_params']['inner_float_precision'] = np.dtype('float64')
    return description


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

    # Keep reference solves out of the timed region.
    for step_index in range(1, steps + 1):
        step_time = endpoint_times[step_index]
        endpoint_errors[step_index] = np.linalg.norm(
            endpoint_solutions[step_index] - np.array(problem.u_exact(step_time), dtype=np.float64),
            np.inf,
        )

    exact = np.array(problem.u_exact(t_end), dtype=np.float64)
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
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
    repeats=2,
):
    dt = (t_end - t0) / steps
    best = None
    memory_description = None
    for _ in range(repeats):
        if method == 'plain-fp64':
            description = make_plain_description(
                dt=dt,
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                lambda0=lambda0,
            )
        elif method == 'newton-sdc-fp64':
            description = make_newton_sdc_fp64_description(
                dt=dt,
                num_nodes=num_nodes,
                outer_tol=tol,
                outer_maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                lambda0=lambda0,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
            )
        elif method in ('gmres-ir', 'newton-sdc-ir'):
            description = make_newton_sdc_ir_description(
                dt=dt,
                num_nodes=num_nodes,
                outer_tol=tol,
                outer_maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                lambda0=lambda0,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
            )
        else:
            raise ValueError(f'Unknown method {method}')

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
        'newton-sdc-fp64': {'color': 'tab:blue', 'marker': 'o'},
        'newton-sdc-ir': {'color': 'tab:pink', 'marker': 'P'},
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
    ax.set_title(f'Fisher Newton-SDC fp64 vs fp64/32 on [{t0:g}, {t_end:.6g}]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_memory_time_plot(output_path, benchmarks, target_tol):
    labels = [entry['method'] for entry in benchmarks]
    times = [entry['elapsed'] * 1e3 for entry in benchmarks]
    memories = [entry['memory_peak_bytes'] / 1024.0 for entry in benchmarks]
    errors = [entry['endpoint_error'] for entry in benchmarks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    colors = ['tab:blue', 'tab:pink']

    axes[0].bar(labels, times, color=colors)
    axes[0].set_ylabel('Time to solution [ms]')
    axes[0].set_title('Runtime')

    axes[1].bar(labels, memories, color=colors)
    axes[1].set_ylabel('Peak traced memory [KiB]')
    axes[1].set_title('Memory')

    fig.suptitle(f'Fisher comparison at target tolerance {target_tol:g}')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_inner_solver_runtime_plot(output_path, solver_benchmarks, target_tol):
    labels = [entry['inner_solver'] for entry in solver_benchmarks]
    fp64_times = [entry['newton-sdc-fp64']['elapsed'] * 1e3 for entry in solver_benchmarks]
    ir_times = [entry['newton-sdc-ir']['elapsed'] * 1e3 for entry in solver_benchmarks]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    ax.bar(x - width / 2, fp64_times, width, label='newton-sdc-fp64', color='tab:blue')
    ax.bar(x + width / 2, ir_times, width, label='newton-sdc-ir', color='tab:pink')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time to solution [ms]')
    ax.set_title(f'Fisher runtime by inner solver at target tolerance {target_tol:g}')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='best')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def import_seaborn():
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError('This benchmark requires `seaborn` to create the heatmap.') from exc

    return sns


def benchmark_heatmap_method(
    method,
    t0,
    t_end,
    steps,
    num_nodes,
    target_tol,
    maxiter,
    nvars,
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
    repeats=2,
):
    dt = (t_end - t0) / steps
    best = None

    for _ in range(repeats):
        if method == 'newton-sdc-fp64':
            description = make_newton_sdc_fp64_description(
                dt=dt,
                num_nodes=num_nodes,
                outer_tol=target_tol,
                outer_maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                lambda0=lambda0,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
            )
        elif method == 'newton-sdc-ir':
            description = make_newton_sdc_ir_description(
                dt=dt,
                num_nodes=num_nodes,
                outer_tol=target_tol,
                outer_maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                lambda0=lambda0,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
            )
        else:
            raise ValueError(f'Unknown method {method}')

        result = _run_method(description, t0, t_end, steps, method)

        if best is None:
            best = result
            continue

        if result['converged'] and not best['converged']:
            best = result
            continue

        if result['converged'] == best['converged'] and result['elapsed'] < best['elapsed']:
            best = result

    return best


def benchmark_speedup_grid(
    t0,
    t_end,
    steps,
    num_nodes_values,
    nvars_values,
    target_tol,
    maxiter,
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
    repeats=2,
):
    speedups = np.full((len(nvars_values), len(num_nodes_values)), np.nan, dtype=np.float64)

    print('Fisher heatmap benchmark: Newton-SDC fp64 vs Newton-SDC IR fp64/32')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={list(num_nodes_values)}, '
        f'nvars={list(nvars_values)}, target_tol={target_tol}, maxiter={maxiter}, '
        f'nu={nu}, lambda0={lambda0}, inner_solver={inner_solver}, inner_qi={inner_qi}, repeats={repeats}'
    )
    print('nvars    num_nodes   fp64_time[s]    ir_time[s]      speedup   status')

    for row, nvars in enumerate(nvars_values):
        for col, num_nodes in enumerate(num_nodes_values):
            fp64 = benchmark_heatmap_method(
                method='newton-sdc-fp64',
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                target_tol=target_tol,
                maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                lambda0=lambda0,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
                repeats=repeats,
            )
            ir = benchmark_heatmap_method(
                method='newton-sdc-ir',
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                target_tol=target_tol,
                maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                lambda0=lambda0,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
                repeats=repeats,
            )

            converged = fp64['converged'] and ir['converged']
            speedup = fp64['elapsed'] / ir['elapsed'] if converged and ir['elapsed'] > 0.0 else np.nan
            speedups[row, col] = speedup

            speedup_text = f'{speedup:.3f}x' if np.isfinite(speedup) else 'n/a'
            print(
                f'{nvars:>5d}   {num_nodes:>9d}   '
                f'{fp64["elapsed"]:>12.6e}   {ir["elapsed"]:>10.6e}   '
                f'{speedup_text:>9}   '
                f'{"ok" if converged else "stalled"}'
            )

    return speedups


def create_speedup_heatmap(output_path, speedups, num_nodes_values, nvars_values):
    sns = import_seaborn()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style='white')
    fig_width = max(6.4, 1.1 * len(num_nodes_values) + 2.8)
    fig_height = max(4.8, 0.7 * len(nvars_values) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    finite = speedups[np.isfinite(speedups)]
    if finite.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'No converged benchmark configurations', ha='center', va='center', fontsize=12)
    else:
        annot = np.full(speedups.shape, '', dtype=object)
        for row, col in np.ndindex(speedups.shape):
            if np.isfinite(speedups[row, col]):
                annot[row, col] = f'{speedups[row, col]:.2f}'

        vmin = min(float(finite.min()), 1.0)
        vmax = max(float(finite.max()), 1.0)
        if np.isclose(vmin, vmax):
            delta = max(0.05, 0.05 * abs(vmax) if vmax else 0.05)
            vmin -= delta
            vmax += delta

        sns.heatmap(
            speedups,
            mask=~np.isfinite(speedups),
            annot=annot,
            fmt='',
            cmap='vlag',
            center=1.0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor='white',
            xticklabels=num_nodes_values,
            yticklabels=nvars_values,
            cbar_kws={'label': 'Speedup = fp64 time / IR time [x]'},
            ax=ax,
        )

        ax.set_xlabel('num_nodes')
        ax.set_ylabel('FD discretization nvars')
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)

    ax.set_title('Fisher Newton-SDC IR runtime speedup')

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_results(benchmarks, t0, t_end, steps, num_nodes, target_tol, maxiter, nvars, inner_solver, inner_qi):
    print('Fisher benchmark: Newton-SDC fp64 vs Newton-SDC IR fp64/32')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={num_nodes}, '
        f'nvars={nvars}, target_tol={target_tol}, maxiter={maxiter}, inner_solver={inner_solver}, inner_qi={inner_qi}'
    )
    print('method              avg_step_time[s]   total_time[s]      end_error       residual        outer_it   inner_it   mem[KiB]   peak[KiB]   status')
    for entry in benchmarks:
        print(
            f"{entry['method']:<18}"
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

    fp64 = next(entry for entry in benchmarks if entry['method'] == 'newton-sdc-fp64')
    ir = next(entry for entry in benchmarks if entry['method'] == 'newton-sdc-ir')
    speedup = fp64['elapsed'] / ir['elapsed'] if ir['elapsed'] > 0.0 else np.inf
    print(f'newton-sdc-fp64 endpoint error = {fp64["endpoint_error"]:.6e}')
    print(f'newton-sdc-ir endpoint error   = {ir["endpoint_error"]:.6e}')
    print(f'newton-sdc-ir speedup vs fp64  = {speedup:.3f}x')
    print(
        f'newton-sdc-fp64 memory    = {fp64["memory_bytes"] / 1024.0:.3f} KiB retained, '
        f'{fp64["memory_peak_bytes"] / 1024.0:.3f} KiB peak traced'
    )
    print(
        f'newton-sdc-ir memory      = {ir["memory_bytes"] / 1024.0:.3f} KiB retained, '
        f'{ir["memory_peak_bytes"] / 1024.0:.3f} KiB peak traced'
    )


def main(
    t0=0.0,
    t_end=1.0,
    steps=32,
    num_nodes=3,
    target_tol=1e-8,
    maxiter=20,
    nvars=2047,
    nu=1.0,
    lambda0=2.0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
    repeats=2,
    compare_inner_solvers=(),
    heatmap=False,
    heatmap_num_nodes_values=(3, 4, 5, 6, 7, 8),
    heatmap_nvars_values=(63, 127, 255, 511, 1023, 2047, 4095),
    heatmap_output_path=None,
):
    images_dir = Path(__file__).resolve().parents[3] / 'images'

    if heatmap:
        num_nodes_values = tuple(dict.fromkeys(heatmap_num_nodes_values))
        nvars_values = tuple(dict.fromkeys(heatmap_nvars_values))
        speedups = benchmark_speedup_grid(
            t0=t0,
            t_end=t_end,
            steps=steps,
            num_nodes_values=num_nodes_values,
            nvars_values=nvars_values,
            target_tol=target_tol,
            maxiter=maxiter,
            nu=nu,
            lambda0=lambda0,
            inner_solver=inner_solver,
            inner_qi=inner_qi,
            cache_inner_step=cache_inner_step,
            repeats=repeats,
        )
        if heatmap_output_path is None:
            heatmap_output_path = images_dir / 'fisher_newton_sdc_ir_speedup_heatmap.png'
        create_speedup_heatmap(heatmap_output_path, speedups, num_nodes_values, nvars_values)
        print(f'Wrote {heatmap_output_path}')
        return

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
            nu=nu,
            lambda0=lambda0,
            inner_solver=inner_solver,
            inner_qi=inner_qi,
            cache_inner_step=cache_inner_step,
            repeats=repeats,
        )
        for method in METHODS
    ]
    solver_benchmarks = None
    if compare_inner_solvers:
        solver_benchmarks = []
        for solver in compare_inner_solvers:
            solver_benchmarks.append(
                {
                    'inner_solver': solver,
                    'newton-sdc-fp64': benchmark_configuration(
                        method='newton-sdc-fp64',
                        t0=t0,
                        t_end=t_end,
                        steps=steps,
                        num_nodes=num_nodes,
                        tol=target_tol,
                        maxiter=maxiter,
                        nvars=nvars,
                        nu=nu,
                        lambda0=lambda0,
                        inner_solver=solver,
                        inner_qi=inner_qi,
                        cache_inner_step=cache_inner_step,
                        repeats=repeats,
                    ),
                    'newton-sdc-ir': benchmark_configuration(
                        method='newton-sdc-ir',
                        t0=t0,
                        t_end=t_end,
                        steps=steps,
                        num_nodes=num_nodes,
                        tol=target_tol,
                        maxiter=maxiter,
                        nvars=nvars,
                        nu=nu,
                        lambda0=lambda0,
                        inner_solver=solver,
                        inner_qi=inner_qi,
                        cache_inner_step=cache_inner_step,
                        repeats=repeats,
                    ),
                }
            )

    memory_output_path = images_dir / 'fisher_fp64_vs_gmres_ir_memory_time.png'
    # output_path = images_dir / 'fisher_fp64_vs_gmres_ir_time_error.png'
    # create_time_error_plot(output_path, benchmarks, t0, t_end)
    create_memory_time_plot(memory_output_path, benchmarks, target_tol)
    if compare_inner_solvers:
        solver_runtime_output_path = images_dir / 'fisher_inner_solver_runtime_comparison.png'
        create_inner_solver_runtime_plot(solver_runtime_output_path, solver_benchmarks, target_tol)
    print_results(benchmarks, t0, t_end, steps, num_nodes, target_tol, maxiter, nvars, inner_solver, inner_qi)
    print(f'Wrote {memory_output_path}')
    if compare_inner_solvers:
        print(f'Wrote {solver_runtime_output_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare full-fp64 Newton-SDC against mixed-precision Newton-SDC IR on generalized Fisher.'
    )
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=5.0, help='Final time')
    parser.add_argument('--steps', type=int, default=200, help='Number of time steps')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=5, help='Number of collocation nodes')
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-8, help='Target stopping tolerance')
    parser.add_argument('--maxiter', type=int, default=20, help='Maximum outer iterations per step')
    parser.add_argument('--nvars', type=int, default=2047, help='Spatial resolution')
    parser.add_argument('--repeats', type=int, default=2, help='Timing repeats per configuration')
    parser.add_argument('--nu', type=float, default=1.0, help='Fisher nonlinearity parameter')
    parser.add_argument('--lambda0', type=float, default=2.0, help='Fisher lambda0 parameter')
    parser.add_argument('--inner-solver', dest='inner_solver', choices=('direct', 'gmres', 'lgmres', 'fgmres', 'sdc'), default='sdc', help='Inner Newton-SDC solver')
    parser.add_argument('--inner-qi', dest='inner_qi', default='LU', help='Inner Newton-SDC preconditioner')
    parser.add_argument('--cache-inner-step', dest='cache_inner_step', action='store_true', help='Reuse inner Newton-SDC work buffers')
    parser.add_argument('--compare-inner-solvers', dest='compare_inner_solvers', nargs='+', choices=('direct', 'sdc', 'gmres', 'lgmres', 'fgmres'), default=(), help='Inner solvers to benchmark additionally and include in the runtime comparison plot')
    parser.add_argument('--heatmap', action='store_true', help='Run the speedup heatmap path over nvars and num_nodes')
    parser.add_argument(
        '--heatmap-num-nodes',
        dest='heatmap_num_nodes_values',
        nargs='+',
        type=int,
        default=(3, 4, 5, 6, 7, 8),
        help='Collocation node counts for the heatmap x-axis',
    )
    parser.add_argument(
        '--heatmap-nvars',
        dest='heatmap_nvars_values',
        nargs='+',
        type=int,
        default=(63, 127, 255, 511, 1023, 2047, 4095),
        help='Fisher FD resolutions for the heatmap y-axis',
    )
    parser.add_argument('--heatmap-output', dest='heatmap_output_path', type=Path, default=None, help='Output path for the heatmap image')
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
        nvars=args.nvars,
        repeats=args.repeats,
        nu=args.nu,
        lambda0=args.lambda0,
        inner_solver=args.inner_solver,
        inner_qi=args.inner_qi,
        cache_inner_step=args.cache_inner_step,
        compare_inner_solvers=args.compare_inner_solvers,
        heatmap=args.heatmap,
        heatmap_num_nodes_values=args.heatmap_num_nodes_values,
        heatmap_nvars_values=args.heatmap_nvars_values,
        heatmap_output_path=args.heatmap_output_path,
    )
