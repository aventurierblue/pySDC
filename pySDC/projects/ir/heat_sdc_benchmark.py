import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.ir.scalar_sdc_benchmark import (
    COMPARISON_METHODS,
    METHOD_BAR_COLORS,
    METHOD_DISPLAY_NAMES,
    METHOD_STYLES,
    collect_reachable_arrays,
    create_speedup_memory_ratio_plot,
    expand_num_nodes_range,
    get_linear_ir_sweeper_class,
    get_snapshot_bytes,
    print_speedup_memory_ratio_summary,
)


DEFAULT_INNER_ETA = 1e-3


def make_heat_description(float_precision, dt, num_nodes, restol, maxiter, nvars=31, nu=0.1, freq=2):
    return {
        'problem_class': heatNd_unforced,
        'problem_params': {
            'nvars': nvars,
            'nu': nu,
            'freq': freq,
            'bc': 'dirichlet-zero',
            'float_precision': float_precision,
        },
        'sweeper_class': generic_implicit,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'initial_guess': 'zero',
            'float_precision': float_precision,
        },
        'level_params': {
            'restol': restol,
            'dt': float_precision.type(dt),
            'residual_type': 'full_abs',
        },
        'step_params': {
            'maxiter': maxiter,
        },
    }


def make_heat_ir_description(
    dt,
    num_nodes,
    outer_tol,
    outer_maxiter,
    inner_tol,
    inner_maxiter,
    method='ir-sdc',
    nvars=31,
    nu=0.1,
    freq=2,
    inner_eta=DEFAULT_INNER_ETA,
):
    return {
        'problem_class': heatNd_unforced,
        'problem_params': {
            'nvars': nvars,
            'nu': nu,
            'freq': freq,
            'bc': 'dirichlet-zero',
            'float_precision': np.dtype('float64'),
        },
        'sweeper_class': get_linear_ir_sweeper_class(method),
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'initial_guess': 'zero',
            'float_precision': np.dtype('float64'),
            'inner_float_precision': np.dtype('float32'),
            'inner_maxiter': inner_maxiter,
            'inner_tol': inner_tol,
            'inner_eta': inner_eta,
            'inner_tol_floor': inner_tol,
        },
        'level_params': {
            'restol': outer_tol,
            'dt': np.dtype('float64').type(dt),
            'residual_type': 'full_abs',
        },
        'step_params': {
            'maxiter': outer_maxiter,
        },
    }


def flatten_to_float64(u):
    return np.asarray(u, dtype=np.float64).reshape(-1).copy()


def run_heat_plain_sdc(
    t0,
    t_end,
    steps,
    float_precision,
    num_nodes,
    restol,
    maxiter,
    nvars=31,
    nu=0.1,
    freq=2,
):
    dt = float_precision.type((t_end - t0) / steps)
    description = make_heat_description(
        float_precision=float_precision,
        dt=dt,
        num_nodes=num_nodes,
        restol=restol,
        maxiter=maxiter,
        nvars=nvars,
        nu=nu,
        freq=freq,
    )
    S = Step(description=description)
    L = S.levels[0]
    P = L.prob

    initial_value = flatten_to_float64(P.u_exact(t0))
    endpoint_values = np.empty((steps + 1, initial_value.size), dtype=np.float64)
    exact_endpoint_values = np.empty_like(endpoint_values)
    endpoint_times = np.empty(steps + 1, dtype=np.float64)

    u_current = P.u_exact(t0)
    endpoint_values[0] = flatten_to_float64(u_current)
    exact_endpoint_values[0] = flatten_to_float64(P.u_exact(t0))
    endpoint_times[0] = float(t0)
    total_niter = 0

    start = perf_counter()
    for step_index in range(steps):
        current_time = t0 + step_index * float(dt)
        S.reset_step()
        S.init_step(u_current)
        L.status.time = current_time

        L.sweep.predict()
        L.sweep.compute_residual()

        S.status.iter = 0
        while S.status.iter < S.params.maxiter and L.status.residual > L.params.restol:
            L.sweep.update_nodes()
            L.sweep.compute_residual()
            S.status.iter += 1

        L.sweep.compute_end_point()
        u_current = P.dtype_u(L.uend)
        total_niter += S.status.iter
        endpoint_times[step_index + 1] = current_time + float(dt)
        endpoint_values[step_index + 1] = flatten_to_float64(u_current)
        exact_endpoint_values[step_index + 1] = flatten_to_float64(P.u_exact(current_time + float(dt)))
    elapsed = perf_counter() - start

    return {
        'endpoint_times': endpoint_times,
        'endpoint_values': endpoint_values,
        'exact_endpoint_values': exact_endpoint_values,
        'elapsed': elapsed,
        'avg_step_time': elapsed / steps,
        'total_niter': total_niter,
        'total_outer_iterations': total_niter,
        'total_inner_iterations': 0,
        'restol': restol,
        'dtype': str(float_precision),
        'num_nodes': num_nodes,
        'nvars': nvars,
    }


def run_heat_ir_sdc(
    t0,
    t_end,
    steps,
    num_nodes,
    outer_tol,
    inner_tol,
    outer_maxiter=50,
    inner_maxiter=5,
    method='ir-sdc',
    nvars=31,
    nu=0.1,
    freq=2,
    inner_eta=DEFAULT_INNER_ETA,
):
    dt = np.float64((t_end - t0) / steps)
    description = make_heat_ir_description(
        dt=dt,
        num_nodes=num_nodes,
        outer_tol=outer_tol,
        outer_maxiter=outer_maxiter,
        inner_tol=inner_tol,
        inner_maxiter=inner_maxiter,
        method=method,
        nvars=nvars,
        nu=nu,
        freq=freq,
        inner_eta=inner_eta,
    )
    S = Step(description=description)
    L = S.levels[0]
    P = L.prob

    initial_value = flatten_to_float64(P.u_exact(t0))
    endpoint_values = np.empty((steps + 1, initial_value.size), dtype=np.float64)
    exact_endpoint_values = np.empty_like(endpoint_values)
    endpoint_times = np.empty(steps + 1, dtype=np.float64)

    u_current = P.u_exact(t0)
    endpoint_values[0] = flatten_to_float64(u_current)
    exact_endpoint_values[0] = flatten_to_float64(P.u_exact(t0))
    endpoint_times[0] = float(t0)
    total_outer = 0
    total_inner = 0

    start = perf_counter()
    for step_index in range(steps):
        current_time = t0 + step_index * float(dt)
        S.reset_step()
        S.init_step(u_current)
        L.status.time = current_time

        L.sweep.reset_ir_stats()
        L.sweep.predict()
        L.sweep.compute_residual()

        S.status.iter = 0
        while S.status.iter < S.params.maxiter and L.status.residual > L.params.restol:
            L.sweep.update_nodes()
            L.sweep.compute_residual()
            S.status.iter += 1

        L.sweep.compute_end_point()
        u_current = P.dtype_u(L.uend)
        total_outer += S.status.iter
        total_inner += L.sweep.total_inner_iterations
        endpoint_times[step_index + 1] = current_time + float(dt)
        endpoint_values[step_index + 1] = flatten_to_float64(u_current)
        exact_endpoint_values[step_index + 1] = flatten_to_float64(P.u_exact(current_time + float(dt)))
    elapsed = perf_counter() - start

    return {
        'endpoint_times': endpoint_times,
        'endpoint_values': endpoint_values,
        'exact_endpoint_values': exact_endpoint_values,
        'elapsed': elapsed,
        'avg_step_time': elapsed / steps,
        'total_outer_iterations': total_outer,
        'total_inner_iterations': total_inner,
        'restol': outer_tol,
        'dtype': 'mixed(fp64/fp32)',
        'num_nodes': num_nodes,
        'nvars': nvars,
    }


def initialize_heat_step_state(step, t0=0.0):
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


def make_heat_benchmark_step(method, dt, num_nodes, tol, maxiter=50, inner_maxiter=5, nvars=31, nu=0.1, freq=2):
    if method == 'plain-fp64':
        return Step(
            description=make_heat_description(
                float_precision=np.dtype('float64'),
                dt=np.dtype('float64').type(dt),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        )

    if method == 'plain-fp32':
        return Step(
            description=make_heat_description(
                float_precision=np.dtype('float32'),
                dt=np.dtype('float32').type(dt),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        )

    if method in COMPARISON_METHODS:
        return Step(
            description=make_heat_ir_description(
                dt=np.dtype('float64').type(dt),
                num_nodes=num_nodes,
                outer_tol=tol,
                outer_maxiter=maxiter,
                inner_tol=max(tol, 1e-12),
                inner_maxiter=inner_maxiter,
                method=method,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        )

    raise ValueError(f'Unknown method {method}')


@lru_cache(maxsize=None)
def audit_heat_algorithm_state_memory(method, num_nodes, maxiter=50, inner_maxiter=5, nvars=31, nu=0.1, freq=2):
    step = make_heat_benchmark_step(
        method=method,
        dt=1.0,
        num_nodes=num_nodes,
        tol=1e-9,
        maxiter=maxiter,
        inner_maxiter=inner_maxiter,
        nvars=nvars,
        nu=nu,
        freq=freq,
    )
    initialize_heat_step_state(step)

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


@lru_cache(maxsize=None)
def profile_heat_algorithm_memory(method, num_nodes, maxiter=50, inner_maxiter=5, nvars=31, nu=0.1, freq=2):
    warm_step = make_heat_benchmark_step(
        method=method,
        dt=1.0,
        num_nodes=num_nodes,
        tol=1e-9,
        maxiter=maxiter,
        inner_maxiter=inner_maxiter,
        nvars=nvars,
        nu=nu,
        freq=freq,
    )
    initialize_heat_step_state(warm_step)
    del warm_step

    import gc
    import tracemalloc

    gc.collect()
    tracemalloc.start(25)
    before = tracemalloc.take_snapshot()
    baseline_current = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()

    step = make_heat_benchmark_step(
        method=method,
        dt=1.0,
        num_nodes=num_nodes,
        tol=1e-9,
        maxiter=maxiter,
        inner_maxiter=inner_maxiter,
        nvars=nvars,
        nu=nu,
        freq=freq,
    )
    initialize_heat_step_state(step)
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


def benchmark_heat_configuration(
    method,
    t0,
    t_end,
    steps,
    num_nodes,
    tol,
    repeats=3,
    profile_memory=True,
    maxiter=50,
    inner_maxiter=5,
    nvars=31,
    nu=0.1,
    freq=2,
):
    timings = []
    result = None

    for _ in range(repeats):
        if method == 'plain-fp64':
            result = run_heat_plain_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                float_precision=np.dtype('float64'),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        elif method == 'plain-fp32':
            result = run_heat_plain_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                float_precision=np.dtype('float32'),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        elif method in COMPARISON_METHODS:
            result = run_heat_ir_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                outer_tol=tol,
                inner_tol=max(tol, 1e-12),
                outer_maxiter=maxiter,
                inner_maxiter=inner_maxiter,
                method=method,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        else:
            raise ValueError(f'Unknown method {method}')
        timings.append(result['elapsed'])

    assert result is not None
    result['elapsed'] = min(timings)
    result['avg_step_time'] = result['elapsed'] / steps
    result['endpoint_error'] = float(np.linalg.norm(result['endpoint_values'] - result['exact_endpoint_values'], np.inf))
    result['memory_estimate_bytes'] = None
    if profile_memory:
        result.update(
            audit_heat_algorithm_state_memory(
                method,
                num_nodes,
                maxiter=maxiter,
                inner_maxiter=inner_maxiter,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        )
        result.update(
            profile_heat_algorithm_memory(
                method,
                num_nodes,
                maxiter=maxiter,
                inner_maxiter=inner_maxiter,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
        )
        result['memory_traced_bytes'] = (
            result['memory_numpy_bytes'] if result['memory_numpy_bytes'] is not None else result['memory_state_bytes']
        )
        result['memory_verification_error_bytes'] = None
        result['memory_bytes'] = result['memory_verified_bytes']
    result['method'] = method
    return result


def benchmark_heat_num_nodes_sweep(
    t0,
    t_end,
    steps,
    num_nodes_values,
    tol,
    repeats=3,
    maxiter=50,
    inner_maxiter=5,
    nvars=31,
    nu=0.1,
    freq=2,
    comparison_methods=COMPARISON_METHODS,
):
    sweep_data = []

    for num_nodes in num_nodes_values:
        fp64 = benchmark_heat_configuration(
            method='plain-fp64',
            t0=t0,
            t_end=t_end,
            steps=steps,
            num_nodes=num_nodes,
            tol=tol,
            repeats=repeats,
            profile_memory=True,
            maxiter=maxiter,
            inner_maxiter=inner_maxiter,
            nvars=nvars,
            nu=nu,
            freq=freq,
        )
        comparisons = {}
        for method in comparison_methods:
            result = benchmark_heat_configuration(
                method=method,
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                tol=tol,
                repeats=repeats,
                profile_memory=True,
                maxiter=maxiter,
                inner_maxiter=inner_maxiter,
                nvars=nvars,
                nu=nu,
                freq=freq,
            )
            comparisons[method] = {
                'benchmark': result,
                'speedup': fp64['elapsed'] / result['elapsed'] if result['elapsed'] > 0.0 else np.nan,
                'memory_ratio': result['memory_peak_bytes'] / fp64['memory_peak_bytes'] if fp64['memory_peak_bytes'] > 0 else np.nan,
            }

        sweep_data.append(
            {
                'num_nodes': num_nodes,
                'plain-fp64': fp64,
                'comparisons': comparisons,
            }
        )

    return sweep_data


def create_heat_memory_time_plot(output_path, benchmarks, target_tol):
    labels = [entry['method'] for entry in benchmarks]
    times = [entry['elapsed'] * 1e3 for entry in benchmarks]
    memories = [entry['memory_peak_bytes'] / 1024.0 for entry in benchmarks]
    errors = [entry['endpoint_error'] for entry in benchmarks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    colors = [METHOD_BAR_COLORS.get(entry['method'], 'tab:gray') for entry in benchmarks]

    axes[0].bar(labels, times, color=colors)
    axes[0].set_ylabel('Time to solution [ms]')
    axes[0].set_title('Runtime')

    axes[1].bar(labels, memories, color=colors)
    axes[1].set_ylabel('Peak traced memory [KiB]')
    axes[1].set_title('Memory')

    fig.suptitle(f'Heat SDC comparison at target tolerance {target_tol:g}')
    text_lines = [f"{entry['method']}: error={err:.2e}" for entry, err in zip(benchmarks, errors, strict=True)]
    fig.text(0.5, -0.02, ' | '.join(text_lines), ha='center', va='top', fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_heat_error_runtime_plot(output_path, runtime_data):
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)

    for method, entries in runtime_data.items():
        color, marker = METHOD_STYLES[method]
        times = [entry['elapsed'] * 1e3 for entry in entries]
        errors = [entry['endpoint_error'] for entry in entries]
        labels = [entry['target_tol'] for entry in entries]
        ax.loglog(
            times,
            errors,
            color=color,
            marker=marker,
            linewidth=1.5,
            label=f'Method: {METHOD_DISPLAY_NAMES.get(method, method)}',
        )
        for time_ms, error, tol in zip(times, errors, labels, strict=True):
            ax.annotate(f'tol={tol:.0e}', (time_ms, error), textcoords='offset points', xytext=(4, 4), fontsize=8)

    ax.set_xlabel('Output metric: time to solution [ms]')
    ax.set_ylabel('Output metric: endpoint error at t_end')
    ax.set_title('Heat Tolerance Sweep: input stopping tolerance vs. output runtime and endpoint error')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(
    t0=0.0,
    t_end=0.1,
    steps=25,
    num_nodes=3,
    num_nodes_range=(3, 9),
    target_tol=1e-9,
    maxiter=50,
    inner_maxiter=5,
    nvars=31,
    nu=0.1,
    freq=2,
):
    #methods = ['plain-fp64', 'plain-fp32', 'ir-sdc', 'sdc-ir']
    methods = ['plain-fp64', 'ir-sdc', 'sdc-ir']
    tolerance_sweep = [1e-8, 1e-9, 1e-10, 1e-11]
    num_nodes_values = expand_num_nodes_range(num_nodes_range)

    benchmarks = [
        benchmark_heat_configuration(
            method=method,
            t0=t0,
            t_end=t_end,
            steps=steps,
            num_nodes=num_nodes,
            tol=target_tol,
            repeats=3,
            profile_memory=True,
            maxiter=maxiter,
            inner_maxiter=inner_maxiter,
            nvars=nvars,
            nu=nu,
            freq=freq,
        )
        for method in methods
    ]

    runtime_data = {
        method: [
            {
                **benchmark_heat_configuration(
                    method=method,
                    t0=t0,
                    t_end=t_end,
                    steps=steps,
                    num_nodes=num_nodes,
                    tol=tol,
                    repeats=2,
                    profile_memory=False,
                    maxiter=maxiter,
                    inner_maxiter=inner_maxiter,
                    nvars=nvars,
                    nu=nu,
                    freq=freq,
                ),
                'target_tol': tol,
            }
            for tol in tolerance_sweep
        ]
        for method in methods
    }
    num_nodes_sweep = benchmark_heat_num_nodes_sweep(
        t0=t0,
        t_end=t_end,
        steps=steps,
        num_nodes_values=num_nodes_values,
        tol=target_tol,
        repeats=3,
        maxiter=maxiter,
        inner_maxiter=inner_maxiter,
        nvars=nvars,
        nu=nu,
        freq=freq,
        comparison_methods=COMPARISON_METHODS,
    )

    images_dir = Path(__file__).resolve().parents[3] / 'images'
    create_heat_memory_time_plot(images_dir / 'heat_sdc_memory_time_comparison_pysdc.png', benchmarks, target_tol)
    create_heat_error_runtime_plot(images_dir / 'heat_sdc_error_vs_runtime_pysdc.png', runtime_data)
    create_speedup_memory_ratio_plot(
        images_dir / 'heat_sdc_ir_speedup_memory_vs_num_nodes_pysdc.png',
        num_nodes_sweep,
        f'Heat IR variants vs. num_nodes at target tolerance {target_tol:g}',
    )

    print('Heat SDC comparison in pySDC')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={num_nodes}, '
        f'num_nodes_range={num_nodes_values}, '
        f'target_tol={target_tol}, maxiter={maxiter}, inner_maxiter={inner_maxiter}, '
        f'nvars={nvars}, nu={nu}, freq={freq}'
    )
    for entry in benchmarks:
        print(entry['method'])
        print(f"  time [ms]      = {entry['elapsed'] * 1e3:.3f}")
        print(f"  outer iters    = {entry['total_outer_iterations']}")
        print(f"  inner iters    = {entry['total_inner_iterations']}")
        print(f"  memory [KiB]   = {entry['memory_bytes'] / 1024.0:.3f} (verified retained arrays)")
        print(
            f"  split [KiB]    = {entry['memory_verified_float64_bytes'] / 1024.0:.3f} fp64 + "
            f"{entry['memory_verified_float32_bytes'] / 1024.0:.3f} fp32"
        )
        print(f"  traced [KiB]   = {entry['memory_traced_bytes'] / 1024.0:.3f} (tracemalloc cross-check)")
        print(f"  total [KiB]    = {entry['memory_state_bytes'] / 1024.0:.3f} (all traced retained state)")
        print(f"  peak [KiB]     = {entry['memory_peak_bytes'] / 1024.0:.3f} (during warm step)")
        print(f"  endpoint error = {entry['endpoint_error']:.3e}")
    print_speedup_memory_ratio_summary('Heat num-node sweep: plain-fp64 vs linear IR variants', num_nodes_sweep)


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark heat-equation plain SDC and IR-SDC variants inside pySDC.')
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=5, help='Final time')
    parser.add_argument('--steps', type=int, default=100, help='Number of timesteps')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=5, help='Number of collocation nodes')
    parser.add_argument(
        '--num-nodes-range',
        dest='num_nodes_range',
        nargs=2,
        type=int,
        metavar=('START', 'STOP'),
        default=(3, 9),
        help='Inclusive num-node sweep range for the speedup/memory plot',
    )
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-11, help='Target stopping tolerance')
    parser.add_argument('--maxiter', '--max-iter', dest='maxiter', type=int, default=50, help='Maximum outer sweeps')
    parser.add_argument(
        '--inner-maxiter',
        '--inner-max-iter',
        dest='inner_maxiter',
        type=int,
        default=15,
        help='Maximum inner IR sweeps',
    )
    parser.add_argument('--nvars', type=int, default=31, help='Number of spatial degrees of freedom per dimension')
    parser.add_argument('--nu', type=float, default=0.1, help='Diffusion coefficient')
    parser.add_argument('--freq', type=int, default=2, help='Frequency of the exact solution')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        t0=args.t0,
        t_end=args.t_end,
        steps=args.steps,
        num_nodes=args.num_nodes,
        num_nodes_range=tuple(args.num_nodes_range),
        target_tol=args.target_tol,
        maxiter=args.maxiter,
        inner_maxiter=args.inner_maxiter,
        nvars=args.nvars,
        nu=args.nu,
        freq=args.freq,
    )
