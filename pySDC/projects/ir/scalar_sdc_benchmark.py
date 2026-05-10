import argparse
from collections import defaultdict
import gc
import tracemalloc
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.TestEquation_0D import real_scalar_testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.ir.sweepers import generic_implicit_ir


def get_supported_precisions():
    return {
        'float32': np.dtype('float32'),
        'float64': np.dtype('float64'),
    }


def make_description(float_precision, dt, num_nodes, restol, maxiter):
    level_params = {
        'restol': restol,
        'dt': float_precision.type(dt),
        'residual_type': 'last_abs',
    }

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': num_nodes,
        'QI': 'LU',
        'initial_guess': 'zero',
        'float_precision': float_precision,
    }

    problem_params = {
        'lam': float_precision.type(-20.0),
        'u0': float_precision.type(1.0),
        'float_precision': float_precision,
    }

    step_params = {
        'maxiter': maxiter,
    }

    description = {
        'problem_class': real_scalar_testequation0d,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }
    return description


def make_ir_description(
    dt,
    num_nodes,
    outer_tol,
    outer_maxiter,
    inner_tol,
    inner_maxiter,
    use_scalar_fast_path=True,
):
    level_params = {
        'restol': outer_tol,
        'dt': np.dtype('float64').type(dt),
        'residual_type': 'last_abs',
    }

    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': num_nodes,
        'QI': 'LU',
        'initial_guess': 'zero',
        'float_precision': np.dtype('float64'),
        'inner_float_precision': np.dtype('float32'),
        'inner_maxiter': inner_maxiter,
        'inner_tol': inner_tol,
        'use_scalar_fast_path': use_scalar_fast_path,
    }

    problem_params = {
        'lam': np.float64(-20.0),
        'u0': np.float64(1.0),
        'float_precision': np.dtype('float64'),
    }

    step_params = {
        'maxiter': outer_maxiter,
    }

    return {
        'problem_class': real_scalar_testequation0d,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit_ir,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }


def run_plain_sdc(t0, t_end, steps, float_precision, num_nodes, restol, maxiter):
    dt = float_precision.type((t_end - t0) / steps)
    description = make_description(
        float_precision=float_precision,
        dt=dt,
        num_nodes=num_nodes,
        restol=restol,
        maxiter=maxiter,
    )
    S = Step(description=description)
    L = S.levels[0]
    P = L.prob

    endpoint_values = np.empty(steps + 1, dtype=np.float64)
    exact_endpoint_values = np.empty(steps + 1, dtype=np.float64)
    endpoint_times = np.empty(steps + 1, dtype=np.float64)

    u_current = P.u_exact(t0)
    endpoint_values[0] = float(u_current[0])
    exact_endpoint_values[0] = float(P.u_exact(t0)[0])
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
        endpoint_values[step_index + 1] = float(u_current[0])
        exact_endpoint_values[step_index + 1] = float(P.u_exact(current_time + float(dt))[0])
    elapsed = perf_counter() - start

    return {
        'endpoint_times': endpoint_times,
        'endpoint_values': endpoint_values,
        'exact_endpoint_values': exact_endpoint_values,
        'elapsed': elapsed,
        'avg_step_time': elapsed / steps,
        'total_niter': total_niter,
        'restol': restol,
        'dtype': str(float_precision),
        'num_nodes': num_nodes,
    }


def run_ir_sdc(
    t0,
    t_end,
    steps,
    num_nodes,
    outer_tol,
    inner_tol,
    outer_maxiter=50,
    inner_maxiter=5,
):
    dt = np.float64((t_end - t0) / steps)
    description = make_ir_description(
        dt=dt,
        num_nodes=num_nodes,
        outer_tol=outer_tol,
        outer_maxiter=outer_maxiter,
        inner_tol=inner_tol,
        inner_maxiter=inner_maxiter,
        use_scalar_fast_path=False,
    )
    S = Step(description=description)
    L = S.levels[0]
    P = L.prob

    endpoint_values = np.empty(steps + 1, dtype=np.float64)
    exact_endpoint_values = np.empty(steps + 1, dtype=np.float64)
    endpoint_times = np.empty(steps + 1, dtype=np.float64)

    u_current = P.u_exact(t0)
    endpoint_values[0] = float(u_current[0])
    exact_endpoint_values[0] = float(P.u_exact(t0)[0])
    endpoint_times[0] = t0
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
        endpoint_values[step_index + 1] = float(u_current[0])
        exact_endpoint_values[step_index + 1] = float(P.u_exact(current_time + float(dt))[0])
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
    }


def estimate_algorithm_state_bytes(method, num_nodes):
    # Count the retained NumPy arrays reachable from one warmed Step. This still
    # excludes Python object overhead and temporary work arrays.
    m = num_nodes

    if method == 'plain-fp64':
        return 8 * (5 * m * m + 15 * m + 5)
    if method == 'plain-fp32':
        return 8 * (2 * m * m + 4 * m) + 4 * (3 * (m + 1) ** 2 + 5 * m + 2)
    if method == 'ir-sdc':
        return 8 * (5 * m * m + 14 * m + 5)
    raise ValueError(f'Unknown method {method}')


def make_benchmark_step(method, dt, num_nodes, tol, maxiter=50, inner_maxiter=5):
    if method == 'plain-fp64':
        return Step(
            description=make_description(
                float_precision=np.dtype('float64'),
                dt=np.dtype('float64').type(dt),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
            )
        )

    if method == 'plain-fp32':
        return Step(
            description=make_description(
                float_precision=np.dtype('float32'),
                dt=np.dtype('float32').type(dt),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
            )
        )

    if method == 'ir-sdc':
        return Step(
            description=make_ir_description(
                dt=np.dtype('float64').type(dt),
                num_nodes=num_nodes,
                outer_tol=tol,
                outer_maxiter=maxiter,
                inner_tol=max(tol, 1e-12),
                inner_maxiter=inner_maxiter,
                use_scalar_fast_path=False,
            )
        )

    raise ValueError(f'Unknown method {method}')


def initialize_step_state(step, t0=0.0):
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


def collect_reachable_arrays(root):
    arrays = {}
    seen_objects = set()
    stack = [root]

    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen_objects:
            continue
        seen_objects.add(obj_id)

        if isinstance(obj, np.ndarray):
            arrays[obj_id] = obj
            continue

        if obj is None or isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, np.generic, np.dtype)):
            continue

        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue

        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
            continue

        if isinstance(obj, type):
            continue

        if hasattr(obj, '__dict__'):
            stack.extend(vars(obj).values())

        slots = getattr(type(obj), '__slots__', ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if hasattr(obj, name):
                try:
                    stack.append(getattr(obj, name))
                except Exception:
                    continue

    return tuple(arrays.values())


@lru_cache(maxsize=None)
def audit_algorithm_state_memory(method, num_nodes, maxiter=50, inner_maxiter=5):
    step = make_benchmark_step(
        method=method,
        dt=1.0,
        num_nodes=num_nodes,
        tol=1e-9,
        maxiter=maxiter,
        inner_maxiter=inner_maxiter,
    )
    initialize_step_state(step)

    arrays = collect_reachable_arrays(step)
    bytes_by_dtype = defaultdict(int)
    total_bytes = 0
    for array in arrays:
        nbytes = int(array.nbytes)
        total_bytes += nbytes
        bytes_by_dtype[str(array.dtype)] += nbytes

    return {
        'memory_verified_bytes': total_bytes,
        'memory_verified_array_count': len(arrays),
        'memory_verified_float64_bytes': bytes_by_dtype.get('float64', 0),
        'memory_verified_float32_bytes': bytes_by_dtype.get('float32', 0),
    }


def get_snapshot_bytes(snapshot, domain=None):
    if domain is not None:
        snapshot = snapshot.filter_traces([tracemalloc.DomainFilter(True, domain)])
    return sum(stat.size for stat in snapshot.statistics('filename'))


@lru_cache(maxsize=None)
def profile_algorithm_memory(method, num_nodes, maxiter=50, inner_maxiter=5):
    # The scalar benchmark memory depends on the method structure and node count,
    # not on the stopping tolerance. Warm up once so one-time library allocations
    # do not pollute the per-method tracemalloc profile.
    warm_step = make_benchmark_step(
        method=method,
        dt=1.0,
        num_nodes=num_nodes,
        tol=1e-9,
        maxiter=maxiter,
        inner_maxiter=inner_maxiter,
    )
    initialize_step_state(warm_step)
    del warm_step
    gc.collect()

    tracemalloc.start(25)
    before = tracemalloc.take_snapshot()
    baseline_current = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()

    step = make_benchmark_step(
        method=method,
        dt=1.0,
        num_nodes=num_nodes,
        tol=1e-9,
        maxiter=maxiter,
        inner_maxiter=inner_maxiter,
    )
    initialize_step_state(step)
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
            get_snapshot_bytes(after, np.lib.tracemalloc_domain)
            - get_snapshot_bytes(before, np.lib.tracemalloc_domain),
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
    repeats=3,
    profile_memory=True,
    maxiter=50,
    inner_maxiter=5,
):
    timings = []
    result = None
    for _ in range(repeats):
        if method == 'plain-fp64':
            result = run_plain_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                float_precision=np.dtype('float64'),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
            )
        elif method == 'plain-fp32':
            result = run_plain_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                float_precision=np.dtype('float32'),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=maxiter,
            )
        elif method == 'ir-sdc':
            result = run_ir_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                outer_tol=tol,
                inner_tol=max(tol, 1e-12),
                outer_maxiter=maxiter,
                inner_maxiter=inner_maxiter,
            )
        else:
            raise ValueError(f'Unknown method {method}')
        timings.append(result['elapsed'])

    assert result is not None
    result['elapsed'] = min(timings)
    result['avg_step_time'] = result['elapsed'] / steps
    result['endpoint_error'] = float(np.linalg.norm(result['endpoint_values'] - result['exact_endpoint_values'], np.inf))
    result['memory_estimate_bytes'] = estimate_algorithm_state_bytes(method, num_nodes)
    if profile_memory:
        result.update(audit_algorithm_state_memory(method, num_nodes, maxiter=maxiter, inner_maxiter=inner_maxiter))
        result.update(profile_algorithm_memory(method, num_nodes, maxiter=maxiter, inner_maxiter=inner_maxiter))
        result['memory_traced_bytes'] = (
            result['memory_numpy_bytes']
            if result['memory_numpy_bytes'] is not None
            else result['memory_state_bytes']
        )
        result['memory_verification_error_bytes'] = result['memory_verified_bytes'] - result['memory_estimate_bytes']
        result['memory_bytes'] = result['memory_verified_bytes']
    result['method'] = method
    return result


def create_memory_time_plot(output_path, benchmarks, target_tol):
    labels = [entry['method'] for entry in benchmarks]
    times = [entry['elapsed'] * 1e3 for entry in benchmarks]
    memories = [entry['memory_bytes'] / 1024.0 for entry in benchmarks]
    errors = [entry['endpoint_error'] for entry in benchmarks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    axes[0].bar(labels, times, color=colors)
    axes[0].set_ylabel('Time to solution [ms]')
    axes[0].set_title('Runtime')

    axes[1].bar(labels, memories, color=colors)
    axes[1].set_ylabel('Verified retained NumPy state [KiB]')
    axes[1].set_title('Memory')

    fig.suptitle(f'Scalar SDC comparison at target tolerance {target_tol:g}')
    text_lines = [f"{entry['method']}: error={err:.2e}" for entry, err in zip(benchmarks, errors, strict=True)]
    fig.text(0.5, -0.02, ' | '.join(text_lines), ha='center', va='top', fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_error_runtime_plot(output_path, runtime_data):
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    styles = {
        'plain-fp64': ('tab:blue', 'o'),
        'plain-fp32': ('tab:orange', 's'),
        'ir-sdc': ('tab:green', '^'),
    }

    for method, entries in runtime_data.items():
        color, marker = styles[method]
        times = [entry['elapsed'] * 1e3 for entry in entries]
        errors = [entry['endpoint_error'] for entry in entries]
        labels = [entry['target_tol'] for entry in entries]
        ax.loglog(times, errors, color=color, marker=marker, linewidth=1.5, label=f'Method: {method}')
        for time_ms, error, tol in zip(times, errors, labels, strict=True):
            ax.annotate(f'tol={tol:.0e}', (time_ms, error), textcoords='offset points', xytext=(4, 4), fontsize=8)

    ax.axhline(1e-10, color='black', linestyle='--', linewidth=1.0, label='Output target: endpoint error = 1e-10')
    ax.set_xlabel('Output metric: time to solution [ms]')
    ax.set_ylabel('Output metric: endpoint error at t_end')
    ax.set_title('Tolerance Sweep: input stopping tolerance vs. output runtime and endpoint error')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(t0=0.0, t_end=5.0, steps=5000, num_nodes=7, target_tol=1e-9, maxiter=50, inner_maxiter=5):
    methods = ['plain-fp64', 'plain-fp32', 'ir-sdc']
    tolerance_sweep = [1e-9, 1e-10, 1e-11, 1e-12]

    benchmarks = [
        benchmark_configuration(
            method=method,
            t0=t0,
            t_end=t_end,
            steps=steps,
            num_nodes=num_nodes,
            tol=target_tol,
            repeats=5,
            profile_memory=True,
            maxiter=maxiter,
            inner_maxiter=inner_maxiter,
        )
        for method in methods
    ]

    runtime_data = {
        method: [
            {
                **benchmark_configuration(
                    method,
                    t0,
                    t_end,
                    steps,
                    num_nodes,
                    tol,
                    repeats=3,
                    profile_memory=False,
                    maxiter=maxiter,
                    inner_maxiter=inner_maxiter,
                ),
                'target_tol': tol,
            }
            for tol in tolerance_sweep
        ]
        for method in methods
    }

    images_dir = Path(__file__).resolve().parents[3] / 'images'
    create_memory_time_plot(images_dir / 'scalar_sdc_memory_time_comparison_pysdc.png', benchmarks, target_tol)
    create_error_runtime_plot(images_dir / 'scalar_sdc_error_vs_runtime_pysdc.png', runtime_data)

    print('Scalar SDC comparison in pySDC')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={num_nodes}, '
        f'target_tol={target_tol}, maxiter={maxiter}, inner_maxiter={inner_maxiter}'
    )
    for entry in benchmarks:
        print(entry['method'])
        print(f"  time [ms]      = {entry['elapsed'] * 1e3:.3f}")
        print(f"  memory [KiB]   = {entry['memory_bytes'] / 1024.0:.3f} (verified retained arrays)")
        print(
            f"  split [KiB]    = {entry['memory_verified_float64_bytes'] / 1024.0:.3f} fp64 + "
            f"{entry['memory_verified_float32_bytes'] / 1024.0:.3f} fp32"
        )
        print(f"  traced [KiB]   = {entry['memory_traced_bytes'] / 1024.0:.3f} (tracemalloc cross-check)")
        print(f"  total [KiB]    = {entry['memory_state_bytes'] / 1024.0:.3f} (all traced retained state)")
        print(f"  peak [KiB]     = {entry['memory_peak_bytes'] / 1024.0:.3f} (during warm step)")
        print(f"  estimate [KiB] = {entry['memory_estimate_bytes'] / 1024.0:.3f} (hand formula)")
        print(f"  delta [B]      = {entry['memory_verification_error_bytes']} (verified - estimate)")
        print(f"  endpoint error = {entry['endpoint_error']:.3e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark scalar plain SDC and IR-SDC variants inside pySDC.')
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=5.0, help='Final time')
    parser.add_argument('--steps', type=int, default=500, help='Number of timesteps')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=3, help='Number of collocation nodes')
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-9, help='Target stopping tolerance')
    parser.add_argument('--maxiter', '--max-iter', dest='maxiter', type=int, default=50, help='Maximum outer sweeps')
    parser.add_argument(
        '--inner-maxiter',
        '--inner-max-iter',
        dest='inner_maxiter',
        type=int,
        default=5,
        help='Maximum inner IR sweeps',
    )
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
        inner_maxiter=args.inner_maxiter,
    )
