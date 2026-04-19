import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.TestEquation_0D import real_scalar_testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.Resilience.sweepers import generic_implicit_efficient
from pySDC.projects.ir.sweepers import generic_implicit_ir


def get_supported_precisions():
    return {
        'float32': np.dtype('float32'),
        'float64': np.dtype('float64'),
    }


def make_description(float_precision, dt, num_nodes, restol, maxiter, efficient=True):
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
        'sweeper_class': generic_implicit_efficient if efficient else generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }
    return description


def make_ir_description(dt, num_nodes, outer_tol, outer_maxiter, inner_tol, inner_maxiter):
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


def run_plain_sdc(t0, t_end, steps, float_precision, num_nodes, restol, maxiter, efficient=True):
    dt = float_precision.type((t_end - t0) / steps)
    description = make_description(
        float_precision=float_precision,
        dt=dt,
        num_nodes=num_nodes,
        restol=restol,
        maxiter=maxiter,
        efficient=efficient,
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


def run_ir_sdc(t0, t_end, steps, num_nodes, outer_tol, inner_tol, outer_maxiter=50, inner_maxiter=5):
    dt = np.float64((t_end - t0) / steps)
    description = make_ir_description(
        dt=dt,
        num_nodes=num_nodes,
        outer_tol=outer_tol,
        outer_maxiter=outer_maxiter,
        inner_tol=inner_tol,
        inner_maxiter=inner_maxiter,
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
    f64 = np.dtype('float64').itemsize
    f32 = np.dtype('float32').itemsize
    vec = num_nodes
    mat = num_nodes * num_nodes

    if method == 'plain-fp64':
        return (2 * mat + 4 * vec) * f64
    if method == 'plain-fp32':
        return (2 * mat + 4 * vec) * f32
    if method == 'ir-sdc':
        return (5 * mat + 7 * vec) * f64 + (2 * mat + 5 * vec) * f32
    raise ValueError(f'Unknown method {method}')


def benchmark_configuration(method, t0, t_end, steps, num_nodes, tol, repeats=3):
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
                maxiter=50,
                efficient=True,
            )
        elif method == 'plain-fp32':
            result = run_plain_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                float_precision=np.dtype('float32'),
                num_nodes=num_nodes,
                restol=tol,
                maxiter=50,
                efficient=True,
            )
        elif method == 'ir-sdc':
            result = run_ir_sdc(
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                outer_tol=tol,
                inner_tol=max(tol, 1e-12),
                outer_maxiter=50,
                inner_maxiter=5,
            )
        else:
            raise ValueError(f'Unknown method {method}')
        timings.append(result['elapsed'])

    assert result is not None
    result['elapsed'] = min(timings)
    result['avg_step_time'] = result['elapsed'] / steps
    result['endpoint_error'] = float(np.linalg.norm(result['endpoint_values'] - result['exact_endpoint_values'], np.inf))
    result['memory_bytes'] = estimate_algorithm_state_bytes(method, num_nodes)
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
    axes[1].set_ylabel('Algorithm state memory [KiB]')
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


def main(t0=0.0, t_end=5.0, steps=5000, num_nodes=7, target_tol=1e-9):
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
        )
        for method in methods
    ]

    runtime_data = {
        method: [
            {**benchmark_configuration(method, t0, t_end, steps, num_nodes, tol, repeats=3), 'target_tol': tol}
            for tol in tolerance_sweep
        ]
        for method in methods
    }

    images_dir = Path(__file__).resolve().parents[3] / 'images'
    create_memory_time_plot(images_dir / 'scalar_sdc_memory_time_comparison_pysdc.png', benchmarks, target_tol)
    create_error_runtime_plot(images_dir / 'scalar_sdc_error_vs_runtime_pysdc.png', runtime_data)

    print('Scalar SDC comparison in pySDC')
    print(f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={num_nodes}, target_tol={target_tol}')
    for entry in benchmarks:
        print(entry['method'])
        print(f"  time [ms]      = {entry['elapsed'] * 1e3:.3f}")
        print(f"  memory [KiB]   = {entry['memory_bytes'] / 1024.0:.3f}")
        print(f"  endpoint error = {entry['endpoint_error']:.3e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark scalar plain SDC and IR-SDC variants inside pySDC.')
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=5.0, help='Final time')
    parser.add_argument('--steps', type=int, default=5000, help='Number of timesteps')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=7, help='Number of collocation nodes')
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-9, help='Target stopping tolerance')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(t0=args.t0, t_end=args.t_end, steps=args.steps, num_nodes=args.num_nodes, target_tol=args.target_tol)
