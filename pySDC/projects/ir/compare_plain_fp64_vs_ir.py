import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pySDC.projects.ir.scalar_sdc_benchmark import run_ir_sdc, run_plain_sdc


DEFAULT_STEP_COUNTS = (1, 2, 3, 6, 10, 18, 32)


def build_reference_solution(t0, t_end, reference_steps, num_nodes, tol, maxiter):
    reference = run_plain_sdc(
        t0=t0,
        t_end=t_end,
        steps=reference_steps,
        float_precision=np.dtype('float64'),
        num_nodes=num_nodes,
        restol=tol,
        maxiter=maxiter,
        efficient=True,
    )
    endpoint_value = float(reference['endpoint_values'][-1])
    exact_endpoint_value = float(reference['exact_endpoint_values'][-1])

    return {
        'steps': reference_steps,
        'dt': (t_end - t0) / reference_steps,
        'endpoint_value': endpoint_value,
        'exact_endpoint_value': exact_endpoint_value,
        'endpoint_error': abs(endpoint_value - exact_endpoint_value),
        'result': reference,
    }


def run_convergence_study(
    t0,
    t_end,
    step_counts,
    reference_steps,
    num_nodes,
    tol,
    maxiter,
    inner_maxiter,
    reference_tol,
    reference_maxiter,
):
    reference = build_reference_solution(
        t0=t0,
        t_end=t_end,
        reference_steps=reference_steps,
        num_nodes=num_nodes,
        tol=reference_tol,
        maxiter=reference_maxiter,
    )

    results = []
    for steps in sorted(step_counts):
        plain = run_plain_sdc(
            t0=t0,
            t_end=t_end,
            steps=steps,
            float_precision=np.dtype('float64'),
            num_nodes=num_nodes,
            restol=tol,
            maxiter=maxiter,
            efficient=True,
        )
        ir = run_ir_sdc(
            t0=t0,
            t_end=t_end,
            steps=steps,
            num_nodes=num_nodes,
            outer_tol=tol,
            inner_tol=max(tol, 1e-12),
            outer_maxiter=maxiter,
            inner_maxiter=inner_maxiter,
        )

        plain_endpoint = float(plain['endpoint_values'][-1])
        ir_endpoint = float(ir['endpoint_values'][-1])
        exact_endpoint = reference['exact_endpoint_value']

        results.append(
            {
                'steps': steps,
                'dt': (t_end - t0) / steps,
                'plain_endpoint': plain_endpoint,
                'ir_endpoint': ir_endpoint,
                'plain_error_vs_reference': abs(plain_endpoint - reference['endpoint_value']),
                'ir_error_vs_reference': abs(ir_endpoint - reference['endpoint_value']),
                'plain_error_vs_exact': abs(plain_endpoint - exact_endpoint),
                'ir_error_vs_exact': abs(ir_endpoint - exact_endpoint),
                'method_difference': abs(plain_endpoint - ir_endpoint),
                'maxiter': maxiter,
                'inner_maxiter': inner_maxiter,
            }
        )

    return reference, results


def get_plot_floor(*sequences):
    positive_values = []
    for sequence in sequences:
        positive_values.extend(float(value) for value in sequence if float(value) > 0.0)
    if positive_values:
        return min(positive_values) / 10.0
    return np.finfo(np.float64).tiny


def create_convergence_plot(output_path, reference, results, t0, t_end, num_nodes, tol):
    dts = np.array([entry['dt'] for entry in results], dtype=np.float64)
    plain_errors = np.array([entry['plain_error_vs_reference'] for entry in results], dtype=np.float64)
    ir_errors = np.array([entry['ir_error_vs_reference'] for entry in results], dtype=np.float64)
    reference_floor = reference['endpoint_error']

    plot_floor = get_plot_floor(plain_errors, ir_errors, [reference_floor])
    plot_plain_errors = np.maximum(plain_errors, plot_floor)
    plot_ir_errors = np.maximum(ir_errors, plot_floor)
    plot_reference_floor = max(reference_floor, plot_floor)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    ax.loglog(dts, plot_plain_errors, color='tab:blue', marker='o', linewidth=1.6, label='plain fp64 endpoint error vs. fine fp64 reference')
    ax.loglog(dts, plot_ir_errors, color='tab:green', marker='x', linewidth=1.6, label='IR-SDC endpoint error vs. fine fp64 reference')
    ax.axhline(
        plot_reference_floor,
        color='black',
        linestyle='--',
        linewidth=1.0,
        label='fine fp64 reference vs. exact',
    )

    expected_order = 2 * num_nodes - 1
    guide_dts = np.array([np.min(dts), np.max(dts)], dtype=np.float64)
    guide_anchor_dt = dts[0]
    guide_anchor_error = plot_ir_errors[0]
    guide_errors = guide_anchor_error * (guide_dts / guide_anchor_dt) ** expected_order
    ax.loglog(guide_dts, guide_errors, color='0.45', linestyle=':', linewidth=1.2, label=f'O(dt^{expected_order}) guide')

    for dt, error, steps in zip(dts, plot_ir_errors, [entry['steps'] for entry in results], strict=True):
        ax.annotate(f'N={steps}', (dt, error), textcoords='offset points', xytext=(4, 4), fontsize=8)

    ax.set_xlabel('Timestep size dt')
    ax.set_ylabel(f'Endpoint error at t={t_end:g}')
    ax.set_title(f'IR-SDC endpoint convergence on [{t0:g}, {t_end:g}]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    fig.text(
        0.5,
        -0.02,
        (
            f'reference: plain fp64 with {reference["steps"]} steps '
            f'(dt={reference["dt"]:.3e}), num_nodes={num_nodes}, tol={tol:g}, '
            f'maxiter={results[0]["maxiter"] if results else "?"}, '
            f'inner_maxiter={results[0]["inner_maxiter"] if results else "?"}'
        ),
        ha='center',
        va='top',
        fontsize=9,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot IR-SDC endpoint convergence against a fine plain fp64 reference.'
    )
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=1.0, help='Final time for the fixed interval')
    parser.add_argument(
        '--step-counts',
        nargs='+',
        type=int,
        default=DEFAULT_STEP_COUNTS,
        help='Timestep counts used for the convergence study',
    )
    parser.add_argument(
        '--reference-steps',
        dest='reference_steps',
        type=int,
        default=500000,
        help='Timestep count for the fine plain fp64 reference solution',
    )
    parser.add_argument(
        '--reference-dt',
        dest='reference_dt',
        type=float,
        default=None,
        help='Reference timestep size for the fine plain fp64 solution. Overrides --reference-steps.',
    )
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=7, help='Number of collocation nodes')
    parser.add_argument('--tol', type=float, default=1e-12, help='Stopping tolerance for both methods')
    parser.add_argument('--maxiter', '--max-iter', dest='maxiter', type=int, default=50, help='Maximum outer sweeps')
    parser.add_argument(
        '--inner-maxiter',
        '--inner-max-iter',
        dest='inner_maxiter',
        type=int,
        default=5,
        help='Maximum inner IR sweeps',
    )
    parser.add_argument(
        '--reference-tol',
        dest='reference_tol',
        type=float,
        default=1e-15,
        help='Stopping tolerance for the fine plain fp64 reference solution',
    )
    parser.add_argument(
        '--reference-maxiter',
        '--reference-max-iter',
        dest='reference_maxiter',
        type=int,
        default=150,
        help='Maximum sweeps for the fine plain fp64 reference solution',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).resolve().parents[3] / 'images' / 'scalar_sdc_plain_fp64_vs_ir_timesteps.png',
        help='Output image path',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.t_end <= args.t0:
        raise ValueError('t_end needs to be larger than t0')

    step_counts = sorted(set(args.step_counts))
    if not step_counts or any(steps < 1 for steps in step_counts):
        raise ValueError('All step counts need to be positive integers')

    if args.reference_dt is not None:
        if args.reference_dt <= 0.0:
            raise ValueError('reference_dt needs to be positive')
        interval = args.t_end - args.t0
        reference_steps = int(np.ceil(interval / args.reference_dt))
    else:
        reference_steps = args.reference_steps

    reference_steps = max(reference_steps, 32 * step_counts[-1])
    if reference_steps <= step_counts[-1]:
        raise ValueError('reference_steps needs to be larger than the coarse timestep counts')

    reference, results = run_convergence_study(
        t0=args.t0,
        t_end=args.t_end,
        step_counts=step_counts,
        reference_steps=reference_steps,
        num_nodes=args.num_nodes,
        tol=args.tol,
        maxiter=args.maxiter,
        inner_maxiter=args.inner_maxiter,
        reference_tol=args.reference_tol,
        reference_maxiter=args.reference_maxiter,
    )
    create_convergence_plot(args.output, reference, results, args.t0, args.t_end, args.num_nodes, args.tol)

    print('plain fp64 vs. IR-SDC convergence study')
    print(
        f'interval=[{args.t0}, {args.t_end}], num_nodes={args.num_nodes}, tol={args.tol}, '
        f'maxiter={args.maxiter}, inner_maxiter={args.inner_maxiter}, '
        f'reference_steps={reference_steps}, reference_dt={reference["dt"]:.6g}, '
        f'reference_tol={args.reference_tol}, reference_maxiter={args.reference_maxiter}'
    )
    print(f'reference endpoint error vs exact = {reference["endpoint_error"]:.6e}')
    print('steps    dt          plain_vs_ref     ir_vs_ref        plain_vs_exact   ir_vs_exact      |plain-ir|')
    for entry in results:
        print(
            f"{entry['steps']:<8d}"
            f"{entry['dt']:<12.6g}"
            f"{entry['plain_error_vs_reference']:<16.6e}"
            f"{entry['ir_error_vs_reference']:<16.6e}"
            f"{entry['plain_error_vs_exact']:<16.6e}"
            f"{entry['ir_error_vs_exact']:<16.6e}"
            f"{entry['method_difference']:<16.6e}"
        )
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
