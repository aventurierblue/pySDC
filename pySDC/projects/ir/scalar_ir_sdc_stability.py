import argparse
from functools import lru_cache
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.TestEquation_0D import real_scalar_testequation0d
from pySDC.projects.ir.sweepers import generic_implicit_ir


DEFAULT_REGION_OUTER_ITERATIONS = (2, 4, 6, 8)
DEFAULT_LAMBDAS = (-1.0, -2.0, -5.0, -10.0, -20.0)
DEFAULT_TABLE_DTS = (0.05, 0.1, 0.2, 0.5, 1.0)


def make_ir_description(lam, dt, num_nodes, outer_iterations, inner_iterations):
    return {
        'problem_class': real_scalar_testequation0d,
        'problem_params': {
            'lam': np.float64(lam),
            'u0': np.float64(1.0),
            'float_precision': np.dtype('float64'),
        },
        'sweeper_class': generic_implicit_ir,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'initial_guess': 'zero',
            'float_precision': np.dtype('float64'),
            'inner_float_precision': np.dtype('float32'),
            'inner_maxiter': inner_iterations,
            'inner_tol': -1.0,
        },
        'level_params': {
            'restol': -1.0,
            'dt': np.float64(dt),
            'residual_type': 'last_abs',
        },
        'step_params': {
            'maxiter': outer_iterations,
        },
    }


def run_ir_sdc_one_step(lam, dt, num_nodes, outer_iterations, inner_iterations):
    step = Step(description=make_ir_description(lam, dt, num_nodes, outer_iterations, inner_iterations))
    level = step.levels[0]
    problem = level.prob

    u0 = problem.u_exact(0.0)
    step.reset_step()
    step.init_step(u0)
    level.status.time = 0.0
    level.sweep.reset_ir_stats()
    level.sweep.predict()
    level.sweep.compute_residual()

    for _ in range(outer_iterations):
        level.sweep.update_nodes()
        level.sweep.compute_residual()

    level.sweep.compute_end_point()
    u_end = complex(np.asarray(level.uend).reshape(-1)[0])
    exact = complex(np.exp(np.float64(lam) * np.float64(dt)))

    return {
        'u_end': u_end,
        'amplification': u_end,
        'abs_amplification': abs(u_end),
        'exact': exact,
        'abs_exact': abs(exact),
        'endpoint_error': abs(u_end - exact),
        'last_residual': float(level.status.residual),
        'total_inner_iterations': int(level.sweep.total_inner_iterations),
    }


@lru_cache(maxsize=None)
def get_collocation_data(num_nodes):
    step = Step(description=make_ir_description(-1.0, 1.0, num_nodes, 1, 1))
    level = step.levels[0]

    return {
        'Q': np.asarray(level.sweep.coll.Qmat[1:, 1:], dtype=np.complex128),
        'Q_delta': np.asarray(level.sweep.QI[1:, 1:], dtype=np.complex128),
        'weights': np.asarray(level.sweep.coll.weights, dtype=np.complex128),
        'right_is_node': bool(level.sweep.coll.right_is_node),
        'do_coll_update': bool(level.sweep.params.do_coll_update),
    }


def amplification_from_stages(z, stages, collocation_data):
    if collocation_data['right_is_node'] and not collocation_data['do_coll_update']:
        return complex(stages[-1])
    return complex(1.0 + z * np.dot(collocation_data['weights'], stages))


def ir_sdc_amplification_factor(z, num_nodes, outer_iterations, inner_iterations):
    collocation_data = get_collocation_data(num_nodes)
    q = collocation_data['Q']
    q_delta = collocation_data['Q_delta']
    num_nodes = q.shape[0]

    identity = np.eye(num_nodes, dtype=np.complex128)
    ones = np.ones(num_nodes, dtype=np.complex128)
    collocation_matrix = identity - z * q
    preconditioner = identity - z * q_delta

    try:
        collocation_solution = np.linalg.solve(collocation_matrix, ones)
        iteration_matrix = np.linalg.solve(preconditioner, z * (q - q_delta))
    except np.linalg.LinAlgError:
        return complex(np.inf)

    grouped_iteration_matrix = identity.copy()
    for _ in range(inner_iterations):
        grouped_iteration_matrix = grouped_iteration_matrix @ iteration_matrix

    outer_error_matrix = identity.copy()
    for _ in range(outer_iterations):
        outer_error_matrix = outer_error_matrix @ grouped_iteration_matrix

    stage_values = collocation_solution - outer_error_matrix @ collocation_solution
    return amplification_from_stages(z, stage_values, collocation_data)


def compute_configuration_table(lambdas, dts, num_nodes, outer_iterations, inner_iterations):
    rows = []
    max_model_difference = 0.0

    for lam in lambdas:
        for dt in dts:
            actual = run_ir_sdc_one_step(lam, dt, num_nodes, outer_iterations, inner_iterations)
            z = np.float64(lam) * np.float64(dt)
            model = ir_sdc_amplification_factor(z, num_nodes, outer_iterations, inner_iterations)
            model_difference = abs(actual['amplification'] - model)
            max_model_difference = max(max_model_difference, model_difference)

            rows.append(
                {
                    'lam': lam,
                    'dt': dt,
                    'z': z,
                    'abs_actual': actual['abs_amplification'],
                    'abs_exact': actual['abs_exact'],
                    'endpoint_error': actual['endpoint_error'],
                    'model_difference': model_difference,
                    'stable': actual['abs_amplification'] <= 1.0,
                    'total_inner_iterations': actual['total_inner_iterations'],
                }
            )

    return rows, max_model_difference


def compute_dt_scan_curves(lambdas, dt_values, num_nodes, outer_iterations, inner_iterations):
    curves = []
    for lam in lambdas:
        errors = []

        for dt in dt_values:
            result = run_ir_sdc_one_step(lam, dt, num_nodes, outer_iterations, inner_iterations)
            errors.append(result['endpoint_error'])

        curves.append(
            {
                'lam': lam,
                'z_values': np.abs(np.float64(lam) * np.asarray(dt_values, dtype=np.float64)),
                'dt_values': np.asarray(dt_values, dtype=np.float64),
                'errors': np.asarray(errors, dtype=np.float64),
            }
        )

    return curves


def compute_stability_region(num_nodes, inner_iterations, outer_iterations_list, re_lims, im_lims, n_points):
    collocation_data = get_collocation_data(num_nodes)
    q = collocation_data['Q']
    q_delta = collocation_data['Q_delta']
    num_stages = q.shape[0]
    identity = np.eye(num_stages, dtype=np.complex128)
    ones = np.ones(num_stages, dtype=np.complex128)
    outer_iterations_list = tuple(sorted(set(outer_iterations_list)))
    max_outer_iterations = max(outer_iterations_list)
    divergence_threshold = 1e12

    re_values = np.linspace(re_lims[0], re_lims[1], n_points)
    im_values = np.linspace(im_lims[0], im_lims[1], n_points)
    amplifications = {outer_iterations: np.empty((n_points, n_points), dtype=np.float64) for outer_iterations in outer_iterations_list}

    for row, imag_part in enumerate(im_values):
        for col, real_part in enumerate(re_values):
            z = complex(real_part, imag_part)
            collocation_matrix = identity - z * q
            preconditioner = identity - z * q_delta

            try:
                collocation_solution = np.linalg.solve(collocation_matrix, ones)
                iteration_matrix = np.linalg.solve(preconditioner, z * (q - q_delta))
            except np.linalg.LinAlgError:
                for outer_iterations in outer_iterations_list:
                    amplifications[outer_iterations][row, col] = np.inf
                continue

            grouped_iteration_matrix = identity.copy()
            for _ in range(inner_iterations):
                grouped_iteration_matrix = grouped_iteration_matrix @ iteration_matrix
                if not np.all(np.isfinite(grouped_iteration_matrix)) or np.linalg.norm(grouped_iteration_matrix, ord=np.inf) > divergence_threshold:
                    grouped_iteration_matrix = None
                    break

            if grouped_iteration_matrix is None:
                for outer_iterations in outer_iterations_list:
                    amplifications[outer_iterations][row, col] = np.inf
                continue

            outer_error_matrix = identity.copy()
            for outer_iterations in range(1, max_outer_iterations + 1):
                outer_error_matrix = outer_error_matrix @ grouped_iteration_matrix
                if not np.all(np.isfinite(outer_error_matrix)) or np.linalg.norm(outer_error_matrix, ord=np.inf) > divergence_threshold:
                    for remaining_outer_iterations in outer_iterations_list:
                        if remaining_outer_iterations >= outer_iterations:
                            amplifications[remaining_outer_iterations][row, col] = np.inf
                    break
                if outer_iterations in amplifications:
                    stage_values = collocation_solution - outer_error_matrix @ collocation_solution
                    amplification = amplification_from_stages(z, stage_values, collocation_data)
                    amplifications[outer_iterations][row, col] = abs(amplification)

    return re_values, im_values, amplifications


def create_dt_scan_plot(output_path, curves, outer_iterations, inner_iterations):
    fig, ax = plt.subplots(figsize=(10.5, 6.6), constrained_layout=True)

    for curve in curves:
        ax.loglog(
            curve['z_values'],
            curve['errors'],
            linewidth=1.6,
            marker='o',
            markersize=3.0,
            label=fr'IR-SDC, $\lambda={curve["lam"]:g}$',
        )

    ax.set_xlabel(r'$|z| = |\lambda \Delta t|$')
    ax.set_ylabel(r'One-step error $|u_1^{\mathrm{IR}} - e^z|$')
    ax.set_title(f'Scalar IR-SDC real-axis error scan, outer={outer_iterations}, inner={inner_iterations}')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower center')

    fig.text(
        0.5,
        -0.02,
        'Each curve shows the one-step IR-SDC error against the exact test-equation solution for fixed lambda.',
        ha='center',
        va='top',
        fontsize=9,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_stability_region_plot(output_path, re_values, im_values, amplifications, num_nodes, inner_iterations):
    outer_iterations_list = tuple(sorted(amplifications.keys()))
    ncols = 2 if len(outer_iterations_list) > 1 else 1
    nrows = int(ceil(len(outer_iterations_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4 * ncols, 7.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    stable_color = '#4c78a8'
    unstable_color = '#d9d9d9'

    for ax, outer_iterations in zip(axes, outer_iterations_list, strict=True):
        stable_mask = amplifications[outer_iterations] <= 1.0
        ax.contourf(
            re_values,
            im_values,
            stable_mask.astype(np.int8),
            levels=[-0.5, 0.5, 1.5],
            colors=[unstable_color, stable_color],
        )
        ax.contour(re_values, im_values, amplifications[outer_iterations], levels=[1.0], colors='black', linewidths=1.2)
        ax.axhline(0.0, color='black', linewidth=0.7, alpha=0.5)
        ax.axvline(0.0, color='black', linewidth=0.7, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlabel(r'Re$(z)$')
        ax.set_ylabel(r'Im$(z)$')
        ax.set_title(f'Outer iterations = {outer_iterations}')

    for ax in axes[len(outer_iterations_list) :]:
        fig.delaxes(ax)

    fig.legend(
        handles=[Patch(facecolor=stable_color, edgecolor='none', label=r'stable: $|R_k(z)| \leq 1$'), Patch(facecolor=unstable_color, edgecolor='none', label=r'unstable: $|R_k(z)| > 1$')],
        loc='lower center',
        ncol=2,
    )

    fig.suptitle(f'Scalar IR-SDC stability region, num_nodes={num_nodes}, inner_iterations={inner_iterations}')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_configuration_table(rows, num_nodes, outer_iterations, inner_iterations, max_model_difference):
    print('Scalar IR-SDC stability scan')
    print(
        f'num_nodes={num_nodes}, outer_iterations={outer_iterations}, inner_iterations={inner_iterations}, '
        f'max(|actual-model|)={max_model_difference:.3e}'
    )
    print('lambda      dt          z=lambda*dt   |u1|          |exp(z)|      endpoint_error model_diff     stable')
    for row in rows:
        print(
            f"{row['lam']:<12.5g}"
            f"{row['dt']:<12.5g}"
            f"{row['z']:<14.5g}"
            f"{row['abs_actual']:<14.6e}"
            f"{row['abs_exact']:<14.6e}"
            f"{row['endpoint_error']:<15.6e}"
            f"{row['model_difference']:<15.6e}"
            f"{'stable' if row['stable'] else 'unstable'}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze scalar IR-SDC stability for different lambda and dt configurations.')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=3, help='Number of collocation nodes')
    parser.add_argument('--inner-iterations', dest='inner_iterations', type=int, default=5, help='Fixed inner SDC sweeps per outer IR step')
    parser.add_argument(
        '--region-outer-iterations',
        dest='region_outer_iterations',
        nargs='+',
        type=int,
        default=DEFAULT_REGION_OUTER_ITERATIONS,
        help='Outer IR iteration counts shown in the complex stability-region plot',
    )
    parser.add_argument(
        '--scan-outer-iterations',
        dest='scan_outer_iterations',
        type=int,
        default=6,
        help='Outer IR iteration count used for the real lambda-dt scan',
    )
    parser.add_argument('--lambdas', nargs='+', type=float, default=DEFAULT_LAMBDAS, help='Real lambda values for actual pySDC scans')
    parser.add_argument('--table-dts', dest='table_dts', nargs='+', type=float, default=DEFAULT_TABLE_DTS, help='Representative dt values printed in the table')
    parser.add_argument('--scan-dt-min', dest='scan_dt_min', type=float, default=1e-2, help='Smallest dt used in the real-axis scan plot')
    parser.add_argument('--scan-dt-max', dest='scan_dt_max', type=float, default=1.0, help='Largest dt used in the real-axis scan plot')
    parser.add_argument('--scan-num-dt', dest='scan_num_dt', type=int, default=40, help='Number of dt values used in the real-axis scan plot')
    parser.add_argument('--re-min', dest='re_min', type=float, default=-12.0, help='Minimum real part of z=lambda dt for the stability region')
    parser.add_argument('--re-max', dest='re_max', type=float, default=2.0, help='Maximum real part of z=lambda dt for the stability region')
    parser.add_argument('--im-max', dest='im_max', type=float, default=8.0, help='Maximum imaginary magnitude for the stability region')
    parser.add_argument('--region-points', dest='region_points', type=int, default=161, help='Grid points per axis for the stability region')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        type=Path,
        default=Path(__file__).resolve().parents[3] / 'images',
        help='Directory for output images',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_nodes < 1:
        raise ValueError('num_nodes needs to be positive')
    if args.inner_iterations < 1:
        raise ValueError('inner_iterations needs to be positive')
    if args.scan_outer_iterations < 1:
        raise ValueError('scan_outer_iterations needs to be positive')
    if any(outer_iterations < 1 for outer_iterations in args.region_outer_iterations):
        raise ValueError('region_outer_iterations need to be positive')
    if args.scan_dt_min <= 0.0 or args.scan_dt_max <= 0.0 or args.scan_dt_min >= args.scan_dt_max:
        raise ValueError('scan_dt_min and scan_dt_max need to satisfy 0 < scan_dt_min < scan_dt_max')
    if any(dt <= 0.0 for dt in args.table_dts):
        raise ValueError('table_dts need to be positive')
    if args.region_points < 16:
        raise ValueError('region_points needs to be at least 16')

    rows, max_model_difference = compute_configuration_table(
        lambdas=tuple(args.lambdas),
        dts=tuple(args.table_dts),
        num_nodes=args.num_nodes,
        outer_iterations=args.scan_outer_iterations,
        inner_iterations=args.inner_iterations,
    )
    print_configuration_table(rows, args.num_nodes, args.scan_outer_iterations, args.inner_iterations, max_model_difference)

    dt_values = np.geomspace(args.scan_dt_min, args.scan_dt_max, args.scan_num_dt)
    curves = compute_dt_scan_curves(
        lambdas=tuple(args.lambdas),
        dt_values=dt_values,
        num_nodes=args.num_nodes,
        outer_iterations=args.scan_outer_iterations,
        inner_iterations=args.inner_iterations,
    )
    dt_scan_path = args.output_dir / 'scalar_ir_sdc_lambda_dt_scan.png'
    create_dt_scan_plot(dt_scan_path, curves, args.scan_outer_iterations, args.inner_iterations)

    re_values, im_values, amplifications = compute_stability_region(
        num_nodes=args.num_nodes,
        inner_iterations=args.inner_iterations,
        outer_iterations_list=tuple(sorted(set(args.region_outer_iterations))),
        re_lims=(args.re_min, args.re_max),
        im_lims=(-args.im_max, args.im_max),
        n_points=args.region_points,
    )
    stability_region_path = args.output_dir / 'scalar_ir_sdc_stability_region.png'
    create_stability_region_plot(
        stability_region_path,
        re_values,
        im_values,
        amplifications,
        args.num_nodes,
        args.inner_iterations,
    )

    print(f'Wrote {dt_scan_path}')
    print(f'Wrote {stability_region_path}')


if __name__ == '__main__':
    main()
