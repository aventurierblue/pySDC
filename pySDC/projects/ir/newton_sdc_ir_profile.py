import argparse
import cProfile
import io
from time import perf_counter
import pstats

from pySDC.core.step import Step
from pySDC.projects.ir.allencahn_fp64_vs_newton_sdc_ir import (
    make_gmres_ir_description,
    make_newton_sdc_fp64_description,
    make_newton_sdc_ir_description,
    make_plain_description,
)


def initialize_step(description, t0=0.0):
    step = Step(description=description)
    level = step.levels[0]
    problem = level.prob

    step.reset_step()
    step.init_step(problem.u_exact(t0))
    level.status.time = t0

    if hasattr(level.sweep, 'reset_ir_stats'):
        level.sweep.reset_ir_stats()

    level.sweep.predict()
    level.sweep.compute_residual()
    return step


def run_outer_sweeps(step, outer_sweeps):
    level = step.levels[0]
    residual_history = [float(level.status.residual)]

    start = perf_counter()
    for _ in range(outer_sweeps):
        level.sweep.update_nodes()
        level.sweep.compute_residual()
        residual_history.append(float(level.status.residual))
        if level.status.residual <= level.params.restol:
            break
    elapsed = perf_counter() - start

    return {
        'elapsed': elapsed,
        'completed_outer_sweeps': len(residual_history) - 1,
        'final_residual': residual_history[-1],
        'residual_history': residual_history,
        'total_inner_iterations': int(getattr(level.sweep, 'total_inner_iterations', 0)),
    }


def make_description(method, dt, num_nodes, target_tol, maxiter, nvars, eps, radius):
    if method == 'newton-sdc-ir':
        return make_newton_sdc_ir_description(
            dt=dt,
            num_nodes=num_nodes,
            outer_tol=target_tol,
            outer_maxiter=maxiter,
            nvars=nvars,
            eps=eps,
            radius=radius,
        )

    if method == 'newton-sdc-fp64':
        return make_newton_sdc_fp64_description(
            dt=dt,
            num_nodes=num_nodes,
            outer_tol=target_tol,
            outer_maxiter=maxiter,
            nvars=nvars,
            eps=eps,
            radius=radius,
        )

    if method == 'gmres-ir':
        return make_gmres_ir_description(
            dt=dt,
            num_nodes=num_nodes,
            outer_tol=target_tol,
            outer_maxiter=maxiter,
            nvars=nvars,
            eps=eps,
            radius=radius,
        )

    if method == 'plain-fp64':
        return make_plain_description(
            dt=dt,
            num_nodes=num_nodes,
            restol=target_tol,
            maxiter=maxiter,
            nvars=nvars,
            eps=eps,
            radius=radius,
        )

    raise ValueError(f'Unknown method {method}')


def profile_configuration(method, dt, outer_sweeps, num_nodes, target_tol, maxiter, nvars, eps, radius, sort_by, stats_lines):
    description = make_description(method, dt, num_nodes, target_tol, maxiter, nvars, eps, radius)

    # Warm up once so one-time setup costs do not dominate the profile.
    warm_step = initialize_step(description)
    run_outer_sweeps(warm_step, 1)

    step = initialize_step(description)
    profiler = cProfile.Profile()
    profiler.enable()
    result = run_outer_sweeps(step, outer_sweeps)
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats(sort_by).print_stats(stats_lines)
    result['stats'] = stream.getvalue()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Profile the Newton-SDC IR hot path on Allen-Cahn without plotting or reference solves.')
    parser.add_argument(
        '--method',
        choices=('newton-sdc-ir', 'newton-sdc-fp64', 'gmres-ir', 'plain-fp64'),
        default='newton-sdc-ir',
        help='Method to profile',
    )
    parser.add_argument('--dt', type=float, default=1e-3, help='Single-step timestep size')
    parser.add_argument('--outer-sweeps', type=int, default=2, help='Maximum outer sweeps to profile')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=3, help='Number of collocation nodes')
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-8, help='Outer residual tolerance')
    parser.add_argument('--maxiter', type=int, default=20, help='Maximum outer iterations configured on the step')
    parser.add_argument('--nvars', nargs=2, type=int, default=(128, 128), help='Allen-Cahn grid shape')
    parser.add_argument('--eps', type=float, default=0.04, help='Allen-Cahn epsilon parameter')
    parser.add_argument('--radius', type=float, default=0.25, help='Allen-Cahn initial radius')
    parser.add_argument('--sort-by', default='cumulative', help='cProfile sort key')
    parser.add_argument('--stats-lines', dest='stats_lines', type=int, default=30, help='Number of profile rows to print')
    return parser.parse_args()


def main():
    args = parse_args()
    result = profile_configuration(
        method=args.method,
        dt=args.dt,
        outer_sweeps=args.outer_sweeps,
        num_nodes=args.num_nodes,
        target_tol=args.target_tol,
        maxiter=args.maxiter,
        nvars=tuple(args.nvars),
        eps=args.eps,
        radius=args.radius,
        sort_by=args.sort_by,
        stats_lines=args.stats_lines,
    )

    print('Newton-SDC IR profile')
    print(
        f'method={args.method}, dt={args.dt}, outer_sweeps={args.outer_sweeps}, '
        f'num_nodes={args.num_nodes}, nvars={tuple(args.nvars)}, target_tol={args.target_tol}'
    )
    print(f"elapsed={result['elapsed']:.6e} s")
    print(f"completed_outer_sweeps={result['completed_outer_sweeps']}")
    print(f"final_residual={result['final_residual']:.6e}")
    print(f"total_inner_iterations={result['total_inner_iterations']}")
    print(f"residual_history={result['residual_history']}")
    print()
    print(result['stats'])


if __name__ == '__main__':
    main()
