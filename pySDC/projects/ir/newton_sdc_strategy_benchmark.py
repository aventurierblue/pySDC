import argparse
from time import perf_counter

from pySDC.core.step import Step
from pySDC.projects.ir.allencahn_fp64_vs_newton_sdc_ir import make_gmres_ir_description as make_allencahn_description
from pySDC.projects.ir.fisher_fp64_vs_newton_sdc_ir import make_gmres_ir_description as make_fisher_description


def build_description(problem, size, dt, num_nodes, target_tol, maxiter, inner_solver, inner_qi):
    common = {
        'dt': dt,
        'num_nodes': num_nodes,
        'outer_tol': target_tol,
        'outer_maxiter': maxiter,
        'inner_solver': inner_solver,
        'inner_qi': inner_qi,
        'cache_inner_step': True,
    }

    if problem == 'allencahn':
        return make_allencahn_description(
            nvars=(size, size),
            eps=0.04,
            radius=0.25,
            **common,
        )
    if problem == 'fisher':
        return make_fisher_description(
            nvars=size,
            nu=1.0,
            lambda0=2.0,
            **common,
        )

    raise ValueError(f'Unknown problem {problem}')


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
        'total_inner_iterations': int(getattr(level.sweep, 'total_inner_iterations', 0)),
    }


def benchmark_configuration(problem, size, inner_solver, inner_qi, repeats, dt, outer_sweeps, num_nodes, target_tol, maxiter):
    description = build_description(
        problem=problem,
        size=size,
        dt=dt,
        num_nodes=num_nodes,
        target_tol=target_tol,
        maxiter=maxiter,
        inner_solver=inner_solver,
        inner_qi=inner_qi,
    )

    warm_step = initialize_step(description)
    run_outer_sweeps(warm_step, 1)

    best = None
    for _ in range(repeats):
        step = initialize_step(description)
        result = run_outer_sweeps(step, outer_sweeps)
        if best is None or result['elapsed'] < best['elapsed']:
            best = result

    best['problem'] = problem
    best['size'] = size
    best['inner_solver'] = inner_solver
    best['inner_qi'] = inner_qi
    return best


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Newton-SDC strategies for Allen-Cahn and Fisher.')
    parser.add_argument('--dt', type=float, default=1e-3, help='Single-step timestep size')
    parser.add_argument('--outer-sweeps', type=int, default=3, help='Maximum outer sweeps to run')
    parser.add_argument('--num-nodes', dest='num_nodes', type=int, default=3, help='Number of collocation nodes')
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-8, help='Outer residual tolerance')
    parser.add_argument('--maxiter', type=int, default=50, help='Maximum outer iterations configured on the step')
    parser.add_argument('--repeats', type=int, default=2, help='Number of timing repeats per configuration')
    return parser.parse_args()


def main():
    args = parse_args()
    problems = {
        'allencahn': [64, 128],
        'fisher': [255, 511],
    }
    solvers = ['sdc', 'gmres', 'lgmres']
    preconditioners = ['LU', 'VDHS', 'MIN-SR-S']

    print('Newton-SDC strategy benchmark')
    print(
        f'dt={args.dt}, outer_sweeps={args.outer_sweeps}, num_nodes={args.num_nodes}, '
        f'target_tol={args.target_tol}, maxiter={args.maxiter}, repeats={args.repeats}'
    )
    print('problem     size    solver   inner_QI      elapsed[s]   outer   inner   final_residual')

    for problem, sizes in problems.items():
        for size in sizes:
            formatted_size = f'({size}, {size})' if problem == 'allencahn' else str(size)
            for inner_solver in solvers:
                for inner_qi in preconditioners:
                    result = benchmark_configuration(
                        problem=problem,
                        size=size,
                        inner_solver=inner_solver,
                        inner_qi=inner_qi,
                        repeats=args.repeats,
                        dt=args.dt,
                        outer_sweeps=args.outer_sweeps,
                        num_nodes=args.num_nodes,
                        target_tol=args.target_tol,
                        maxiter=args.maxiter,
                    )
                    print(
                        f'{problem:<11} '
                        f'{formatted_size:<7} '
                        f'{inner_solver:<8} '
                        f'{inner_qi:<13} '
                        f'{result["elapsed"]:>10.6f} '
                        f'{result["completed_outer_sweeps"]:>7d} '
                        f'{result["total_inner_iterations"]:>7d} '
                        f'{result["final_residual"]:>16.6e}'
                    )


if __name__ == '__main__':
    main()
