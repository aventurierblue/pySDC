import numpy as np

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger
from pySDC.projects.ir.sweepers import generic_implicit_newton_sdc_ir


def run_auzinger_newton_sdc_ir(dt=0.1, outer_iterations=6, inner_iterations=2):
    description = {
        'problem_class': auzinger,
        'problem_params': {
            'newton_maxiter': 20,
            'newton_tol': 1e-14,
            'float_precision': np.dtype('float64'),
        },
        'sweeper_class': generic_implicit_newton_sdc_ir,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': 3,
            'QI': 'LU',
            'initial_guess': 'spread',
            'float_precision': np.dtype('float64'),
            'inner_float_precision': np.dtype('float32'),
            'inner_maxiter': inner_iterations,
            'inner_tol': 1e-10,
        },
        'level_params': {
            'restol': 1e-13,
            'dt': np.float64(dt),
            'residual_type': 'full_abs',
        },
        'step_params': {
            'maxiter': outer_iterations,
        },
    }

    step = Step(description=description)
    level = step.levels[0]
    problem = level.prob

    step.reset_step()
    level.status.time = 0.0
    level.u[0] = problem.u_exact(0.0)

    sweep = level.sweep
    if hasattr(sweep, 'reset_ir_stats'):
        sweep.reset_ir_stats()
    sweep.predict()
    sweep.compute_residual()

    residual_history = [float(level.status.residual)]
    for _ in range(outer_iterations):
        sweep.update_nodes()
        sweep.compute_residual()
        residual_history.append(float(level.status.residual))

    sweep.compute_end_point()

    uend = np.array(level.uend, dtype=np.float64)
    reference = np.array(problem.u_exact(dt), dtype=np.float64)
    err = np.linalg.norm(uend - reference, np.inf)

    return {
        'u0': np.array(problem.u_exact(0.0), dtype=np.float64),
        'uend': uend,
        'reference': reference,
        'dt': float(dt),
        'endpoint_error_inf': float(err),
        'residual_history': residual_history,
        'total_inner_iterations': int(sweep.total_inner_iterations),
    }


def main():
    result = run_auzinger_newton_sdc_ir()
    print('Auzinger Newton-SDC IR run')
    print(f"u0                = {result['u0']}")
    print(f"uend              = {result['uend']}")
    print(f"exact(dt)         = {result['reference']}")
    print(f"endpoint inf-error= {result['endpoint_error_inf']:.6e}")
    print(f"residual history  = {result['residual_history']}")
    print(f"total inner sweeps= {result['total_inner_iterations']}")


if __name__ == '__main__':
    main()
