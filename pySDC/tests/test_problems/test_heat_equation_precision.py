import numpy as np
import pytest


@pytest.mark.base
@pytest.mark.parametrize('precision', [np.dtype('float32'), np.dtype('float64')])
def test_heat_equation_respects_float_precision(precision):
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced

    problem = heatNd_forced(
        nvars=31,
        nu=0.1,
        freq=4,
        bc='dirichlet-zero',
        float_precision=precision,
    )

    u = problem.u_exact(precision.type(0.1))
    f = problem.eval_f(u, precision.type(0.1))
    u_sol = problem.solve_system(problem.dtype_u(u), precision.type(0.05), u, precision.type(0.1))

    assert problem.float_precision == precision
    assert problem.xvalues.dtype == precision
    assert problem.A.dtype == precision
    assert problem.Id.dtype == precision
    assert u.dtype == precision
    assert f.impl.dtype == precision
    assert f.expl.dtype == precision
    assert u_sol.dtype == precision


@pytest.mark.base
def test_heat_equation_unsupported_high_precision_raises():
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced

    with pytest.raises(TypeError, match='does not support float128'):
        heatNd_forced(
            nvars=31,
            nu=0.1,
            freq=4,
            bc='dirichlet-zero',
            float_precision=np.dtype(np.longdouble),
        )
