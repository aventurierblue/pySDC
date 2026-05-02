import weakref

import numpy as np

from pySDC.core.problem import Problem
from pySDC.implementations.problem_classes.generic_ND_FD import GenericNDimFinDiff
from pySDC.projects.Resilience.sweepers import generic_implicit_efficient


_INNER_CORRECTION_CACHE = weakref.WeakKeyDictionary()


class _AffineLinearCorrectionProblem(Problem):
    """
    Temporary correction problem for affine-linear implicit problems.

    If the original problem satisfies ``f(u, t) = A(t) u + g(t)``, then the correction
    equation uses only the homogeneous part ``A(t) d``. This wrapper constructs that
    homogeneous operator from the original problem via ``f(u, t) - f(0, t)`` and adjusts
    the implicit solve accordingly.
    """

    def __init__(self, base_problem_class, base_problem_params):
        self.base_problem = base_problem_class(**base_problem_params)
        self.dtype_u = self.base_problem.dtype_u
        self.dtype_f = self.base_problem.dtype_f
        self._zero_state = self.dtype_u(self.base_problem.init, val=0.0)
        super().__init__(init=self.base_problem.init, float_precision=self.base_problem.float_precision)

    def _forcing(self, t):
        return self.base_problem.eval_f(self._zero_state, t)

    def eval_f(self, u, t):
        f = self.base_problem.eval_f(u, t)
        f -= self._forcing(t)
        return f

    def solve_system(self, rhs, factor, u0, t):
        adjusted_rhs = self.dtype_u(rhs)
        adjusted_rhs -= factor * self._forcing(t)
        return self.base_problem.solve_system(adjusted_rhs, factor, u0, t)


class _InnerCorrectionData:
    def __init__(self, sweeper):
        outer_problem = sweeper.level.prob
        problem_params = sweeper._clone_problem_params(outer_problem, sweeper.inner_float_precision)
        self._uses_homogeneous_linear_shortcuts = (
            isinstance(outer_problem, GenericNDimFinDiff)
            and type(outer_problem).eval_f is GenericNDimFinDiff.eval_f
            and type(outer_problem).solve_system is GenericNDimFinDiff.solve_system
            and getattr(outer_problem, 'solver_type', None) == 'direct'
        )
        self.problem = (
            type(outer_problem)(**problem_params)
            if self._uses_homogeneous_linear_shortcuts
            else _AffineLinearCorrectionProblem(type(outer_problem), problem_params)
        )
        self.qmat = np.asarray(sweeper.coll.Qmat, dtype=sweeper.inner_float_precision)
        self.qi = np.asarray(sweeper.QI, dtype=sweeper.inner_float_precision)
        self.nodes = np.asarray(sweeper.coll.nodes, dtype=sweeper.inner_float_precision)
        self.num_nodes = sweeper.coll.num_nodes
        self.u = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes + 1)]
        self.f = [self.problem.dtype_f(self.problem.init, val=0.0) for _ in range(self.num_nodes + 1)]
        self.tau = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
        self.residuals = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
        self.integrals = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
        self.rhs = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]

        if self._uses_homogeneous_linear_shortcuts:
            self._linear_operator = self.problem.A.tocsc()
            self._identity = self.problem.Id.tocsc()
            self._solve_cache = {}

    def eval_f_into(self, out, u, t):
        if self._uses_homogeneous_linear_shortcuts:
            out.view(np.ndarray)[:] = self._linear_operator.dot(u.view(np.ndarray).reshape(-1)).reshape(out.shape)
            return out

        out[:] = self.problem.eval_f(u, t)
        return out

    def solve_system_into(self, out, rhs, factor, t):
        if self._uses_homogeneous_linear_shortcuts:
            out_view = out.view(np.ndarray)
            rhs_view = rhs.view(np.ndarray).reshape(-1)

            if factor == 0:
                np.copyto(out_view, rhs.view(np.ndarray))
                return out

            key = float(factor)
            solver = self._solve_cache.get(key)
            if solver is None:
                from scipy.sparse.linalg import splu

                solver = splu((self._identity - factor * self._linear_operator).tocsc())
                self._solve_cache[key] = solver

            out_view[:] = solver.solve(rhs_view).reshape(out.shape)
            return out

        out[:] = self.problem.solve_system(rhs, factor, out, t)
        return out


class generic_implicit_ir(generic_implicit_efficient):
    """
    Mixed-precision iterative-refinement SDC sweeper.

    One call to ``update_nodes`` performs one outer iterative-refinement update on the
    collocation system. The inner correction equation is approximated in lower
    precision using the SDC preconditioner.

    By default, the scalar linear test equation uses a specialized low-storage matrix
    path. For more general affine-linear implicit problems, or when
    ``use_scalar_fast_path=False``, a temporary low-precision inner step is built on
    the homogeneous correction problem, so the outer step does not retain additional
    low-precision stage or operator state.
    """

    def __init__(self, params, level):
        params = dict(params)
        params.setdefault('inner_float_precision', np.dtype('float32'))
        params.setdefault('inner_maxiter', 5)
        params.setdefault('inner_tol', 1e-9)
        params.setdefault('use_scalar_fast_path', True)
        params.setdefault('cache_inner_step', True)
        super().__init__(params, level)

        self.outer_float_precision = np.dtype(self.params.float_precision)
        self.inner_float_precision = np.dtype(self.params.inner_float_precision)
        self.last_inner_iterations = 0
        self.total_inner_iterations = 0

    def reset_ir_stats(self):
        self.last_inner_iterations = 0
        self.total_inner_iterations = 0

    def _uses_outer_rhs_shortcuts(self):
        problem = self.level.prob
        return (
            not self._is_scalar_linear_test_equation()
            and isinstance(problem, GenericNDimFinDiff)
            and type(problem).eval_f is GenericNDimFinDiff.eval_f
            and type(problem).solve_system is GenericNDimFinDiff.solve_system
        )

    def _predict_with_outer_rhs_storage(self):
        L = self.level
        P = L.prob

        for m in range(1, self.coll.num_nodes + 1):
            node_time = L.time + L.dt * self.coll.nodes[m - 1]
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m], node_time)
            elif self.params.initial_guess == 'copy':
                if L.f[0] is None:
                    L.f[0] = P.eval_f(L.u[0], L.time)
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.dtype_f(L.f[0])
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.eval_f(L.u[m], node_time)
            elif self.params.initial_guess == 'random':
                L.u[m] = P.dtype_u(init=P.init, val=self.rng.rand(1)[0])
                L.f[m] = P.eval_f(L.u[m], node_time)
            else:
                raise ValueError(f'initial_guess option {self.params.initial_guess} not implemented')

        L.status.unlocked = True
        L.status.updated = True

    def predict(self):
        if self._uses_outer_rhs_shortcuts():
            self._predict_with_outer_rhs_storage()
            return None

        L = self.level
        P = L.prob

        for m in range(1, self.coll.num_nodes + 1):
            if self.params.initial_guess in ('spread', 'copy'):
                L.u[m] = P.dtype_u(L.u[0])
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
            elif self.params.initial_guess == 'random':
                L.u[m] = P.dtype_u(init=P.init, val=self.rng.rand(1)[0])
            else:
                raise ValueError(f'initial_guess option {self.params.initial_guess} not implemented')

        # Keep only the outer stage values. Residuals and end-point values are
        # recomputed on demand so the outer step does not retain node-wise RHS arrays.
        L.status.unlocked = True
        L.status.updated = True

    def _get_stage_vector(self):
        L = self.level
        values = np.zeros(self.coll.num_nodes, dtype=self.outer_float_precision)
        for m in range(self.coll.num_nodes):
            if L.u[m + 1] is not None:
                values[m] = self.outer_float_precision.type(np.asarray(L.u[m + 1]).reshape(-1)[0])
        return values

    @staticmethod
    def _clone_problem_params(problem, float_precision):
        params = dict(problem.params)
        if 'float_precision' in params:
            params['float_precision'] = np.dtype(float_precision)
        return params

    def _get_inner_correction_data(self):
        if not self.params.cache_inner_step:
            return _InnerCorrectionData(self)

        inner_data = _INNER_CORRECTION_CACHE.get(self)
        if inner_data is None:
            inner_data = _InnerCorrectionData(self)
            _INNER_CORRECTION_CACHE[self] = inner_data

        return inner_data

    def _initialize_inner_correction(self, inner_data, outer_residuals):
        for m in range(inner_data.num_nodes + 1):
            inner_data.u[m][:] = 0.0
            inner_data.f[m][:] = 0.0

        for m, residual in enumerate(outer_residuals):
            inner_data.tau[m][:] = residual

    def _compute_inner_correction_residual(self, inner_data, dt_inner):
        residual_norms = np.zeros(inner_data.num_nodes, dtype=np.float64)

        for m in range(inner_data.num_nodes):
            residual = inner_data.residuals[m]
            residual_view = residual.view(np.ndarray)
            np.copyto(residual_view, inner_data.tau[m].view(np.ndarray))
            for j in range(1, inner_data.num_nodes + 1):
                residual_view += dt_inner * inner_data.qmat[m + 1, j] * inner_data.f[j].view(np.ndarray)
            residual_view -= inner_data.u[m + 1].view(np.ndarray)
            residual_norms[m] = abs(residual)

        residual_type = self.level.params.residual_type
        if residual_type == 'full_abs':
            return float(np.max(residual_norms))
        if residual_type == 'last_abs':
            return float(residual_norms[-1])

        u0_norm = abs(self.level.u[0])
        if residual_type == 'full_rel':
            return float(np.max(residual_norms) / u0_norm)
        if residual_type == 'last_rel':
            return float(residual_norms[-1] / u0_norm)

        raise ValueError(
            f'residual_type = {residual_type} not implemented, choose '
            'full_abs, last_abs, full_rel or last_rel instead'
        )

    def _perform_inner_correction_sweep(self, inner_data, dt_inner, time_inner):
        inner_problem = inner_data.problem

        for m in range(inner_data.num_nodes):
            integral = inner_data.integrals[m]
            integral_view = integral.view(np.ndarray)
            np.copyto(integral_view, inner_data.tau[m].view(np.ndarray))
            for j in range(1, inner_data.num_nodes + 1):
                integral_view += (
                    dt_inner
                    * (inner_data.qmat[m + 1, j] - inner_data.qi[m + 1, j])
                    * inner_data.f[j].view(np.ndarray)
                )

        for m in range(inner_data.num_nodes):
            rhs = inner_data.rhs[m]
            np.copyto(rhs.view(np.ndarray), inner_data.integrals[m].view(np.ndarray))
            for j in range(1, m + 1):
                rhs.view(np.ndarray)[:] += dt_inner * inner_data.qi[m + 1, j] * inner_data.f[j].view(np.ndarray)

            node_time = time_inner + dt_inner * inner_data.nodes[m]
            inner_data.solve_system_into(inner_data.u[m + 1], rhs, dt_inner * inner_data.qi[m + 1, m + 1], node_time)
            inner_data.eval_f_into(inner_data.f[m + 1], inner_data.u[m + 1], node_time)

    def _is_scalar_linear_test_equation(self):
        if not self.params.use_scalar_fast_path:
            return False

        problem = self.level.prob
        return hasattr(problem, 'lam') and getattr(problem, 'init', None) == (1, None, self.outer_float_precision)

    def _compute_full_residuals(self):
        L = self.level
        P = L.prob

        residuals = []
        for m in range(self.coll.num_nodes):
            res = P.dtype_u(P.init, val=0.0)
            for j in range(1, self.coll.num_nodes + 1):
                if L.u[j] is None:
                    continue
                f_j = L.f[j] if L.f[j] is not None else P.eval_f(L.u[j], L.time + L.dt * self.coll.nodes[j - 1])
                res += L.dt * self.coll.Qmat[m + 1, j] * f_j
            res += L.u[0] - L.u[m + 1]
            if L.tau[m] is not None:
                res += L.tau[m]
            residuals.append(res)
        return residuals

    def compute_residual(self, stage=''):
        L = self.level
        P = L.prob

        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        if not self._is_scalar_linear_test_equation():
            if self._uses_outer_rhs_shortcuts():
                return generic_implicit_efficient.compute_residual(self, stage=stage)

            residuals = self._compute_full_residuals()
            residual_norms = np.array([abs(res) for res in residuals], dtype=np.float64)

            if L.params.residual_type == 'full_abs':
                L.status.residual = float(np.max(residual_norms))
            elif L.params.residual_type == 'last_abs':
                L.status.residual = float(residual_norms[-1])
            elif L.params.residual_type == 'full_rel':
                L.status.residual = float(np.max(residual_norms) / abs(L.u[0]))
            elif L.params.residual_type == 'last_rel':
                L.status.residual = float(residual_norms[-1] / abs(L.u[0]))
            else:
                raise ValueError(
                    f'residual_type = {L.params.residual_type} not implemented, choose '
                    'full_abs, last_abs, full_rel or last_rel instead'
                )

            L.status.updated = False
            return None

        stage_values = self._get_stage_vector()
        z_outer = self.outer_float_precision.type(L.dt) * self.outer_float_precision.type(P.lam)
        q_outer = np.asarray(self.coll.Qmat[1:, 1:], dtype=self.outer_float_precision)
        u0 = self.outer_float_precision.type(np.asarray(L.u[0]).reshape(-1)[0])
        residuals = u0 - stage_values + z_outer * (q_outer @ stage_values)
        residual_norms = np.abs(residuals.astype(np.float64))

        if L.params.residual_type == 'full_abs':
            L.status.residual = float(np.max(residual_norms))
        elif L.params.residual_type == 'last_abs':
            L.status.residual = float(residual_norms[-1])
        elif L.params.residual_type == 'full_rel':
            L.status.residual = float(np.max(residual_norms) / abs(u0))
        elif L.params.residual_type == 'last_rel':
            L.status.residual = float(residual_norms[-1] / abs(u0))
        else:
            raise ValueError(
                f'residual_type = {L.params.residual_type} not implemented, choose '
                'full_abs, last_abs, full_rel or last_rel instead'
            )

        L.status.updated = False
        return None

    def compute_end_point(self):
        L = self.level
        P = L.prob

        if not self._is_scalar_linear_test_equation():
            if self._uses_outer_rhs_shortcuts():
                if self.coll.right_is_node and not self.params.do_coll_update:
                    L.uend = P.dtype_u(L.u[-1])
                    return None

                L.uend = P.dtype_u(L.u[0])
                for m in range(self.coll.num_nodes):
                    L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]
                if L.tau[-1] is not None:
                    L.uend += L.tau[-1]
                return None

            if self.coll.right_is_node and not self.params.do_coll_update:
                L.uend = P.dtype_u(L.u[-1])
                return None

            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                f_m = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
                L.uend += L.dt * self.coll.weights[m] * f_m
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]
            return None

        if self.coll.right_is_node and not self.params.do_coll_update:
            L.uend = P.dtype_u(L.u[-1])
            return None

        stage_values = self._get_stage_vector()
        z_outer = self.outer_float_precision.type(L.dt) * self.outer_float_precision.type(P.lam)
        weights = np.asarray(self.coll.weights, dtype=self.outer_float_precision)
        u0 = self.outer_float_precision.type(np.asarray(L.u[0]).reshape(-1)[0])

        if L.uend is None:
            L.uend = P.dtype_u(P.init, val=0.0)
        L.uend[:] = u0 + z_outer * np.dot(weights, stage_values)
        return None

    def update_nodes(self):
        L = self.level
        P = L.prob

        assert L.status.unlocked

        if not self._is_scalar_linear_test_equation():
            outer_residuals = self._compute_full_residuals()
            inner_data = self._get_inner_correction_data()
            self._initialize_inner_correction(inner_data, outer_residuals)
            dt_inner = self.inner_float_precision.type(L.dt)
            time_inner = self.inner_float_precision.type(L.time)

            self.last_inner_iterations = 0
            inner_residual = self._compute_inner_correction_residual(inner_data, dt_inner)
            while self.last_inner_iterations < self.params.inner_maxiter and inner_residual > self.params.inner_tol:
                self._perform_inner_correction_sweep(inner_data, dt_inner, time_inner)
                self.last_inner_iterations += 1
                inner_residual = self._compute_inner_correction_residual(inner_data, dt_inner)

            self.total_inner_iterations += self.last_inner_iterations

            for m in range(self.coll.num_nodes):
                if L.u[m + 1] is None:
                    L.u[m + 1] = P.dtype_u(P.init, val=0.0)
                L.u[m + 1][:] += np.asarray(inner_data.u[m + 1], dtype=L.u[m + 1].dtype)
                if self._uses_outer_rhs_shortcuts():
                    node_time = L.time + L.dt * self.coll.nodes[m]
                    if L.f[m + 1] is None:
                        L.f[m + 1] = P.eval_f(L.u[m + 1], node_time)
                    else:
                        L.f[m + 1][:] = P.eval_f(L.u[m + 1], node_time)

            L.status.updated = True
            return None

        stage_values = self._get_stage_vector()
        q_outer = np.asarray(self.coll.Qmat[1:, 1:], dtype=self.outer_float_precision)
        u0 = self.outer_float_precision.type(np.asarray(L.u[0]).reshape(-1)[0])
        z_outer = self.outer_float_precision.type(L.dt) * self.outer_float_precision.type(P.lam)
        rhs_outer = u0 - stage_values + z_outer * (q_outer @ stage_values)

        q_delta_inner = np.asarray(self.QI[1:, 1:], dtype=self.inner_float_precision)
        q_inner = np.asarray(self.coll.Qmat[1:, 1:], dtype=self.inner_float_precision)
        q_diff_inner = q_inner - q_delta_inner
        z_inner = self.inner_float_precision.type(L.dt) * self.inner_float_precision.type(P.lam)
        p_inner = np.eye(self.coll.num_nodes, dtype=self.inner_float_precision) - z_inner * q_delta_inner

        rhs_inner = rhs_outer.astype(self.inner_float_precision)

        correction_inner = np.zeros(self.coll.num_nodes, dtype=self.inner_float_precision)
        self.last_inner_iterations = 0

        for _ in range(self.params.inner_maxiter):
            inner_rhs = rhs_inner + z_inner * (q_diff_inner @ correction_inner)
            inner_residual = inner_rhs - p_inner @ correction_inner
            if float(np.max(np.abs(inner_residual.astype(np.float64)))) <= self.params.inner_tol:
                break

            correction_inner = np.asarray(np.linalg.solve(p_inner, inner_rhs), dtype=self.inner_float_precision)
            self.last_inner_iterations += 1

        self.total_inner_iterations += self.last_inner_iterations

        for m in range(self.coll.num_nodes):
            if L.u[m + 1] is None:
                L.u[m + 1] = P.dtype_u(P.init, val=0.0)
            L.u[m + 1][:] = stage_values[m] + self.outer_float_precision.type(correction_inner[m])

        L.status.updated = True
        return None
