import weakref

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, gmres, lgmres, splu, spsolve, spilu

from pySDC.core.problem import Problem
from pySDC.implementations.problem_classes.generic_ND_FD import GenericNDimFinDiff
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


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


class _NewtonInnerCorrectionData:
    def __init__(self, sweeper):
        self.sweeper = sweeper
        self._uses_krylov_solver = sweeper.params.inner_solver in ('gmres', 'lgmres', 'fgmres')
        outer_problem = sweeper.level.prob
        if not hasattr(outer_problem, 'eval_jacobian'):
            raise TypeError(f'{type(outer_problem).__name__} does not implement eval_jacobian')

        self.outer_problem = outer_problem
        problem_params = sweeper._clone_problem_params(outer_problem, sweeper.inner_float_precision)
        self.problem = type(outer_problem)(**problem_params)
        self.qmat = np.asarray(sweeper.coll.Qmat, dtype=sweeper.inner_float_precision)
        self.qi = np.asarray(sweeper.inner_QI, dtype=sweeper.inner_float_precision)
        self.nodes = np.asarray(sweeper.coll.nodes, dtype=sweeper.inner_float_precision)
        self.num_nodes = sweeper.coll.num_nodes
        self.u = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes + 1)]
        self.f = [self.problem.dtype_f(self.problem.init, val=0.0) for _ in range(self.num_nodes + 1)]
        self.tau = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
        self.residuals = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
        self.integrals = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
        self.rhs = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
        self.jacobians = [None] * self.num_nodes
        self.linear_systems = [None] * self.num_nodes
        self.linear_system_solvers = [None] * self.num_nodes
        self._supports_problem_jacobian_solve = hasattr(self.problem, 'solve_system_jacobian')
        self.block_shape = self.u[1].shape
        self.block_size = int(np.prod(self.block_shape))
        self._sparse_identity = sp.eye(self.block_size, format='csc', dtype=self.problem.float_precision)

        if self._uses_krylov_solver:
            # Reuse Krylov work buffers across matvec and preconditioner calls.
            self._gmres_rhs = np.zeros(self.num_nodes * self.block_size, dtype=self.problem.float_precision)
            self._gmres_x0 = np.zeros(self.num_nodes * self.block_size, dtype=self.problem.float_precision)
            self._gmres_matvec_blocks_in = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
            self._gmres_matvec_blocks_out = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
            self._gmres_matvec_jacobian_products = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
            self._gmres_precond_blocks_rhs = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes)]
            self._gmres_precond_blocks_out = [self.problem.dtype_u(self.problem.init, val=0.0) for _ in range(self.num_nodes + 1)]
            self._gmres_matvec_out = np.zeros(self.num_nodes * self.block_size, dtype=self.problem.float_precision)
            self._gmres_precond_out = np.zeros(self.num_nodes * self.block_size, dtype=self.problem.float_precision)
        else:
            self._gmres_rhs = np.zeros(self.num_nodes * self.block_size, dtype=self.problem.float_precision)
            self._gmres_x0 = None
            self._gmres_matvec_blocks_in = None
            self._gmres_matvec_blocks_out = None
            self._gmres_matvec_jacobian_products = None
            self._gmres_precond_blocks_rhs = None
            self._gmres_precond_blocks_out = None
            self._gmres_matvec_out = None
            self._gmres_precond_out = None

    @staticmethod
    def _cast_jacobian(jacobian, dtype):
        if sp.issparse(jacobian):
            return jacobian.astype(dtype)
        return np.asarray(jacobian, dtype=dtype)

    def copy_vec_to_blocks(self, vec, blocks):
        for m in range(self.num_nodes):
            blocks[m][:] = vec[m * self.block_size : (m + 1) * self.block_size].reshape(self.block_shape)
        return blocks

    def copy_blocks_to_vec(self, blocks, out_vec):
        for m in range(self.num_nodes):
            out_vec[m * self.block_size : (m + 1) * self.block_size] = blocks[m].view(np.ndarray).reshape(-1)
        return out_vec

    def initialize(self, outer_stage_values, outer_residuals, dt_inner, time_inner):
        problem_dtype = self.problem.float_precision
        self.dt_inner = problem_dtype.type(dt_inner)
        self.time_inner = problem_dtype.type(time_inner)
        outer_dtype = self.outer_problem.float_precision
        outer_time = outer_dtype.type(time_inner)
        outer_dt = outer_dtype.type(dt_inner)

        for m in range(self.num_nodes + 1):
            self.u[m][:] = 0.0
            self.f[m][:] = 0.0

        for m, residual in enumerate(outer_residuals):
            self.tau[m][:] = np.asarray(residual, dtype=problem_dtype)

        for m, stage in enumerate(outer_stage_values):
            stage_outer = self.outer_problem.dtype_u(self.outer_problem.init, val=0.0)
            stage_outer[:] = np.asarray(stage, dtype=outer_dtype)
            node_time = outer_time + outer_dt * outer_dtype.type(self.nodes[m])
            jacobian = self._cast_jacobian(self.outer_problem.eval_jacobian(stage_outer, node_time), problem_dtype)
            self.jacobians[m] = jacobian

            factor = self.dt_inner * self.qi[m + 1, m + 1]
            if sp.issparse(jacobian):
                system = (self._sparse_identity - factor * jacobian).tocsc()
                self.linear_systems[m] = system
                self.linear_system_solvers[m] = self._build_sparse_solver(system, m)
            else:
                self.linear_systems[m] = None
                self.linear_system_solvers[m] = None

    def _build_sparse_solver(self, system, node_index):
        solver_type = getattr(self.sweeper.params, 'preconditioner_solver', 'cg')
        if solver_type == 'splu':
            return splu(system)
        if solver_type == 'spilu':
            drop_tol = getattr(self.sweeper.params, 'preconditioner_spilu_drop_tol', None)
            fill_factor = getattr(self.sweeper.params, 'preconditioner_spilu_fill_factor', None)
            kwargs = {}
            if drop_tol is not None:
                kwargs['drop_tol'] = float(drop_tol)
            if fill_factor is not None:
                kwargs['fill_factor'] = float(fill_factor)
            return spilu(system, **kwargs)
        return None

    def eval_jacobian_product_into(self, out, node_index, u):
        jacobian = self.jacobians[node_index]
        out_view = out.view(np.ndarray)
        u_vec = u.view(np.ndarray).reshape(-1)
        product = jacobian.dot(u_vec)
        out_view[:] = np.asarray(product, dtype=self.problem.float_precision).reshape(out.shape)
        return out

    def solve_system_into(self, out, node_index, rhs):
        rhs_vec = rhs.view(np.ndarray).reshape(-1)
        system = self.linear_systems[node_index]
        solver = self.linear_system_solvers[node_index]

        if solver is not None:
            out_view = out.view(np.ndarray)
            out_view[:] = np.asarray(solver.solve(rhs_vec), dtype=self.problem.float_precision).reshape(out.shape)
            return out

        if sp.issparse(system) and hasattr(self.problem, 'lin_tol') and hasattr(self.problem, 'lin_maxiter'):
            out_view = out.view(np.ndarray)
            rtol = self.sweeper.params.preconditioner_lin_tol
            if rtol is None:
                rtol = self.problem.lin_tol
            maxiter = self.sweeper.params.preconditioner_lin_maxiter
            if maxiter is None:
                maxiter = self.problem.lin_maxiter
            solution = cg(
                system,
                rhs_vec,
                x0=out_view.reshape(-1),
                rtol=rtol,
                maxiter=maxiter,
                atol=0.0,
            )[0]
            out_view[:] = np.asarray(solution, dtype=self.problem.float_precision).reshape(out.shape)
            return out

        factor = self.dt_inner * self.qi[node_index + 1, node_index + 1]

        if self._supports_problem_jacobian_solve:
            out[:] = self.problem.solve_system_jacobian(
                self.jacobians[node_index],
                rhs_vec,
                factor,
                u0=out,
                t=self.time_inner + self.dt_inner * self.nodes[node_index],
            )
            return out

        jacobian = np.asarray(self.jacobians[node_index], dtype=self.problem.float_precision)
        system = np.eye(jacobian.shape[0], dtype=self.problem.float_precision) - factor * jacobian
        out[:] = np.linalg.solve(system, rhs_vec).reshape(out.shape)
        return out

    def apply_operator(self, out_blocks, in_blocks, dt_inner):
        for j in range(self.num_nodes):
            self.eval_jacobian_product_into(self._gmres_matvec_jacobian_products[j], j, in_blocks[j])

        for m in range(self.num_nodes):
            out = out_blocks[m]
            out_view = out.view(np.ndarray)
            np.copyto(out_view, in_blocks[m].view(np.ndarray))
            for j in range(self.num_nodes):
                out_view -= (
                    dt_inner
                    * self.qmat[m + 1, j + 1]
                    * self._gmres_matvec_jacobian_products[j].view(np.ndarray)
                )
        return out_blocks

    def apply_preconditioner(self, out_blocks, rhs_blocks, dt_inner, sweeps=1):
        for m in range(self.num_nodes + 1):
            out_blocks[m][:] = 0.0
            self.f[m][:] = 0.0

        for _ in range(max(int(sweeps), 1)):
            for m in range(self.num_nodes):
                integral = self.integrals[m]
                integral_view = integral.view(np.ndarray)
                np.copyto(integral_view, rhs_blocks[m].view(np.ndarray))
                for j in range(1, self.num_nodes + 1):
                    integral_view += dt_inner * (self.qmat[m + 1, j] - self.qi[m + 1, j]) * self.f[j].view(np.ndarray)

            for m in range(self.num_nodes):
                rhs = self.rhs[m]
                rhs_view = rhs.view(np.ndarray)
                np.copyto(rhs_view, self.integrals[m].view(np.ndarray))
                for j in range(1, m + 1):
                    rhs_view += dt_inner * self.qi[m + 1, j] * self.f[j].view(np.ndarray)

                self.solve_system_into(out_blocks[m + 1], m, rhs)
                self.eval_jacobian_product_into(self.f[m + 1], m, out_blocks[m + 1])

        return out_blocks[1:]


class generic_implicit_ir(generic_implicit):
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
        params.setdefault('inner_tol_floor', None)
        params.setdefault('adaptive_inner', True)
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
        shared_rhs = None

        for m in range(1, self.coll.num_nodes + 1):
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                if shared_rhs is None:
                    shared_rhs = P.eval_f(L.u[0], L.time)
                L.f[m] = P.dtype_f(shared_rhs)
            elif self.params.initial_guess == 'copy':
                if L.f[0] is None:
                    L.f[0] = P.eval_f(L.u[0], L.time)
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.dtype_f(L.f[0])
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            elif self.params.initial_guess == 'random':
                node_time = L.time + L.dt * self.coll.nodes[m - 1]
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
        if hasattr(problem, 'float_precision'):
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

    def _get_inner_target_tol(self):
        target = float(self.params.inner_tol)
        if self.params.adaptive_inner and self.params.inner_tol_floor is not None:
            target = max(float(self.params.inner_tol_floor), target)
        return float(target)

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

    def _compute_last_residual(self):
        L = self.level
        P = L.prob

        res = P.dtype_u(P.init, val=0.0)
        for j in range(1, self.coll.num_nodes + 1):
            if L.u[j] is None:
                continue
            f_j = L.f[j] if L.f[j] is not None else P.eval_f(L.u[j], L.time + L.dt * self.coll.nodes[j - 1])
            res += L.dt * self.coll.Qmat[-1, j] * f_j

        res += L.u[0] - L.u[-1]
        if L.tau[-1] is not None:
            res += L.tau[-1]

        return res

    def compute_residual(self, stage=''):
        L = self.level
        P = L.prob

        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        if not self._is_scalar_linear_test_equation():
            if L.params.residual_type in ('last_abs', 'last_rel'):
                residual_norm = float(abs(self._compute_last_residual()))
                if L.params.residual_type == 'last_abs':
                    L.status.residual = residual_norm
                else:
                    L.status.residual = float(residual_norm / abs(L.u[0]))
            else:
                residuals = self._compute_full_residuals()
                residual_norms = np.array([abs(res) for res in residuals], dtype=np.float64)

                if L.params.residual_type == 'full_abs':
                    L.status.residual = float(np.max(residual_norms))
                elif L.params.residual_type == 'full_rel':
                    L.status.residual = float(np.max(residual_norms) / abs(L.u[0]))
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
            inner_target_tol = self._get_inner_target_tol()

            self.last_inner_iterations = 0
            inner_residual = self._compute_inner_correction_residual(inner_data, dt_inner)
            while self.last_inner_iterations < self.params.inner_maxiter and inner_residual > inner_target_tol:
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
        inner_target_tol = self._get_inner_target_tol()

        for _ in range(self.params.inner_maxiter):
            inner_rhs = rhs_inner + z_inner * (q_diff_inner @ correction_inner)
            inner_residual = inner_rhs - p_inner @ correction_inner
            if float(np.max(np.abs(inner_residual.astype(np.float64)))) <= inner_target_tol:
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


class generic_implicit_newton_sdc_ir(generic_implicit_ir):
    """
    Mixed-precision Newton-SDC iterative-refinement sweeper for nonlinear collocation problems.

    One call to ``update_nodes`` performs one outer Newton update on the collocation residual.
    The Newton correction equation is approximately solved by a small number of linearized SDC
    sweeps in lower precision, using Jacobians frozen at the current outer stage values.
    """

    def __init__(self, params, level):
        params = dict(params)
        params.setdefault('inner_float_precision', np.dtype('float32'))
        params.setdefault('inner_maxiter', 20)
        params.setdefault('inner_tol', 1e-9)
        params.setdefault('inner_tol_floor', None)
        params.setdefault('adaptive_inner', True)
        params.setdefault('inner_solver', 'gmres')
        params.setdefault('inner_QI', None)
        params.setdefault('gmres_restart', None)
        params.setdefault('gmres_maxiter', None)
        params.setdefault('gmres_preconditioner_sweeps', 1)
        params.setdefault('preconditioner_solver', 'cg')
        params.setdefault('preconditioner_lin_tol', None)
        params.setdefault('preconditioner_lin_maxiter', None)
        params.setdefault('preconditioner_spilu_drop_tol', None)
        params.setdefault('preconditioner_spilu_fill_factor', None)
        params.setdefault('gmres_warm_start', False)
        params.setdefault('cache_inner_step', True)
        super().__init__(params, level)
        self._newton_inner_data = None
        inner_qi_type = self.params.inner_QI if self.params.inner_QI is not None else self.params.QI
        self.inner_QI = self.get_Qdelta_implicit(inner_qi_type).astype(self.inner_float_precision, copy=False)

    def _get_newton_inner_data(self):
        if not self.params.cache_inner_step:
            return _NewtonInnerCorrectionData(self)

        if self._newton_inner_data is None:
            self._newton_inner_data = _NewtonInnerCorrectionData(self)

        return self._newton_inner_data

    def _get_outer_stage_values(self):
        L = self.level
        P = L.prob
        stage_values = []
        for m in range(self.coll.num_nodes):
            if L.u[m + 1] is None:
                stage_values.append(P.dtype_u(P.init, val=0.0))
            else:
                stage_values.append(P.dtype_u(L.u[m + 1]))
        return stage_values

    def _compute_newton_inner_residual(self, inner_data, dt_inner):
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

    def _perform_newton_inner_sweep(self, inner_data, dt_inner):
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
            rhs_view = rhs.view(np.ndarray)
            np.copyto(rhs_view, inner_data.integrals[m].view(np.ndarray))
            for j in range(1, m + 1):
                rhs_view += dt_inner * inner_data.qi[m + 1, j] * inner_data.f[j].view(np.ndarray)

            inner_data.solve_system_into(inner_data.u[m + 1], m, rhs)
            inner_data.eval_jacobian_product_into(inner_data.f[m + 1], m, inner_data.u[m + 1])

    def _build_newton_inner_linear_operators(self, inner_data, dt_inner, inner_target_tol):
        ncomp = inner_data.block_size
        size = inner_data.num_nodes * ncomp

        rhs = inner_data._gmres_rhs
        for m in range(inner_data.num_nodes):
            rhs[m * ncomp : (m + 1) * ncomp] = inner_data.tau[m].view(np.ndarray).reshape(-1)

        if not self.params.gmres_warm_start:
            inner_data._gmres_x0.fill(0.0)

        def matvec(vec):
            inner_data.copy_vec_to_blocks(vec, inner_data._gmres_matvec_blocks_in)
            inner_data.apply_operator(inner_data._gmres_matvec_blocks_out, inner_data._gmres_matvec_blocks_in, dt_inner)
            return inner_data.copy_blocks_to_vec(inner_data._gmres_matvec_blocks_out, inner_data._gmres_matvec_out)

        def precond(vec):
            inner_data.copy_vec_to_blocks(vec, inner_data._gmres_precond_blocks_rhs)
            correction_blocks = inner_data.apply_preconditioner(
                inner_data._gmres_precond_blocks_out,
                inner_data._gmres_precond_blocks_rhs,
                dt_inner,
                sweeps=self.params.gmres_preconditioner_sweeps,
            )
            return inner_data.copy_blocks_to_vec(correction_blocks, inner_data._gmres_precond_out)

        operator = LinearOperator((size, size), matvec=matvec, dtype=self.inner_float_precision)
        preconditioner = LinearOperator((size, size), matvec=precond, dtype=self.inner_float_precision)

        counter = {'iters': 0}

        def callback(_):
            counter['iters'] += 1

        return rhs, operator, preconditioner, counter, callback

    def _solve_newton_inner_gmres(self, inner_data, dt_inner, inner_target_tol):
        ncomp = inner_data.block_size
        rhs, operator, preconditioner, counter, callback = self._build_newton_inner_linear_operators(
            inner_data, dt_inner, inner_target_tol
        )

        solution, info = gmres(
            operator,
            rhs,
            x0=inner_data._gmres_x0,
            rtol=float(inner_target_tol),
            atol=0.0,
            restart=self.params.gmres_restart,
            maxiter=self.params.gmres_maxiter if self.params.gmres_maxiter is not None else self.params.inner_maxiter,
            M=preconditioner,
            callback=callback,
            callback_type='pr_norm',
        )

        self.last_inner_iterations = counter['iters']
        if info > 0:
            self.last_inner_iterations = max(self.last_inner_iterations, int(info))

        inner_data._gmres_x0[:] = np.asarray(solution, dtype=self.inner_float_precision)

        for m in range(inner_data.num_nodes):
            inner_data.u[m + 1][:] = solution[m * ncomp : (m + 1) * ncomp].reshape(inner_data.u[m + 1].shape)
            inner_data.eval_jacobian_product_into(inner_data.f[m + 1], m, inner_data.u[m + 1])

        residual = operator.matvec(solution) - rhs
        return float(np.max(np.abs(np.asarray(residual, dtype=np.float64))))

    def _solve_newton_inner_lgmres(self, inner_data, dt_inner, inner_target_tol):
        ncomp = inner_data.block_size
        rhs, operator, preconditioner, counter, callback = self._build_newton_inner_linear_operators(
            inner_data, dt_inner, inner_target_tol
        )

        solution, info = lgmres(
            operator,
            rhs,
            x0=inner_data._gmres_x0,
            rtol=float(inner_target_tol),
            atol=0.0,
            maxiter=self.params.gmres_maxiter if self.params.gmres_maxiter is not None else self.params.inner_maxiter,
            M=preconditioner,
            callback=callback,
            inner_m=self.params.gmres_restart if self.params.gmres_restart is not None else 30,
        )

        self.last_inner_iterations = counter['iters']
        if info > 0:
            self.last_inner_iterations = max(self.last_inner_iterations, int(info))

        inner_data._gmres_x0[:] = np.asarray(solution, dtype=self.inner_float_precision)

        for m in range(inner_data.num_nodes):
            inner_data.u[m + 1][:] = solution[m * ncomp : (m + 1) * ncomp].reshape(inner_data.u[m + 1].shape)
            inner_data.eval_jacobian_product_into(inner_data.f[m + 1], m, inner_data.u[m + 1])

        residual = operator.matvec(solution) - rhs
        return float(np.max(np.abs(np.asarray(residual, dtype=np.float64))))

    def _solve_newton_inner_direct(self, inner_data, dt_inner):
        ncomp = inner_data.block_size
        dtype = self.inner_float_precision

        rhs = inner_data._gmres_rhs
        for m in range(inner_data.num_nodes):
            rhs[m * ncomp : (m + 1) * ncomp] = inner_data.tau[m].view(np.ndarray).reshape(-1)

        sparse_jacobians = []
        for jacobian in inner_data.jacobians:
            if sp.issparse(jacobian):
                sparse_jacobians.append(jacobian.astype(dtype))
            else:
                sparse_jacobians.append(sp.csc_matrix(np.asarray(jacobian, dtype=dtype)))

        identity = sp.eye(ncomp, format='csc', dtype=dtype)
        blocks = []
        for m in range(inner_data.num_nodes):
            row = []
            for j in range(inner_data.num_nodes):
                block = -dt_inner * inner_data.qmat[m + 1, j + 1] * sparse_jacobians[j]
                if m == j:
                    block = identity + block
                row.append(block)
            blocks.append(row)
        system = sp.bmat(blocks, format='csc')
        solution = np.asarray(spsolve(system, rhs), dtype=dtype)

        self.last_inner_iterations = 1

        for m in range(inner_data.num_nodes):
            inner_data.u[m + 1][:] = solution[m * ncomp : (m + 1) * ncomp].reshape(inner_data.u[m + 1].shape)
            inner_data.eval_jacobian_product_into(inner_data.f[m + 1], m, inner_data.u[m + 1])

        return 0.0

    def _get_inner_target_tol(self, outer_residual):
        target = float(self.params.inner_tol)
        if self.params.adaptive_inner and self.params.inner_tol_floor is not None:
            target = max(float(self.params.inner_tol_floor), target)
        return float(target)

    def update_nodes(self):
        L = self.level
        P = L.prob

        assert L.status.unlocked

        outer_residuals = self._compute_full_residuals()
        outer_stage_values = self._get_outer_stage_values()

        inner_data = self._get_newton_inner_data()
        dt_inner = self.inner_float_precision.type(L.dt)
        time_inner = self.inner_float_precision.type(L.time)
        inner_data.initialize(outer_stage_values, outer_residuals, dt_inner, time_inner)

        self.last_inner_iterations = 0
        outer_residual_norm = float(max(abs(res) for res in outer_residuals))
        inner_target_tol = self._get_inner_target_tol(outer_residual_norm)
        inner_iteration_limit = int(self.params.inner_maxiter)

        if self.params.inner_solver == 'direct':
            inner_residual = self._solve_newton_inner_direct(inner_data, dt_inner)
        elif self.params.inner_solver == 'gmres':
            inner_residual = self._solve_newton_inner_gmres(inner_data, dt_inner, inner_target_tol)
        elif self.params.inner_solver in ('lgmres', 'fgmres'):
            inner_residual = self._solve_newton_inner_lgmres(inner_data, dt_inner, inner_target_tol)
        elif self.params.inner_solver == 'sdc':
            inner_residual = self._compute_newton_inner_residual(inner_data, dt_inner)
            while self.last_inner_iterations < inner_iteration_limit and inner_residual > inner_target_tol:
                self._perform_newton_inner_sweep(inner_data, dt_inner)
                self.last_inner_iterations += 1
                inner_residual = self._compute_newton_inner_residual(inner_data, dt_inner)
        else:
            raise ValueError(f'Unknown inner_solver {self.params.inner_solver}')

        correction = []
        for m in range(self.coll.num_nodes):
            corr_m = P.dtype_u(P.init, val=0.0)
            corr_m[:] = np.asarray(inner_data.u[m + 1], dtype=P.float_precision)
            correction.append(corr_m)

        self.total_inner_iterations += self.last_inner_iterations

        for m in range(self.coll.num_nodes):
            node_time = L.time + L.dt * self.coll.nodes[m]
            if L.u[m + 1] is None:
                L.u[m + 1] = P.dtype_u(P.init, val=0.0)
            L.u[m + 1][:] = outer_stage_values[m].view(np.ndarray) + correction[m].view(np.ndarray)
            L.f[m + 1] = P.eval_f(L.u[m + 1], node_time)

        L.status.updated = True
        return None


class generic_implicit_newton_sdc(generic_implicit_newton_sdc_ir):
    """
    Full-precision Newton-SDC sweeper for nonlinear collocation problems.

    This reuses the Newton-SDC implementation of
    ``generic_implicit_newton_sdc_ir`` but keeps the inner linearized correction
    solve in the same precision as the outer collocation state by default.
    """

    def __init__(self, params, level):
        params = dict(params)
        outer_float_precision = np.dtype(
            params.get(
                'float_precision',
                getattr(getattr(level, 'prob', None), 'float_precision', np.dtype('float64')),
            )
        )
        params['inner_float_precision'] = outer_float_precision
        super().__init__(params, level)
