import numpy as np

from pySDC.projects.Resilience.sweepers import generic_implicit_efficient


class generic_implicit_ir(generic_implicit_efficient):
    """
    Mixed-precision iterative-refinement SDC sweeper for the real scalar test equation.

    One call to ``update_nodes`` performs one outer iterative-refinement update on the
    collocation system. The inner correction equation is approximated in lower
    precision using the SDC preconditioner.
    """

    def __init__(self, params, level):
        params = dict(params)
        params.setdefault('inner_float_precision', np.dtype('float32'))
        params.setdefault('inner_maxiter', 5)
        params.setdefault('inner_tol', 1e-9)
        super().__init__(params, level)

        self.outer_float_precision = np.dtype(self.params.float_precision)
        self.inner_float_precision = np.dtype(self.params.inner_float_precision)
        self.last_inner_iterations = 0
        self.total_inner_iterations = 0

        self._q_outer = np.array(self.coll.Qmat[1:, 1:], dtype=self.outer_float_precision, copy=True)
        self._q_delta_inner = np.array(self.QI[1:, 1:], dtype=self.inner_float_precision, copy=True)
        self._q_inner = np.array(self.coll.Qmat[1:, 1:], dtype=self.inner_float_precision, copy=True)
        self._q_diff_inner = self._q_inner - self._q_delta_inner
        self._ones_outer = np.ones(self.coll.num_nodes, dtype=self.outer_float_precision)
        self._cached_step_key = None
        self._k_outer = None
        self._p_inner = None
        self._z_inner = None

    def _ensure_step_operators(self):
        P = self.level.prob
        dt_outer = self.outer_float_precision.type(self.level.dt)
        lam_outer = self.outer_float_precision.type(P.lam)
        key = (float(dt_outer), float(lam_outer))

        if key != self._cached_step_key:
            z_outer = dt_outer * lam_outer
            z_inner = self.inner_float_precision.type(self.level.dt) * self.inner_float_precision.type(P.lam)
            self._k_outer = np.eye(self.coll.num_nodes, dtype=self.outer_float_precision) - z_outer * self._q_outer
            self._p_inner = np.eye(self.coll.num_nodes, dtype=self.inner_float_precision) - z_inner * self._q_delta_inner
            self._z_inner = z_inner
            self._cached_step_key = key

    def reset_ir_stats(self):
        self.last_inner_iterations = 0
        self.total_inner_iterations = 0

    def _get_stage_vector(self):
        L = self.level
        values = np.zeros(self.coll.num_nodes, dtype=self.outer_float_precision)
        for m in range(self.coll.num_nodes):
            if L.u[m + 1] is not None:
                values[m] = self.outer_float_precision.type(np.asarray(L.u[m + 1]).reshape(-1)[0])
        return values

    def update_nodes(self):
        L = self.level
        P = L.prob

        assert L.status.unlocked
        self._ensure_step_operators()

        stage_values = self._get_stage_vector()
        u0 = self.outer_float_precision.type(np.asarray(L.u[0]).reshape(-1)[0])
        rhs_outer = u0 * self._ones_outer - self._k_outer @ stage_values
        rhs_inner = rhs_outer.astype(self.inner_float_precision)

        correction_inner = np.zeros(self.coll.num_nodes, dtype=self.inner_float_precision)
        self.last_inner_iterations = 0

        for _ in range(self.params.inner_maxiter):
            inner_residual = rhs_inner + self._z_inner * (self._q_diff_inner @ correction_inner) - self._p_inner @ correction_inner
            if float(np.max(np.abs(inner_residual.astype(np.float64)))) <= self.params.inner_tol:
                break

            inner_rhs = rhs_inner + self._z_inner * (self._q_diff_inner @ correction_inner)
            correction_inner = np.asarray(np.linalg.solve(self._p_inner, inner_rhs), dtype=self.inner_float_precision)
            self.last_inner_iterations += 1

        self.total_inner_iterations += self.last_inner_iterations
        updated_stage_values = stage_values + correction_inner.astype(self.outer_float_precision)

        for m in range(self.coll.num_nodes):
            if L.u[m + 1] is None:
                L.u[m + 1] = P.dtype_u(P.init, val=0.0)
            L.u[m + 1][:] = updated_stage_values[m]
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        L.status.updated = True
        return None
