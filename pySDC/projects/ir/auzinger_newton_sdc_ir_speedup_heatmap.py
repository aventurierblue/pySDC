import argparse
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.core.problem import Problem
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.ir.auzinger_fp64_vs_newton_sdc_ir import _run_method
from pySDC.projects.ir.sweepers import generic_implicit_newton_sdc, generic_implicit_newton_sdc_ir


class auzinger_nvars(Problem):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars=1, newton_maxiter=100, newton_tol=1e-12, float_precision=np.dtype('float64')):
        float_precision = np.dtype(float_precision)
        nvars = int(nvars)
        if nvars < 1:
            raise ValueError(f'nvars must be positive, got {nvars}')

        super().__init__((2 * nvars, None, float_precision), float_precision=float_precision)
        self._makeAttributeAndRegister('nvars', 'newton_maxiter', 'newton_tol', 'float_precision', localVars=locals(), readOnly=True)
        self._identity_block = np.eye(2, dtype=float_precision)
        self._block_indices = np.arange(self.nvars, dtype=np.int32)
        self._block_indptr = np.arange(self.nvars + 1, dtype=np.int32)

    def _state_view(self, u):
        return np.asarray(u, dtype=self.float_precision).reshape(self.nvars, 2)

    def _rhs_jacobian_blocks(self, u):
        state = self._state_view(u)
        x1 = state[:, 0]
        x2 = state[:, 1]

        blocks = np.empty((self.nvars, 2, 2), dtype=self.float_precision)
        blocks[:, 0, 0] = 1 - 3 * x1**2 - x2**2
        blocks[:, 0, 1] = -1 - 2 * x1 * x2
        blocks[:, 1, 0] = 1 - 6 * x1 * x2
        blocks[:, 1, 1] = 3 - 3 * x1**2 - 9 * x2**2
        return blocks

    def _extract_jacobian_blocks(self, dfdu):
        if sp.issparse(dfdu):
            bsr = dfdu.tobsr(blocksize=(2, 2))
            blocks = np.zeros((self.nvars, 2, 2), dtype=self.float_precision)
            for row in range(self.nvars):
                for idx in range(bsr.indptr[row], bsr.indptr[row + 1]):
                    if bsr.indices[idx] == row:
                        blocks[row] = np.asarray(bsr.data[idx], dtype=self.float_precision)
                        break
            return blocks

        matrix = np.asarray(dfdu, dtype=self.float_precision)
        blocks = np.empty((self.nvars, 2, 2), dtype=self.float_precision)
        for row in range(self.nvars):
            block_slice = slice(2 * row, 2 * row + 2)
            blocks[row] = matrix[block_slice, block_slice]
        return blocks

    def _solve_block_systems(self, jacobian_blocks, rhs_blocks, factor):
        systems = self._identity_block[None, :, :] - self.float_precision.type(factor) * jacobian_blocks
        a = systems[:, 0, 0]
        b = systems[:, 0, 1]
        c = systems[:, 1, 0]
        d = systems[:, 1, 1]
        r0 = rhs_blocks[:, 0]
        r1 = rhs_blocks[:, 1]
        det = a * d - b * c

        solution = np.empty_like(rhs_blocks)
        solution[:, 0] = (d * r0 - b * r1) / det
        solution[:, 1] = (-c * r0 + a * r1) / det
        return solution

    def u_exact(self, t):
        me = self.dtype_u(self.init)
        me[0::2] = np.cos(t)
        me[1::2] = np.sin(t)
        return me

    def eval_f(self, u, t):
        state = self._state_view(u)
        x1 = state[:, 0]
        x2 = state[:, 1]
        radius_sq = x1**2 + x2**2

        f = self.dtype_f(self.init)
        f_view = f.view(np.ndarray).reshape(self.nvars, 2)
        f_view[:, 0] = -x2 + x1 * (1 - radius_sq)
        f_view[:, 1] = x1 + 3 * x2 * (1 - radius_sq)
        return f

    def eval_jacobian(self, u, t=None):
        return sp.bsr_matrix(
            (self._rhs_jacobian_blocks(u), self._block_indices, self._block_indptr),
            shape=(2 * self.nvars, 2 * self.nvars),
        )

    def solve_system_jacobian(self, dfdu, rhs, factor, u0, t):
        rhs_blocks = np.asarray(rhs, dtype=self.float_precision).reshape(self.nvars, 2)
        solution = self._solve_block_systems(self._extract_jacobian_blocks(dfdu), rhs_blocks, factor)
        me = self.dtype_u(self.init)
        me[:] = solution.reshape(-1)
        return me

    def solve_jacobian(self, rhs, dt, u, t=0.0, **kwargs):
        return self.solve_system_jacobian(self.eval_jacobian(u, t), rhs, dt, u0=u, t=t)

    def solve_system(self, rhs, dt, u0, t):
        u = self.dtype_u(u0)
        u_view = u.view(np.ndarray).reshape(self.nvars, 2)
        rhs_view = np.asarray(rhs, dtype=self.float_precision).reshape(self.nvars, 2)
        dt = self.float_precision.type(dt)

        for _ in range(self.newton_maxiter):
            x1 = u_view[:, 0]
            x2 = u_view[:, 1]
            radius_sq = x1**2 + x2**2

            residual = np.empty_like(u_view)
            residual[:, 0] = x1 - dt * (-x2 + x1 * (1 - radius_sq)) - rhs_view[:, 0]
            residual[:, 1] = x2 - dt * (x1 + 3 * x2 * (1 - radius_sq)) - rhs_view[:, 1]

            if float(np.max(np.abs(residual))) < self.newton_tol:
                break

            correction = self._solve_block_systems(self._rhs_jacobian_blocks(u), residual, dt)
            u_view -= correction

        return u


def import_seaborn():
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError('This benchmark requires `seaborn` to create the heatmap.') from exc

    return sns


def make_newton_sdc_ir_description(
    dt,
    num_nodes,
    outer_tol,
    outer_maxiter,
    nvars,
    inner_solver='direct',
    inner_qi='LU',
    cache_inner_step=False,
):
    return {
        'problem_class': auzinger_nvars,
        'problem_params': {
            'nvars': nvars,
            'newton_maxiter': 20,
            'newton_tol': 1e-14,
            'float_precision': np.dtype('float64'),
        },
        'sweeper_class': generic_implicit_newton_sdc_ir,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'inner_QI': inner_qi,
            'initial_guess': 'spread',
            'float_precision': np.dtype('float64'),
            'inner_float_precision': np.dtype('float32'),
            'inner_solver': inner_solver,
            'adaptive_inner': False,
            'inner_tol_floor': None,
            'inner_maxiter': 20,
            'gmres_maxiter': 20,
            'gmres_restart': None,
            'inner_tol': 1e-8,
            'cache_inner_step': cache_inner_step,
        },
        'level_params': {
            'restol': outer_tol,
            'dt': np.float64(dt),
            'residual_type': 'full_abs',
        },
        'step_params': {
            'maxiter': outer_maxiter,
        },
    }


def make_newton_sdc_fp64_description(
    dt,
    num_nodes,
    outer_tol,
    outer_maxiter,
    nvars,
    inner_solver='direct',
    inner_qi='LU',
    cache_inner_step=False,
):
    description = make_newton_sdc_ir_description(
        dt=dt,
        num_nodes=num_nodes,
        outer_tol=outer_tol,
        outer_maxiter=outer_maxiter,
        nvars=nvars,
        inner_solver=inner_solver,
        inner_qi=inner_qi,
        cache_inner_step=cache_inner_step,
    )
    description['sweeper_class'] = generic_implicit_newton_sdc
    description['sweeper_params'] = dict(description['sweeper_params'])
    description['sweeper_params']['inner_float_precision'] = np.dtype('float64')
    return description


def make_description(
    method,
    dt,
    num_nodes,
    target_tol,
    maxiter,
    nvars,
    inner_solver='direct',
    inner_qi='LU',
    cache_inner_step=False,
):
    common = {
        'dt': dt,
        'num_nodes': num_nodes,
        'outer_tol': target_tol,
        'outer_maxiter': maxiter,
        'nvars': nvars,
        'inner_solver': inner_solver,
        'inner_qi': inner_qi,
        'cache_inner_step': cache_inner_step,
    }

    if method == 'newton-sdc-fp64':
        return make_newton_sdc_fp64_description(**common)
    if method == 'newton-sdc-ir':
        return make_newton_sdc_ir_description(**common)

    raise ValueError(f'Unknown method {method}')


def benchmark_method(
    method,
    t0,
    t_end,
    steps,
    num_nodes,
    target_tol,
    maxiter,
    nvars,
    inner_solver='direct',
    inner_qi='LU',
    cache_inner_step=False,
    repeats=2,
):
    dt = (t_end - t0) / steps
    best = None

    for _ in range(repeats):
        description = make_description(
            method=method,
            dt=dt,
            num_nodes=num_nodes,
            target_tol=target_tol,
            maxiter=maxiter,
            nvars=nvars,
            inner_solver=inner_solver,
            inner_qi=inner_qi,
            cache_inner_step=cache_inner_step,
        )
        result = _run_method(description, t0, t_end, steps, method)

        if best is None:
            best = result
            continue

        if result['converged'] and not best['converged']:
            best = result
            continue

        if result['converged'] == best['converged'] and result['elapsed'] < best['elapsed']:
            best = result

    return best


def benchmark_speedup_grid(
    t0,
    t_end,
    steps,
    num_nodes_values,
    nvars_values,
    target_tol,
    maxiter,
    inner_solver='direct',
    inner_qi='LU',
    cache_inner_step=False,
    repeats=2,
):
    speedups = np.full((len(nvars_values), len(num_nodes_values)), np.nan, dtype=np.float64)

    print('Auzinger heatmap benchmark: Newton-SDC fp64 vs Newton-SDC IR fp64/32')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={list(num_nodes_values)}, '
        f'nvars={list(nvars_values)}, target_tol={target_tol}, maxiter={maxiter}, '
        f'inner_solver={inner_solver}, inner_qi={inner_qi}, repeats={repeats}'
    )
    print('nvars    num_nodes   fp64_time[s]    ir_time[s]      speedup   status')

    for row, nvars in enumerate(nvars_values):
        for col, num_nodes in enumerate(num_nodes_values):
            fp64 = benchmark_method(
                method='newton-sdc-fp64',
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                target_tol=target_tol,
                maxiter=maxiter,
                nvars=nvars,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
                repeats=repeats,
            )
            ir = benchmark_method(
                method='newton-sdc-ir',
                t0=t0,
                t_end=t_end,
                steps=steps,
                num_nodes=num_nodes,
                target_tol=target_tol,
                maxiter=maxiter,
                nvars=nvars,
                inner_solver=inner_solver,
                inner_qi=inner_qi,
                cache_inner_step=cache_inner_step,
                repeats=repeats,
            )

            converged = fp64['converged'] and ir['converged']
            speedup = fp64['elapsed'] / ir['elapsed'] if converged and ir['elapsed'] > 0.0 else np.nan
            speedups[row, col] = speedup

            speedup_text = f'{speedup:.3f}x' if np.isfinite(speedup) else 'n/a'
            print(
                f'{nvars:>5d}   {num_nodes:>9d}   '
                f'{fp64["elapsed"]:>12.6e}   {ir["elapsed"]:>10.6e}   '
                f'{speedup_text:>9}   '
                f'{"ok" if converged else "stalled"}'
            )

    return speedups


def create_speedup_heatmap(
    sns,
    output_path,
    speedups,
    num_nodes_values,
    nvars_values,
    t0,
    t_end,
    steps,
    target_tol,
    inner_solver,
    inner_qi,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style='white')
    fig_width = max(6.4, 1.1 * len(num_nodes_values) + 2.8)
    fig_height = max(4.8, 0.7 * len(nvars_values) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    finite = speedups[np.isfinite(speedups)]
    if finite.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'No converged benchmark configurations', ha='center', va='center', fontsize=12)
    else:
        annot = np.full(speedups.shape, '', dtype=object)
        for row, col in np.ndindex(speedups.shape):
            if np.isfinite(speedups[row, col]):
                annot[row, col] = f'{speedups[row, col]:.2f}'

        vmin = min(float(finite.min()), 1.0)
        vmax = max(float(finite.max()), 1.0)
        if np.isclose(vmin, vmax):
            delta = max(0.05, 0.05 * abs(vmax) if vmax else 0.05)
            vmin -= delta
            vmax += delta

        sns.heatmap(
            speedups,
            mask=~np.isfinite(speedups),
            annot=annot,
            fmt='',
            cmap='vlag',
            center=1.0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor='white',
            xticklabels=num_nodes_values,
            yticklabels=nvars_values,
            cbar_kws={'label': 'Speedup = fp64 time / IR time [x]'},
            ax=ax,
        )
        ax.set_xlabel('num_nodes')
        ax.set_ylabel('nvars (replicated Auzinger blocks)')
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)

    ax.set_title('Auzinger Newton-SDC IR runtime speedup')
    fig.text(
        0.5,
        0.0,
        f'steps={steps}, t in [{t0:g}, {t_end:g}], target_tol={target_tol:g}, '
        f'inner_solver={inner_solver}, inner_qi={inner_qi}',
        ha='center',
        va='bottom',
        fontsize=9,
    )

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a seaborn speedup heatmap for Newton-SDC IR on an nvars-scaled Auzinger benchmark.'
    )
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=4.0 * np.pi, help='Final time')
    parser.add_argument('--steps', type=int, default=2000, help='Number of time steps')
    parser.add_argument(
        '--num-nodes',
        dest='num_nodes_values',
        nargs='+',
        type=int,
        default=(3, 4, 5, 6, 7),
        help='Collocation node counts to benchmark',
    )
    parser.add_argument(
        '--nvars',
        nargs='+',
        type=int,
        default=(31, 63, 127, 255, 511),
        help='Replicated Auzinger state sizes to benchmark',
    )
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-10, help='Target stopping tolerance')
    parser.add_argument('--maxiter', type=int, default=20, help='Maximum outer iterations per step')
    parser.add_argument('--repeats', type=int, default=2, help='Timing repeats per configuration')
    parser.add_argument(
        '--inner-solver',
        dest='inner_solver',
        choices=('direct', 'gmres', 'lgmres', 'fgmres', 'sdc'),
        default='direct',
        help='Inner Newton-SDC solver',
    )
    parser.add_argument('--inner-qi', dest='inner_qi', default='LU', help='Inner Newton-SDC preconditioner')
    parser.add_argument(
        '--cache-inner-step',
        dest='cache_inner_step',
        action='store_true',
        help='Reuse inner Newton-SDC work buffers',
    )
    parser.add_argument('--output', type=Path, default=None, help='Output path for the heatmap image')
    return parser.parse_args()


def main(
    t0=0.0,
    t_end=4.0 * np.pi,
    steps=2000,
    num_nodes_values=(3, 4, 5, 6, 7),
    nvars_values=(31, 63, 127, 255, 511),
    target_tol=1e-10,
    maxiter=20,
    repeats=2,
    inner_solver='direct',
    inner_qi='LU',
    cache_inner_step=False,
    output_path=None,
):
    sns = import_seaborn()
    num_nodes_values = tuple(dict.fromkeys(num_nodes_values))
    nvars_values = tuple(dict.fromkeys(nvars_values))

    speedups = benchmark_speedup_grid(
        t0=t0,
        t_end=t_end,
        steps=steps,
        num_nodes_values=num_nodes_values,
        nvars_values=nvars_values,
        target_tol=target_tol,
        maxiter=maxiter,
        inner_solver=inner_solver,
        inner_qi=inner_qi,
        cache_inner_step=cache_inner_step,
        repeats=repeats,
    )

    if output_path is None:
        output_path = Path(__file__).resolve().parents[3] / 'images' / 'auzinger_newton_sdc_ir_speedup_heatmap.png'

    create_speedup_heatmap(
        sns=sns,
        output_path=output_path,
        speedups=speedups,
        num_nodes_values=num_nodes_values,
        nvars_values=nvars_values,
        t0=t0,
        t_end=t_end,
        steps=steps,
        target_tol=target_tol,
        inner_solver=inner_solver,
        inner_qi=inner_qi,
    )
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    args = parse_args()
    main(
        t0=args.t0,
        t_end=args.t_end,
        steps=args.steps,
        num_nodes_values=args.num_nodes_values,
        nvars_values=args.nvars,
        target_tol=args.target_tol,
        maxiter=args.maxiter,
        repeats=args.repeats,
        inner_solver=args.inner_solver,
        inner_qi=args.inner_qi,
        cache_inner_step=args.cache_inner_step,
        output_path=args.output,
    )
