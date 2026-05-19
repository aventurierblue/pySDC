import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pySDC.projects.ir.fisher_fp64_vs_newton_sdc_ir import (
    _run_method,
    make_newton_sdc_fp64_description,
    make_newton_sdc_ir_description,
)


def import_seaborn():
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError('This benchmark requires `seaborn` to create the heatmap.') from exc

    return sns


def make_description(
    method,
    dt,
    num_nodes,
    target_tol,
    maxiter,
    nvars,
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
):
    common = {
        'dt': dt,
        'num_nodes': num_nodes,
        'outer_tol': target_tol,
        'outer_maxiter': maxiter,
        'nvars': nvars,
        'nu': nu,
        'lambda0': lambda0,
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
    nu,
    lambda0,
    inner_solver='gmres',
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
            nu=nu,
            lambda0=lambda0,
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
    nu,
    lambda0,
    inner_solver='gmres',
    inner_qi='LU',
    cache_inner_step=False,
    repeats=2,
):
    speedups = np.full((len(nvars_values), len(num_nodes_values)), np.nan, dtype=np.float64)

    print('Fisher heatmap benchmark: Newton-SDC fp64 vs Newton-SDC IR fp64/32')
    print(
        f't0={t0}, t_end={t_end}, steps={steps}, num_nodes={list(num_nodes_values)}, '
        f'nvars={list(nvars_values)}, target_tol={target_tol}, maxiter={maxiter}, '
        f'nu={nu}, lambda0={lambda0}, inner_solver={inner_solver}, inner_qi={inner_qi}, repeats={repeats}'
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
                nu=nu,
                lambda0=lambda0,
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
                nu=nu,
                lambda0=lambda0,
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
    fig_height = max(4.8, 0.7 * len(nvars_values) + 2.6)
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
        ax.set_ylabel('FD discretization nvars')
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)

    ax.set_title('Fisher Newton-SDC IR runtime speedup')

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a seaborn speedup heatmap for Newton-SDC IR on the Fisher FD benchmark.'
    )
    parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
    parser.add_argument('--t-end', dest='t_end', type=float, default=1.0, help='Final time')
    parser.add_argument('--steps', type=int, default=32, help='Number of time steps')
    parser.add_argument(
        '--num-nodes',
        dest='num_nodes_values',
        nargs='+',
        type=int,
        default=(3, 4, 5, 6),
        help='Collocation node counts to benchmark',
    )
    parser.add_argument(
        '--nvars',
        nargs='+',
        type=int,
        default=(127, 255, 511, 1023, 2047),
        help='Fisher FD resolutions to benchmark',
    )
    parser.add_argument('--target-tol', dest='target_tol', type=float, default=1e-8, help='Target stopping tolerance')
    parser.add_argument('--maxiter', type=int, default=20, help='Maximum outer iterations per step')
    parser.add_argument('--repeats', type=int, default=2, help='Timing repeats per configuration')
    parser.add_argument('--nu', type=float, default=1.0, help='Fisher nonlinearity parameter')
    parser.add_argument('--lambda0', type=float, default=2.0, help='Fisher lambda0 parameter')
    parser.add_argument(
        '--inner-solver',
        dest='inner_solver',
        choices=('direct', 'gmres', 'lgmres', 'fgmres', 'sdc'),
        default='gmres',
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
    t_end=1.0,
    steps=32,
    num_nodes_values=(3, 4, 5, 6),
    nvars_values=(255, 511, 1023, 2047),
    target_tol=1e-8,
    maxiter=20,
    repeats=2,
    nu=1.0,
    lambda0=2.0,
    inner_solver='gmres',
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
        nu=nu,
        lambda0=lambda0,
        inner_solver=inner_solver,
        inner_qi=inner_qi,
        cache_inner_step=cache_inner_step,
        repeats=repeats,
    )

    if output_path is None:
        output_path = Path(__file__).resolve().parents[3] / 'images' / 'fisher_newton_sdc_ir_speedup_heatmap.png'

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
        nu=args.nu,
        lambda0=args.lambda0,
        inner_solver=args.inner_solver,
        inner_qi=args.inner_qi,
        cache_inner_step=args.cache_inner_step,
        output_path=args.output,
    )
