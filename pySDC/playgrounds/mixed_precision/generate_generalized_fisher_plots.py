import sys

import numpy as np
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher
from pySDC.playgrounds.mixed_precision.paths import ensure_output_dirs, presentation_path


COLORMAP = "magma"


def plot_generalized_fisher():
    t_end = 0.1
    nvars = 63

    prob = generalized_fisher(nvars=nvars, nu=1.0, lambda0=2.0)
    x = np.array([(i + 1 - (nvars + 1) / 2) * prob.dx for i in range(nvars)])
    y = np.linspace(0.0, 1.0, 2)
    X, Y = np.meshgrid(x, y)
    out_paths = []
    for time, label in ((0.0, "t0p0"), (t_end, "t0p1")):
        u = np.asarray(prob.u_exact(t=time))
        Z = np.tile(u, (len(y), 1))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=COLORMAP, linewidth=0, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.view_init(elev=25, azim=-70)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

        out_path = presentation_path(f"generalized_fisher_exact_{label}_magma.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def animate_generalized_fisher():
    t_start = 0.0
    t_end = 1.0
    nvars = 63
    num_frames = 151
    fps = 30

    prob = generalized_fisher(nvars=nvars, nu=1.0, lambda0=2.0)
    x = np.array([(i + 1 - (nvars + 1) / 2) * prob.dx for i in range(nvars)])
    y = np.linspace(0.0, 1.0, 8)
    X, Y = np.meshgrid(x, y)
    t_values = np.linspace(t_start, t_end, num_frames)
    u_frames = np.array([np.asarray(prob.u_exact(t=time)) for time in t_values])

    u_min = float(np.min(u_frames))
    u_max = float(np.max(u_frames))
    pad = 0.05 * (u_max - u_min) if u_max > u_min else 1.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    Z0 = np.tile(u_frames[0], (len(y), 1))
    surf = [
        ax.plot_surface(
            X,
            Y,
            Z0,
            cmap=COLORMAP,
            vmin=u_min,
            vmax=u_max,
            linewidth=0,
            antialiased=True,
        )
    ]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_zlim(u_min - pad, u_max + pad)
    ax.view_init(elev=25, azim=-70)

    norm = Normalize(vmin=u_min, vmax=u_max)
    mappable = ScalarMappable(norm=norm, cmap=COLORMAP)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        surf[0].remove()
        Z = np.tile(u_frames[frame], (len(y), 1))
        surf[0] = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=COLORMAP,
            vmin=u_min,
            vmax=u_max,
            linewidth=0,
            antialiased=True,
        )
        time_text.set_text(f"t={t_values[frame]:.3f}")
        return surf[0], time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )
    out_path = presentation_path("generalized_fisher_exact_t0p0_to_t1p0.mp4")
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ensure_output_dirs()

    paths = []
    paths.extend(plot_generalized_fisher())
    paths.append(animate_generalized_fisher())

    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
