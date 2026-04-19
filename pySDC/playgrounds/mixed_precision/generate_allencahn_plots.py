import sys

import numpy as np
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.playgrounds.mixed_precision.paths import ensure_output_dirs, presentation_path


COLORMAP = "magma"


def plot_allencahn():
    t_end = 0.01
    nvars = (64, 64)

    prob = allencahn_fullyimplicit(nvars=nvars, nu=2, eps=0.04)

    x = prob.xvalues
    X, Y = np.meshgrid(x, x)
    out_paths = []
    for time, label in ((0.0, "t0p0"), (t_end, "t0p01")):
        u = np.asarray(prob.u_exact(t=time))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, u, cmap=COLORMAP, linewidth=0, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.view_init(elev=30, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)

        out_path = presentation_path(f"allencahn_exact_{label}_magma.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def animate_allencahn():
    t_start = 0.0
    t_end = 0.035
    nvars = (64, 64)
    num_frames = 121
    fps = 30

    prob = allencahn_fullyimplicit(nvars=nvars, nu=2, eps=0.04)
    x = prob.xvalues
    X, Y = np.meshgrid(x, x)
    t_values = np.linspace(t_start, t_end, num_frames)
    u_frames = np.array([np.asarray(prob.u_exact(t=time)) for time in t_values])

    u_min = float(np.min(u_frames))
    u_max = float(np.max(u_frames))
    pad = 0.05 * (u_max - u_min) if u_max > u_min else 1.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = [
        ax.plot_surface(
            X,
            Y,
            u_frames[0],
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
    ax.view_init(elev=30, azim=-60)

    norm = Normalize(vmin=u_min, vmax=u_max)
    mappable = ScalarMappable(norm=norm, cmap=COLORMAP)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        surf[0].remove()
        surf[0] = ax.plot_surface(
            X,
            Y,
            u_frames[frame],
            cmap=COLORMAP,
            vmin=u_min,
            vmax=u_max,
            linewidth=0,
            antialiased=True,
        )
        time_text.set_text(f"t={t_values[frame]:.4f}")
        return surf[0], time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )
    out_path = presentation_path("allencahn_exact_t0p0_to_t0p01.mp4")
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ensure_output_dirs()

    paths = []
    paths.extend(plot_allencahn())
    paths.append(animate_allencahn())

    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
