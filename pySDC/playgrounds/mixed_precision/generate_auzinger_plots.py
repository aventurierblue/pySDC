import sys

import numpy as np
import matplotlib

if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger
from pySDC.playgrounds.mixed_precision.paths import ensure_output_dirs, presentation_path


COLORMAP = "magma"


def plot_auzinger():
    t_end = 4 * np.pi
    dt = 0.0005

    prob = auzinger()
    num_steps = int(round(t_end / dt))
    t = np.linspace(0.0, t_end, num_steps + 1)
    u = np.array([np.asarray(prob.u_exact(time)) for time in t])
    colors = plt.get_cmap(COLORMAP)(np.linspace(0.2, 0.8, 2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, u[:, 0], label="y1", color=colors[0])
    ax.plot(t, u[:, 1], label="y2", color=colors[1])
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = presentation_path("auzinger_exact_t0p1_magma.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return [out_path]


def animate_auzinger():
    t_start = 0.0
    t_end = 2 * np.pi
    num_frames = 151
    fps = 30

    prob = auzinger()
    t_values = np.linspace(t_start, t_end, num_frames)
    u_frames = np.array([np.asarray(prob.u_exact(time)) for time in t_values])

    u_min = float(np.min(u_frames))
    u_max = float(np.max(u_frames))
    pad = 0.05 * (u_max - u_min) if u_max > u_min else 1.0

    colors = plt.get_cmap(COLORMAP)(np.linspace(0.2, 0.8, 2))
    fig, ax = plt.subplots(figsize=(7, 4))
    (line1,) = ax.plot([], [], label="y1", color=colors[0], animated=True)
    (line2,) = ax.plot([], [], label="y2", color=colors[1], animated=True)
    (marker1,) = ax.plot([], [], marker="o", color=colors[0], linestyle="None", animated=True)
    (marker2,) = ax.plot([], [], marker="o", color=colors[1], linestyle="None", animated=True)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, animated=True)
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(u_min - pad, u_max + pad)
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True, alpha=0.3)

    def update(frame):
        line1.set_data(t_values[: frame + 1], u_frames[: frame + 1, 0])
        line2.set_data(t_values[: frame + 1], u_frames[: frame + 1, 1])
        marker1.set_data([t_values[frame]], [u_frames[frame, 0]])
        marker2.set_data([t_values[frame]], [u_frames[frame, 1]])
        time_text.set_text(f"t={t_values[frame]:.3f}")
        return line1, line2, marker1, marker2, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=True,
    )
    out_path = presentation_path("auzinger_exact_t0p0_to_t12p57.mp4")
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ensure_output_dirs()

    paths = []
    paths.extend(plot_auzinger())
    paths.append(animate_auzinger())

    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
