"""Problem 1: reproducible formation-control animation for the name DHRUV."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "assets"
ASSET.mkdir(exist_ok=True)
SEED, N, P = 20260905, 20, 0.25
DT, TOL, MAX_STEPS = 0.12, 1e-3, 260


def connected_graph():
    for s in range(SEED, SEED + 200):
        rng = np.random.default_rng(s)
        upper = rng.random((N, N)) < P
        A = np.triu(upper, 1).astype(float); A += A.T
        seen = {0}; frontier = [0]
        while frontier:
            i = frontier.pop()
            for j in np.flatnonzero(A[i]):
                if int(j) not in seen: seen.add(int(j)); frontier.append(int(j))
        if len(seen) == N: return A, s
    raise RuntimeError("No connected graph found in the documented retries.")


def sample_polyline(points, count, include_end=True):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.r_[0.0, np.cumsum(lengths)]
    values = np.linspace(0.0, dist[-1], count, endpoint=include_end)
    out = []
    for d in values:
        j = min(np.searchsorted(dist, d, side="right") - 1, len(points) - 2)
        frac = 0.0 if lengths[j] == 0 else (d - dist[j]) / lengths[j]
        out.append(points[j] + frac * (points[j + 1] - points[j]))
    return np.asarray(out)


def letter_strokes(letter):
    h = 6.0
    if letter == "D":
        return [[(-2.4, -h / 2), (-2.4, h / 2), (-0.4, h / 2), (1.0, 2.2),
                 (1.0, -2.2), (-0.4, -h / 2), (-2.4, -h / 2)]]
    if letter == "H":
        return [[(-2.3, -h / 2), (-2.3, h / 2)], [(2.3, -h / 2), (2.3, h / 2)],
                [(-2.3, 0), (2.3, 0)]]
    if letter == "R":
        return [[(-2.3, -h / 2), (-2.3, h / 2), (-0.2, h / 2), (1.0, 2.1),
                 (-0.2, 0), (-2.3, 0)], [(-0.2, 0), (2.2, -h / 2)]]
    if letter == "U":
        return [[(-2.3, h / 2), (-2.3, -2.0), (-1.4, -h / 2), (1.4, -h / 2),
                 (2.3, -2.0), (2.3, h / 2)]]
    if letter == "V":
        return [[(-2.4, h / 2), (0, -h / 2), (2.4, h / 2)]]
    raise ValueError(letter)


def offsets(letter):
    strokes = letter_strokes(letter)
    lengths = [np.linalg.norm(np.diff(np.asarray(s), axis=0), axis=1).sum() for s in strokes]
    counts = np.maximum(2, np.floor(N * np.asarray(lengths) / sum(lengths)).astype(int))
    while counts.sum() < N:
        counts[np.argmax(lengths)] += 1
    while counts.sum() > N:
        j = int(np.argmax(counts))
        if counts[j] > 2:
            counts[j] -= 1
        else:
            break
    chunks = [sample_polyline(s, int(c), include_end=(k == len(strokes) - 1))
              for k, (s, c) in enumerate(zip(strokes, counts))]
    r = np.vstack(chunks)
    return r - r.mean(axis=0)


def main():
    A, graph_seed = connected_graph(); edges = list(zip(*np.where(np.triu(A, 1) > 0)))
    L = np.diag(A.sum(axis=1)) - A
    lam_max = np.linalg.eigvalsh(L).max()
    rng = np.random.default_rng(SEED)
    p = rng.uniform([-5.0, -4.0], [5.0, 4.0], size=(N, 2))
    letters = "DHRUV"
    R = {letter: offsets(letter) for letter in letters}
    frames, records = [p.copy()], []
    for letter in letters:
        r = R[letter]
        eps = float("inf")
        for k in range(MAX_STEPS):
            eps = max((np.linalg.norm((p[i] - p[j]) - (r[i] - r[j]))
                       for i, j in edges), default=0.0)
            if eps <= TOL:
                break
            p = p - DT * (L @ (p - r))
            frames.append(p.copy())
        records.append((letter, k + 1, eps))
        frames.append(p.copy())

    # Communication graph.
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False); pos = np.c_[np.cos(theta), np.sin(theta)]
    for i, j in edges: ax.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]], color="#64748b", lw=.8)
    ax.scatter(pos[:,0], pos[:,1], s=430, c="#bfdbfe", edgecolors="#334155")
    for i in range(N): ax.text(pos[i,0], pos[i,1], str(i+1), ha="center", va="center", fontsize=7)
    ax.set_title(f"Connected Erdos-Renyi graph G({N}, {P})")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(ASSET / "p1_graph.png", dpi=220); plt.close(fig)

    # Initial cloud and final V shape.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].scatter(frames[0][:, 0], frames[0][:, 1], c=np.arange(N), cmap="viridis")
    axes[0].set_title("Initial positions"); axes[0].set_aspect("equal")
    axes[1].scatter(p[:, 0], p[:, 1], c=np.arange(N), cmap="viridis")
    axes[1].set_title("Final formation: V"); axes[1].set_aspect("equal")
    for ax in axes:
        ax.set_xlim(-6, 6); ax.set_ylim(-5, 5); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(ASSET / "p1_initial_final.png", dpi=220); plt.close(fig)

    # A small frame strip for the report.
    chosen = np.linspace(0, len(frames) - 1, 6, dtype=int)
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
    for ax, idx in zip(axes, chosen):
        xy = frames[idx]; ax.scatter(xy[:, 0], xy[:, 1], s=12, c=np.arange(N), cmap="viridis")
        ax.set_xlim(-6, 6); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.set_title(f"k={idx}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(ASSET / "p1_frame_strip.png", dpi=220); plt.close(fig)

    # The full animation uses fixed axes and equal aspect ratio.
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    scat = ax.scatter([], [], c=[], cmap="viridis", s=30)
    ax.set_xlim(-6, 6); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(alpha=.2)
    def update(xy):
        scat.set_offsets(xy); return (scat,)
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=frames, interval=35, blit=True)
    anim.save(ASSET / "p1_DHRUV_formation.gif", writer=PillowWriter(fps=18)); plt.close(fig)

    print(f"seed={SEED}, graph_seed={graph_seed}, N={N}, p={P}, edges={len(edges)}")
    print(f"connected=True, lambda_max={lam_max:.6f}, dt={DT}, 2/lambda_max={2/lam_max:.6f}")
    print("letters=", letters, "initial_rectangle=[-5,5] x [-4,4]")
    for letter, steps, eps in records: print(f"{letter}: steps={steps}, final_edge_error={eps:.6e}")
    print("A="); print(A.astype(int))
    print("D="); print(np.diag(A.sum(axis=1)).astype(int))
    print("L="); print(L.astype(int))


if __name__ == "__main__":
    main()
