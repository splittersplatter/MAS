"""Problem 2: distributed gradient-tracking estimate of a moving target."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent; ASSET = ROOT / "assets"; ASSET.mkdir(exist_ok=True)
SEED, N, P, TMAX = 20260905, 6, 0.55, 30
ALPHA, MAX_ITERS = 0.018, 7000


def graph_and_weights():
    for s in range(SEED, SEED + 100):
        rng = np.random.default_rng(s); upper = rng.random((N, N)) < P
        A = np.triu(upper, 1).astype(float); A += A.T
        seen = {0}; frontier = [0]
        while frontier:
            i = frontier.pop()
            for j in np.flatnonzero(A[i]):
                if int(j) not in seen: seen.add(int(j)); frontier.append(int(j))
        if len(seen) == N: break
    else: raise RuntimeError("No connected graph found.")
    deg = A.sum(axis=1).astype(int); W = np.zeros((N, N))
    edges = list(zip(*np.where(np.triu(A, 1) > 0)))
    for i, j in edges: W[i, j] = W[j, i] = 1 / (1 + max(deg[i], deg[j]))
    for i in range(N): W[i, i] = 1 - W[i].sum()
    return A, W, s


def grad(local, q, mean, inv0, invw, invv):
    g = np.zeros_like(local)
    g[:, 0] += (2 / N) * ((local[:, 0] - mean) @ inv0.T)
    g[:, 1:] += (2 / N) * ((local[:, 1:] - local[:, :-1]) @ invw.T)
    g[:, :-1] -= (2 / N) * ((local[:, 1:] - local[:, :-1]) @ invw.T)
    g += 2 * (local - q) @ invv.transpose(0, 2, 1)
    return g


def objective(local, q, mean, inv0, invw, invv):
    total = np.zeros(N)
    for i in range(N):
        val = np.sum((local[i, 0] - mean) @ inv0 * (local[i, 0] - mean)) / N
        d = local[i, 1:] - local[i, :-1]
        val += np.sum((d @ invw) * d) / N
        e = q[i] - local[i]
        val += np.sum((e @ invv[i]) * e)
        total[i] = val
    return total


def main():
    rng = np.random.default_rng(SEED); A, W, graph_seed = graph_and_weights(); edges = list(zip(*np.where(np.triu(A, 1) > 0)))
    mean = np.array([2.0, -1.0, 3.0]); S0 = np.diag([0.8, 0.8, 0.6])
    Sw = np.diag([0.12, 0.10, 0.16]); Sv = np.array([np.diag([0.22, 0.20, 0.28]) * (1 + .1*i) for i in range(N)])
    inv0, invw, invv = np.linalg.inv(S0), np.linalg.inv(Sw), np.linalg.inv(Sv)
    z = np.zeros((TMAX + 1, 3)); z[0] = rng.multivariate_normal(mean, S0)
    for t in range(TMAX): z[t + 1] = z[t] + rng.multivariate_normal(np.zeros(3), Sw)
    x = np.array([[0, 0, 0], [4, 0, 1], [0, 4, 2], [4, 4, 1], [2, 2, 5], [6, 2, 3]], dtype=float)
    v = np.array([rng.multivariate_normal(np.zeros(3), Sv[i // (TMAX + 1)]) for i in range(N * (TMAX + 1))]).reshape(N, TMAX + 1, 3)
    y = x[:, None, :] - z[None, :, :] + v
    q = x[:, None, :] - y

    local = np.tile(mean, (N, TMAX + 1, 1)); local += rng.normal(0, .05, local.shape)
    tracker = grad(local, q, mean, inv0, invw, invv); history = []; previous_obj = None
    for k in range(MAX_ITERS):
        a = ALPHA / (1 + 0.0004 * k)
        nxt = np.einsum("ij,jtd->itd", W, local) - a * tracker
        tracker = np.einsum("ij,jtd->itd", W, tracker) + grad(nxt, q, mean, inv0, invw, invv) - grad(local, q, mean, inv0, invw, invv)
        local = nxt
        disagreement = max((np.linalg.norm(local[i] - local[j]) for i, j in edges), default=0.0)
        obj = float(objective(local, q, mean, inv0, invw, invv).sum())
        history.append((disagreement, obj))
        if k > 100 and disagreement < 2e-4 and previous_obj is not None and abs(obj - previous_obj) < 1e-6: break
        previous_obj = obj
    estimate = local.mean(axis=0); err = estimate - z; rmse = float(np.sqrt(np.mean(err ** 2)))

    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(221, projection="3d")
    ax.plot(z[:, 0], z[:, 1], z[:, 2], color="#111827", lw=2, label="true target")
    ax.plot(estimate[:, 0], estimate[:, 1], estimate[:, 2], color="#dc2626", ls="--", label="estimate")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], color="#2563eb", s=35, label="fixed drones")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title("3D target tracking"); ax.legend(fontsize=7)
    ax2 = fig.add_subplot(222); ax2.plot(z, lw=1.5, label=["true x", "true y", "true z"]); ax2.plot(estimate, ls="--", lw=1.0, label=["estimate x", "estimate y", "estimate z"]); ax2.set_title("Coordinates"); ax2.set_xlabel("time"); ax2.grid(alpha=.2); ax2.legend(fontsize=7, ncol=2)
    ax3 = fig.add_subplot(223); ax3.plot(err, lw=1.0, label=["e_x", "e_y", "e_z"]); ax3.axhline(0, color="black", lw=.8); ax3.set_title(f"Estimation error (RMSE={rmse:.3f})"); ax3.set_xlabel("time"); ax3.set_ylabel("estimate - truth"); ax3.grid(alpha=.2); ax3.legend(fontsize=8)
    ax4 = fig.add_subplot(224); h = np.asarray(history); objective_change = np.abs(np.diff(h[:, 1], prepend=h[0, 1])); objective_change[0] = max(objective_change[0], 1e-16); ax4.semilogy(np.maximum(h[:, 0], 1e-16), label="edge disagreement"); ax4.semilogy(np.maximum(objective_change, 1e-16), label="objective change"); ax4.set_title("Convergence history"); ax4.set_xlabel("gradient-tracking iteration"); ax4.set_ylabel("value"); ax4.grid(alpha=.2); ax4.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ASSET / "p2_tracking_results.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2)); theta = np.linspace(0, 2*np.pi, N, endpoint=False); pos = np.c_[np.cos(theta), np.sin(theta)]
    for i,j in edges: ax.plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]],color="#64748b",lw=1)
    ax.scatter(pos[:,0],pos[:,1],s=500,c="#bfdbfe",edgecolors="#334155")
    for i in range(N): ax.text(pos[i,0],pos[i,1],str(i+1),ha="center",va="center")
    ax.set_title(f"Drone communication graph G({N}, {P})"); ax.axis("off"); fig.tight_layout(); fig.savefig(ASSET / "p2_graph.png", dpi=220); plt.close(fig)
    print(f"seed={SEED}, graph_seed={graph_seed}, N={N}, p={P}, TMAX={TMAX}, iterations={len(history)}")
    print(f"connected=True, alpha0={ALPHA}, final_disagreement={history[-1][0]:.6e}, RMSE={rmse:.6f}")
    print("W="); print(W)
    print("drone locations="); print(x)
    print("final estimate first/last="); print(estimate[0], estimate[-1])


if __name__ == "__main__": main()
