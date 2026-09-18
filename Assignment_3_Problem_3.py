"""Problem 3: consensus-constrained transportation ADMM reference solver."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent; ASSET = ROOT / "assets"; ASSET.mkdir(exist_ok=True)
SEED, N, M, P, RHO = 20260905, 4, 7, 0.70, 1.0
CAPACITY = np.array([2, 2, 2, 1], dtype=float); TOL = 1e-5; MAX_ITERS = 12000


def graph():
    for s in range(SEED, SEED + 100):
        rng = np.random.default_rng(s); upper = rng.random((N, N)) < P
        A = np.triu(upper, 1).astype(float); A += A.T
        seen = {0}; frontier = [0]
        while frontier:
            i = frontier.pop()
            for j in np.flatnonzero(A[i]):
                if int(j) not in seen: seen.add(int(j)); frontier.append(int(j))
        if len(seen) == N: return A, s
    raise RuntimeError("No connected graph found.")


def project_capped_simplex(v, total):
    lo, hi = float(np.min(v) - 1), float(np.max(v) + 1)
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.clip(v - mid, 0, 1).sum() > total: lo = mid
        else: hi = mid
    return np.clip(v - (lo + hi) / 2, 0, 1)


def solve(C):
    X = np.vstack([project_capped_simplex(-C[i] / RHO, CAPACITY[i]) for i in range(N)])
    Z = np.vstack([project_capped_simplex(X[:, j], 1) for j in range(M)]).T
    U = np.zeros((N, M)); previous_Z = Z.copy(); history = []
    for k in range(1, MAX_ITERS + 1):
        X = np.vstack([project_capped_simplex(Z[i] - U[i] - C[i] / RHO, CAPACITY[i]) for i in range(N)])
        previous_Z = Z.copy(); Z = np.vstack([project_capped_simplex(X[:, j] + U[:, j], 1) for j in range(M)]).T
        U += X - Z
        primal = np.linalg.norm(X - Z); dual = RHO * np.linalg.norm(Z - previous_Z)
        history.append((primal, dual))
        if primal < TOL and dual < TOL: break
    binary = (Z > 0.5).astype(int)
    # The transportation relaxation is integral here; this check is the only post-processing.
    integral = bool(np.max(np.minimum(np.abs(Z), np.abs(Z - 1))) < 2e-4)
    cost = float(np.sum(C * binary))
    feasible = bool(np.all(binary.sum(axis=0) == 1) and np.all(binary.sum(axis=1) == CAPACITY))
    return Z, binary, k, primal, dual, integral, feasible, cost, history


def main():
    rng = np.random.default_rng(SEED); A, graph_seed = graph(); edges = list(zip(*np.where(np.triu(A, 1) > 0)))
    costs = [rng.uniform(0.5, 9.5, (N, M)) for _ in range(3)]
    results = [solve(C) for C in costs]
    owners = [r[1].argmax(axis=0) for r in results]
    changed = [0] + [int(np.count_nonzero(owners[i] != owners[0])) for i in range(1, 3)]

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2))
    theta = np.linspace(0, 2*np.pi, N, endpoint=False); pos = np.c_[np.cos(theta), np.sin(theta)]
    for i,j in edges: axes[0,0].plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]],color="#64748b",lw=1)
    axes[0,0].scatter(pos[:,0],pos[:,1],s=450,c="#bfdbfe",edgecolors="#334155")
    for i in range(N): axes[0,0].text(pos[i,0],pos[i,1],str(i+1),ha="center",va="center")
    axes[0, 0].set_title("Connected communication graph"); axes[0, 0].axis("off")
    for j, (C, result) in enumerate(zip(costs, results)):
        im = axes[0, j + 1].imshow(C, cmap="YlOrRd", aspect="auto"); axes[0, j + 1].set_title(f"Cost matrix {j+1}"); axes[0, j + 1].set_xlabel("task"); axes[0, j + 1].set_ylabel("agent")
        axes[0, j + 1].set_xticks(range(M)); axes[0, j + 1].set_yticks(range(N)); fig.colorbar(im, ax=axes[0, j + 1], fraction=.046)
        axes[1, j].imshow(result[1], cmap="Blues", vmin=0, vmax=1, aspect="auto"); axes[1, j].set_title(f"Allocation {j+1}, cost={result[7]:.2f}"); axes[1, j].set_xlabel("task"); axes[1, j].set_ylabel("agent"); axes[1, j].set_xticks(range(M)); axes[1, j].set_yticks(range(N))
    axes[1, 3].axis("off")
    fig.tight_layout(); fig.savefig(ASSET / "p3_allocation_results.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for idx, result in enumerate(results, 1):
        hist = np.asarray(result[8])
        axes[0].semilogy(np.maximum(hist[:, 0], 1e-16), label=f"case {idx}")
        axes[1].semilogy(np.maximum(hist[:, 1], 1e-16), label=f"case {idx}")
    axes[0].set_title("Primal residual"); axes[1].set_title("Dual residual")
    for ax in axes:
        ax.set_xlabel("ADMM iteration"); ax.set_ylabel("residual norm"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ASSET / "p3_residuals.png", dpi=220); plt.close(fig)

    print(f"seed={SEED}, graph_seed={graph_seed}, N={N}, M={M}, p={P}, rho={RHO}, capacities={CAPACITY.astype(int).tolist()}")
    print(f"connected=True, edges={[(i+1,j+1) for i,j in edges]}")
    for idx, (C, result) in enumerate(zip(costs, results), 1):
        Z, binary, iters, primal, dual, integral, feasible, cost, history = result
        print(f"case {idx}: iterations={iters}, primal={primal:.3e}, dual={dual:.3e}, integral={integral}, feasible={feasible}, cost={cost:.6f}, changed={changed[idx-1]}")
        print("row sums=", binary.sum(axis=1), "column sums=", binary.sum(axis=0))
        print("C="); print(np.round(C, 4)); print("X="); print(binary)


if __name__ == "__main__": main()
