"""Hit-level mu- capture search after the SiPM time correction (input from fill_capture2.py).

Adjacent-crystal hits are fitted as k * far(t) + A exp(-t/tau) on clean bins (residual spikes masked), with
tau fixed to 70/75/80 ns (mu- in CsI) and free. Far crystals (200-400 mm) carry no capture signal.
Usage: python plot_capture2.py <capture2.npz> <outdir> [run]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit

from plot_calo import C_DISK, C_FIT, INK2, errbar, stamp
from plot_capture_hits import NAMES, project, spike_mask

TAUS = (70.0, 75.0, 80.0)
SELS = [("all hits", {}, "k"), ("E < 10 MeV, 2 SiPMs, unclustered", dict(emax=10, ns=1, inc=0), C_FIT)]


def fit(a, f, c, w, tau=None):
    def chi2(k, A, t):
        mu = k * f + A * np.exp(-c / t)
        return np.sum(((a - mu) ** 2 / np.maximum(a + k * k * f, 1.0))[w])
    m = Minuit(chi2, k=a[w].sum() / f[w].sum(), A=0, t=tau or 75.0)
    m.errordef = 1
    if tau is None:
        m.limits["t"] = (20, 1000)
        m.limits["A"] = (0, None)
    else:
        m.fixed["t"] = True
    m.migrad()
    m.hesse()
    return m


if __name__ == "__main__":
    D = np.load(sys.argv[1])
    out = sys.argv[2]
    run = sys.argv[3] if len(sys.argv) > 3 else "125329"
    os.makedirs(out, exist_ok=True)
    H, bt, BE = D["hist"], D["B_DT"], D["B_E"]
    c = 0.5 * (bt[1:] + bt[:-1])
    npr = int(D["nprompt"])
    mask = spike_mask(project(H, BE, slice(0, 5)))
    fitw = (~mask) & (c > 40) & (c < 1200)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    ax = axs[0]
    idx = np.flatnonzero((c > -200) & (c < 1200))
    edges = bt[idx[0]:idx[-1] + 2]
    far = project(H, BE, 4)
    adj = project(H, BE, 1)
    late = (c > 600) & (c < 1200) & ~mask
    k_late = adj[late].sum() / far[late].sum()
    for kk, col in [(0, "k"), (1, C_DISK[0])]:
        ax.stairs(project(H, BE, kk)[idx], edges, color=col, lw=1, label=NAMES[kk])
    ax.stairs((k_late * far)[idx], edges, color=C_DISK[1], lw=1, ls="--", label=f"far × {k_late:.1f}")
    for i in idx[mask[idx]]:
        ax.axvspan(bt[i], bt[i + 1], color="#f2c4b3", alpha=0.5, lw=0)
    ax.set_yscale("log")
    ax.set_xlim(-200, 1200)
    ax.set_xlabel("Δt hit − prompt cluster [ns], SiPM times corrected")
    ax.set_ylabel("delayed hits per 10 ns")
    ax.legend(fontsize=9, loc="upper right", title="shaded: masked residual spikes", title_fontsize=8.5)
    stamp(ax, run, f"   |   {npr:,} prompt clusters > 15 MeV")

    ax = axs[1]
    norm = (c > 1200) & (c < 3000) & ~mask
    grp = np.arange(40, 1201, 40)
    lines = []
    for off, (lab, kw, col) in zip((-5, 5), SELS):
        a, f = project(H, BE, 1, **kw), project(H, BE, 4, **kw)
        kn = a[norm].sum() / f[norm].sum()
        xs, rs, es = [], [], []
        for lo, hi in zip(grp[:-1], grp[1:]):
            m = fitw & (c >= lo) & (c < hi)
            sa, sf = a[m].sum(), f[m].sum()
            if m.sum() < 2 or sa == 0 or sf == 0:
                continue
            xs.append(c[m].mean())
            rs.append(sa / (kn * sf))
            es.append(rs[-1] * np.sqrt(1 / sa + 1 / sf))
        errbar(ax, np.array(xs) + off, rs, es, color=col, ms=4, label=lab)
        for tau in TAUS:
            m = fit(a, f, c, fitw, tau)
            lines.append((lab, f"τ = {tau:.0f} ns", m))
        lines.append((lab, "τ free", fit(a, f, c, fitw)))
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xlabel("Δt [ns] (clean bins, grouped by 40 ns)")
    ax.set_ylabel("adjacent / far  (normalised at 1.2–3 µs)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Capture would push this above 1 at small Δt", fontsize=10, loc="left")

    ax = axs[2]
    ax.axis("off")
    txt = ["fit: adjacent = k·far + A·e^(−t/τ), clean bins 40–1200 ns", ""]
    for lab, tl, m in lines:
        A, eA, t = m.values["A"], m.errors["A"], m.values["t"]
        nf = A * t / 10.0
        sig = f"{A / eA:+5.1f}σ" if eA > 0 else "at limit"
        txt.append(f"{lab[:14]:14s} {tl:11s} A = {A:7.0f} ± {eA:5.0f}  ({sig})"
                   + (f"  τ = {t:.0f} ± {m.errors['t']:.0f} ns" if tl == "τ free" else "")
                   + f"   χ²/ndf {m.fval:.0f}/{fitw.sum() - 2 - (tl == 'τ free')}")
        txt.append(f"{'':14s} {'':11s} → {nf:,.0f} fast hits = {nf / npr:.2e} per prompt cluster")
    ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8.5, transform=ax.transAxes)
    fig.savefig(f"{out}/19_capture_hits_corrected.png")
    plt.close(fig)
    for t in txt:
        print(t)
