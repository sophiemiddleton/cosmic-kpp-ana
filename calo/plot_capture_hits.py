"""Plots for the hit-level mu- capture search (input from fill_capture.py).

Usage: python plot_capture_hits.py <capture.npz> <outdir> [run]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit

from plot_calo import C_DISK, C_FIT, INK2, errbar, stamp

TAU_CAP = 75.0  # ns, mu- lifetime in CsI (~70-80 ns)
NAMES = ["same crystal", "adjacent (≤ 50 mm)", "2nd ring (50–100 mm)", "gap", "far (200–400 mm)"]


def project(H, BE, dcls, emax=None, ns=None, inc=None):
    h = H[:, dcls]
    if h.ndim == 5:
        h = h.sum(1)
    if emax is not None:
        h = h[:, :np.searchsorted(BE, emax)]
    h = h.sum(1)
    h = h[:, ns] if ns is not None else h.sum(1)
    h = h[:, inc] if inc is not None else h.sum(-1)
    return h.astype(float)


def spike_mask(tot):
    med = np.array([np.median(tot[max(0, i - 15):i + 15]) for i in range(len(tot))])
    sp = tot > 1.5 * med
    m = sp.copy()
    for s in (-2, -1, 1, 2):
        m |= np.roll(sp, s)
    return m


def template_fit(a, f, c, w):
    """near(t) = k * far(t) + A exp(-t/TAU_CAP) on clean bins w."""
    def chi2(k, A):
        mu = k * f + A * np.exp(-c / TAU_CAP)
        return np.sum(((a - mu) ** 2 / (a + k * k * f + 1e-9))[w])
    m = Minuit(chi2, k=a[w].sum() / f[w].sum(), A=0)
    m.errordef = 1
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
    tot = project(H, BE, slice(0, 5))
    mask = spike_mask(tot)
    fitw = (~mask) & (c > 40) & (c < 1200)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")

    # (a) dt per distance class, far scaled to adjacent at late times
    ax = axs[0]
    idx = np.flatnonzero((c > -200) & (c < 1200))
    edges = bt[idx[0]:idx[-1] + 2]
    late = (c > 600) & (c < 1200) & ~mask
    far = project(H, BE, 4)
    for k, col in [(0, "k"), (1, C_DISK[0])]:
        a = project(H, BE, k)
        ax.stairs(a[idx], edges, color=col, lw=1, label=NAMES[k])
    adj = project(H, BE, 1)
    k_late = adj[late].sum() / far[late].sum()
    ax.stairs((k_late * far)[idx], edges, color=C_DISK[1], lw=1, ls="--", label=f"far × {k_late:.1f} (matched at 0.6–1.2 µs)")
    ax.set_yscale("log")
    ax.set_xlim(-200, 1200)
    ax.set_xlabel("Δt hit − prompt cluster [ns]")
    ax.set_ylabel("delayed hits per 10 ns")
    ax.legend(fontsize=9, loc="upper right")
    stamp(ax, run, f"   |   {int(D['nprompt']):,} prompt clusters > 15 MeV")

    # (b) adjacent/far ratio, one point per clean gap between echo spikes, far normalised at 1.2-3 us
    ax = axs[1]
    rows = []
    norm = (c > 1200) & (c < 3000) & ~mask
    gap = np.cumsum(np.r_[1, np.diff(fitw.astype(int)) == 1]) * fitw  # label contiguous clean runs
    for off, (lab, kw, col) in zip((-6, 6), [("all hits", {}, "k"),
                                              ("E < 10 MeV, 2 SiPMs, unclustered", dict(emax=10, ns=1, inc=0), C_FIT)]):
        a, f = project(H, BE, 1, **kw), project(H, BE, 4, **kw)
        rows.append((lab, template_fit(a, f, c, fitw)))
        kn = a[norm].sum() / f[norm].sum()
        xs, rs, es = [], [], []
        for g in np.unique(gap[gap > 0]):
            m = gap == g
            sa, sf = a[m].sum(), f[m].sum()
            xs.append(c[m].mean())
            rs.append(sa / (kn * sf))
            es.append(rs[-1] * np.sqrt(1 / sa + 1 / sf))
        errbar(ax, np.array(xs) + off, rs, es, color=col, ms=4, label=lab)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xlabel("Δt [ns] (each point = one clean gap between echo spikes)")
    ax.set_ylabel("adjacent / far  (normalised at 1.2–3 µs)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Capture would push this above 1 at small Δt; it is below", fontsize=10, loc="left")

    # (c) same-crystal: tail after the muon pulse vs tail after the 175 ns echo
    ax = axs[2]
    same = project(H, BE, 0)
    i1 = np.argmax(np.where((c > 160) & (c < 190), same, 0))
    for t0, lo, hi, lab, col in [(0.0, 60, 100, "after the muon pulse (t = 0)", "k"),
                                 (c[i1], 20, 120, f"after the echo at {c[i1]:.0f} ns", C_FIT)]:
        x = c - t0
        m = (x > lo) & (x < hi)
        y = same[m]
        ref = np.interp(90, x[m], y)
        p = np.polyfit(x[m], np.log(y), 1, w=np.sqrt(y))
        ax.plot(x[m], y / ref, color=col, lw=1.4, marker="o", ms=3,
                label=f"{lab}: slope τ ≈ {-1 / p[0]:.0f} ns")
    xx = np.linspace(20, 120, 50)
    ax.plot(xx, np.exp(-(xx - 90) / TAU_CAP), color=INK2, ls="--", lw=1.2, label=f"{TAU_CAP:.0f} ns (µ⁻ in CsI)")
    ax.set_yscale("log")
    ax.set_xlabel("time since pulse [ns]")
    ax.set_ylabel("same-crystal hits (normalised at +90 ns)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Pulse tails with capture-like time constants", fontsize=10, loc="left")
    fig.savefig(f"{out}/14_capture_hits.png")
    plt.close(fig)
    for lab, m in rows:
        print(f"{lab:35s} k = {m.values['k']:.2f}  A = {m.values['A']:.0f} +- {m.errors['A']:.0f} /10ns  "
              f"chi2/ndf = {m.fval:.0f}/{fitw.sum() - 2}")
