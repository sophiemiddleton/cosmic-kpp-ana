"""Calo plots from fill_calo.py output: stopped-muon lifetime, Michel spectrum, cluster/crystal QA.

Usage: python plot_calo.py <calo.npz> <outdir> [run]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.special import erfc

C_DISK = {0: "#2a78d6", 1: "#eb6834"}
C_FIT = "#c0392b"
INK2 = "#555555"
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
    "axes.edgecolor": "#444444", "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 140, "savefig.bbox": "tight",
})

# analysis cuts
DR_CUT = 80.0          # mm, delayed cluster COG to prompt cluster COG
ED_MAX = 60.0          # MeV, "small" delayed cluster
T_LO, T_HI = 1.25, 20  # us, lifetime fit range (starts after the echo comb at <=1.2 us)
SIG = (1.25, 10.0)     # us, Michel signal window
SB = (11.0, 20.0)      # us, accidental sideband
TAU_REF = 2.1969811    # us, PDG mu+ lifetime
CRY_PITCH = 34.3       # mm


def errbar(ax, x, y, ey, **kw):
    kw.setdefault("fmt", "o")
    kw.setdefault("ms", 3.5)
    kw.setdefault("color", "k")
    kw.setdefault("elinewidth", 1)
    kw.setdefault("capsize", 0)
    return ax.errorbar(x, y, ey, **kw)


def stamp(ax, run, extra=""):
    ax.text(0.0, 1.01, f"Mu2e calorimeter, run {run}{extra}", transform=ax.transAxes,
            fontsize=9, color=INK2, va="bottom")


def fit_lifetime(dt, lo=T_LO, hi=T_HI, bw=0.25):
    edges = np.arange(lo, hi + 1e-9, bw)
    n, _ = np.histogram(dt, edges)

    def cdf(x, n0, tau, b):  # expected cumulative counts from lo, n0/b in counts per bin
        return (n0 / bw) * tau * (np.exp(-(lo) / tau) - np.exp(-x / tau)) + (b / bw) * (x - lo)

    m = Minuit(ExtendedBinnedNLL(n, edges, cdf), n0=max(n[0], 1) * np.exp(lo / 2.2), tau=2.2,
               b=max(n[-5:].mean(), 0.1))
    m.limits["tau"] = (0.5, 10)
    m.limits["b"] = (0, None)
    m.limits["n0"] = (0, None)
    m.migrad()
    m.hesse()
    mu = np.diff(cdf(edges, *m.values))
    ok = n > 0
    chi2 = np.sum((n[ok] - mu[ok]) ** 2 / n[ok])
    return dict(m=m, edges=edges, n=n, mu=mu, chi2=chi2, ndf=ok.sum() - 3,
                tau=m.values["tau"], etau=m.errors["tau"])


def model_curve(f, x):
    n0, tau, b = f["m"].values
    return n0 * np.exp(-x / tau) + b


def plot_lifetime(P, run, out):
    sel = (P["dr"] < DR_CUT) & (P["ed"] < ED_MAX)
    f = fit_lifetime(P["dt"][sel])
    c = 0.5 * (f["edges"][1:] + f["edges"][:-1])
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3.2, 1], hspace=0.05))
    errbar(ax, c, f["n"], np.sqrt(f["n"]), label=f"data ({f['n'].sum():,} pairs)")
    x = np.linspace(T_LO, T_HI, 400)
    ax.plot(x, model_curve(f, x), color=C_FIT, lw=1.6, label=r"fit $N_0\,e^{-\Delta t/\tau}+B$")
    ax.set_yscale("log")
    ax.set_ylabel("pairs per 0.25 µs")
    m = f["m"]
    ax.text(0.97, 0.95,
            f"τ = {f['tau']:.3f} ± {f['etau']:.3f} µs\n"
            f"B = {m.values['b']:.1f} ± {m.errors['b']:.1f} / bin\n"
            f"χ²/ndf = {f['chi2']:.0f} / {f['ndf']}\n"
            f"PDG τ(µ⁺) = {TAU_REF:.3f} µs",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(fc="white", ec="#cccccc"))
    ax.legend(loc="lower left")
    stamp(ax, run, f"   |   same disk, ΔR < {DR_CUT:.0f} mm, E$_{{del}}$ < {ED_MAX:.0f} MeV")
    pull = (f["n"] - f["mu"]) / np.sqrt(np.maximum(f["mu"], 1))
    axp.bar(c, pull, width=0.22, color="#888888")
    axp.axhline(0, color="k", lw=0.8)
    axp.set_ylim(-4, 4)
    axp.set_ylabel("pull")
    axp.set_xlabel("Δt delayed − prompt cluster [µs]")
    fig.savefig(f"{out}/01_muon_lifetime.png")
    plt.close(fig)
    return f


def plot_dt_fine(P, run, out):
    sel = (P["dr"] < DR_CUT) & (P["ed"] < ED_MAX)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (lo, hi, bw) in zip(axs, [(0, 3, 0.01), (0, 25, 0.1)]):
        e = np.arange(lo, hi + 1e-9, bw)
        n, _ = np.histogram(P["dt"][sel], e)
        ax.stairs(n, e, color="k", lw=0.9)
        ax.axvspan(0, T_LO, color="#f2c4b3", alpha=0.45, lw=0, label="excluded (echo comb)")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Δt [µs]")
        ax.set_ylabel(f"pairs per {bw*1000:.0f} ns")
        stamp(ax, run)
    axs[0].legend(loc="upper right")
    axs[0].set_title("Fixed-delay echo clusters below 1.2 µs", fontsize=11, loc="left", pad=16)
    axs[1].set_title("Full candidate range", fontsize=11, loc="left", pad=16)
    fig.savefig(f"{out}/02_dt_fine_echo_comb.png")
    plt.close(fig)


def plot_tau_stability(P, run, out):
    base = dict(dr=DR_CUT, ed=ED_MAX, lo=T_LO, disk=None)
    scans = {
        "Δt fit start [µs]": ("lo", [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
        "ΔR cut [mm]": ("dr", [40, 50, 60, 80, 100, 120, 150]),
        "E$_{del}$ max [MeV]": ("ed", [30, 40, 50, 60, 80, 100]),
        "disk": ("disk", [0, 1]),
    }
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (lab, (k, vals)) in zip(axs, scans.items()):
        taus, errs = [], []
        for v in vals:
            c = dict(base, **{k: v})
            s = (P["dr"] < c["dr"]) & (P["ed"] < c["ed"])
            if c["disk"] is not None:
                s &= P["disk"] == c["disk"]
            f = fit_lifetime(P["dt"][s], lo=c["lo"])
            taus.append(f["tau"])
            errs.append(f["etau"])
        xs = np.arange(len(vals)) if k == "disk" else np.array(vals, float)
        colors = [C_DISK[v] for v in vals] if k == "disk" else ["k"] * len(vals)
        for x, t, e, col in zip(xs, taus, errs, colors):
            errbar(ax, x, t, e, color=col)
        if k == "disk":
            ax.set_xticks(xs, [f"disk {v}" for v in vals])
            ax.set_xlim(-0.6, 1.6)
        ax.axhline(TAU_REF, color=INK2, ls="--", lw=1)
        ax.set_xlabel(lab)
    axs[0].set_ylabel("fitted τ [µs]")
    axs[0].text(0.02, 0.04, "dashed: PDG τ(µ⁺)", transform=axs[0].transAxes, fontsize=9, color=INK2)
    stamp(axs[0], run)
    fig.savefig(f"{out}/03_tau_stability.png")
    plt.close(fig)


def fit_two_component(dt, lo=T_LO, hi=T_HI, bw=0.25):
    """mu+ (tau fixed to PDG) + a second exponential with free lifetime + flat background."""
    edges = np.arange(lo, hi + 1e-9, bw)
    n, _ = np.histogram(dt, edges)

    def cdf(x, n1, n2, t2, b):
        return ((n1 / bw) * TAU_REF * (np.exp(-lo / TAU_REF) - np.exp(-x / TAU_REF))
                + (n2 / bw) * t2 * (np.exp(-lo / t2) - np.exp(-x / t2)) + (b / bw) * (x - lo))

    m = Minuit(ExtendedBinnedNLL(n, edges, cdf), n1=n[0], n2=n[0] / 4, t2=1.0, b=max(n[-5:].mean(), 0.1))
    m.limits["t2"] = (0.1, 2.1)
    for p in ("n1", "n2", "b"):
        m.limits[p] = (0, None)
    m.migrad()
    m.hesse()
    v = m.values
    a1 = v["n1"] * TAU_REF * np.exp(-lo / TAU_REF)
    a2 = v["n2"] * v["t2"] * np.exp(-lo / v["t2"])
    return m, edges, n, a2 / (a1 + a2)


def plot_tau_radius(P, run, out):
    sel = (P["dr"] < DR_CUT) & (P["ed"] < ED_MAX)
    r = np.hypot(P["xd"], P["yd"])
    bands = [(0, 430), (430, 500), (500, 570), (570, 700)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    ax = axs[0]
    for off, lo_t, col in [(-4, T_LO, "k"), (4, 3.0, C_FIT)]:
        xs, ts, es = [], [], []
        for lo, hi in bands:
            m = sel & (r >= lo) & (r < hi)
            f = fit_lifetime(P["dt"][m], lo=lo_t)
            xs.append(np.median(r[m & (P["dt"] > T_LO) & (P["dt"] < 10)]))
            ts.append(f["tau"])
            es.append(f["etau"])
        errbar(ax, np.array(xs) + off, ts, es, color=col, label=f"fit from {lo_t} µs")
    ax.axhline(TAU_REF, color=INK2, ls="--", lw=1, label="PDG τ(µ⁺)")
    for lo, hi in bands[1:]:
        ax.axvline(lo, color="#dddddd", lw=0.8)
    ax.set_xlabel("decay radius on the disk (median of band) [mm]")
    ax.set_ylabel("fitted τ [µs]")
    ax.legend(fontsize=9)
    stamp(ax, run)

    ax = axs[1]
    m, edges, n, frac = fit_two_component(P["dt"][sel])
    c = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]
    v = m.values
    errbar(ax, c, n, np.sqrt(n), label="data")
    x = np.linspace(T_LO, T_HI, 400)
    ax.plot(x, v["n1"] * np.exp(-x / TAU_REF) + v["n2"] * np.exp(-x / v["t2"]) + v["b"], color=C_FIT, lw=1.6,
            label="total")
    ax.plot(x, v["n1"] * np.exp(-x / TAU_REF), color=C_DISK[0], lw=1.2, ls="--", label="µ⁺, τ fixed 2.197 µs")
    ax.plot(x, v["n2"] * np.exp(-x / v["t2"]), color=C_DISK[1], lw=1.2, ls="--",
            label=f"2nd: τ = {v['t2']:.2f} ± {m.errors['t2']:.2f} µs")
    ax.set_yscale("log")
    ax.set_ylim(3, None)
    ax.set_xlabel("Δt [µs]")
    ax.set_ylabel(f"pairs per {bw:.2f} µs")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.03, 0.05, f"2nd component: {frac:.1%} of decays after {T_LO} µs", transform=ax.transAxes, fontsize=9)
    fig.savefig(f"{out}/16_tau_radius_two_component.png")
    plt.close(fig)
    return m, frac


def michel_hist(P, sel, bw=1.0, emax=72):
    e = np.arange(0, emax + 1e-9, bw)
    t = P["dt"]
    ns, _ = np.histogram(P["ed"][sel & (t > SIG[0]) & (t < SIG[1])], e)
    nb, _ = np.histogram(P["ed"][sel & (t > SB[0]) & (t < SB[1])], e)
    k = (SIG[1] - SIG[0]) / (SB[1] - SB[0])
    return e, ns, nb, k, ns - k * nb, np.sqrt(ns + k * k * nb)


def edge(E, A, E0, s):
    return A / 2 * erfc((E - E0) / (np.sqrt(2) * s))


def fit_edge(e, y, ey, lo=30, hi=52):
    c = 0.5 * (e[1:] + e[:-1])
    m = (c > lo) & (c < hi) & (ey > 0)
    mn = Minuit(LeastSquares(c[m], y[m], ey[m], edge), A=y[m].max(), E0=43, s=5)
    mn.migrad()
    mn.hesse()
    return mn, m.sum() - 3


def plot_michel(P, run, out):
    sel = P["dr"] < DR_CUT  # no E_del cut here: the spectrum is the observable
    e, ns, nb, k, y, ey = michel_hist(P, sel)
    c = 0.5 * (e[1:] + e[:-1])
    mn, ndf = fit_edge(e, y, ey)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    ax.stairs(ns, e, color="k", lw=1.2, label=f"signal window {SIG[0]}–{SIG[1]:.0f} µs")
    ax.stairs(k * nb, e, color="#888888", lw=1.2, ls="--",
              label=f"sideband {SB[0]:.0f}–{SB[1]:.0f} µs × {k:.3f}")
    ax.set_xlabel("delayed cluster energy [MeV]")
    ax.set_ylabel("clusters per MeV")
    ax.legend()
    stamp(ax, run)

    ax = axs[1]
    errbar(ax, c, y, ey, label=f"signal − sideband\n({y.sum():,.0f} decays)")
    x = np.linspace(30, 52, 200)
    ax.plot(x, edge(x, *mn.values), color=C_FIT, lw=1.8,
            label="erfc edge fit\n30–52 MeV")
    ax.axvline(52.8, color=INK2, ls=":", lw=1)
    ax.text(52.8, ax.get_ylim()[1] * 0.02, " 52.8 MeV\n Michel endpoint", fontsize=8, color=INK2)
    ax.text(0.97, 0.97,
            f"E₀ = {mn.values['E0']:.2f} ± {mn.errors['E0']:.2f} MeV\n"
            f"σ = {mn.values['s']:.2f} ± {mn.errors['s']:.2f} MeV\n"
            f"χ²/ndf = {mn.fval:.0f} / {ndf}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(fc="white", ec="#cccccc"))
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), fontsize=9)
    ax.set_xlabel("delayed cluster energy [MeV]")
    ax.set_ylabel("decays per MeV")
    stamp(ax, run, f"   |   ΔR < {DR_CUT:.0f} mm")
    fig.savefig(f"{out}/04_michel_spectrum.png")
    plt.close(fig)
    return mn


def plot_michel_disks(P, run, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rows = []
    for d in (0, 1):
        sel = (P["dr"] < DR_CUT) & (P["disk"] == d)
        e, ns, nb, k, y, ey = michel_hist(P, sel)
        mn, ndf = fit_edge(e, y, ey)
        norm = y.sum()
        c = 0.5 * (e[1:] + e[:-1])
        errbar(ax, c, y / norm, ey / norm, color=C_DISK[d], ms=3,
               label=f"disk {d}: E₀ = {mn.values['E0']:.2f} ± {mn.errors['E0']:.2f}, "
                     f"σ = {mn.values['s']:.2f} ± {mn.errors['s']:.2f} MeV  ({norm:,.0f} decays)")
        x = np.linspace(30, 52, 200)
        ax.plot(x, edge(x, *mn.values) / norm, color=C_DISK[d], lw=1.4)
        rows.append((d, mn))
    ax.set_xlabel("delayed cluster energy [MeV]")
    ax.set_ylabel("fraction per MeV")
    ax.legend(fontsize=9, loc="upper right")
    stamp(ax, run, "   |   background-subtracted, unit area")
    fig.savefig(f"{out}/05_michel_per_disk.png")
    plt.close(fig)
    return rows


def plot_decay_map_and_dr(P, run, out):
    t = P["dt"]
    k = (SIG[1] - SIG[0]) / (SB[1] - SB[0])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    # delta-R distribution: signal minus scaled sideband (loose cut window from the filler)
    e = np.arange(0, 151, 5)
    base = P["ed"] < ED_MAX
    ns, _ = np.histogram(P["dr"][base & (t > SIG[0]) & (t < SIG[1])], e)
    nb, _ = np.histogram(P["dr"][base & (t > SB[0]) & (t < SB[1])], e)
    c = 0.5 * (e[1:] + e[:-1])
    ax = axs[0]
    errbar(ax, c, ns - k * nb, np.sqrt(ns + k * k * nb), label="signal − scaled sideband")
    ax.stairs(k * nb, e, color="#888888", ls="--", label="scaled sideband")
    ax.axvline(DR_CUT, color=C_FIT, lw=1.2, ls="--")
    ax.text(DR_CUT + 2, ax.get_ylim()[1] * 0.9, f"cut {DR_CUT:.0f} mm", color=C_FIT, fontsize=9)
    ax.set_xlabel("ΔR delayed − prompt COG [mm]")
    ax.set_ylabel("pairs per 5 mm")
    ax.legend(fontsize=9)
    stamp(ax, run)
    # decay position on each disk (signal window, no subtraction: background is ~1%)
    sel = (P["dr"] < DR_CUT) & base & (t > SIG[0]) & (t < SIG[1])
    b = np.linspace(-700, 700, 57)
    hmax = 0
    hs = []
    for d in (0, 1):
        h, _, _ = np.histogram2d(P["xd"][sel & (P["disk"] == d)], P["yd"][sel & (P["disk"] == d)], [b, b])
        hs.append(h)
        hmax = max(hmax, h.max())
    for d, ax in zip((0, 1), axs[1:]):
        h = np.where(hs[d] > 0, hs[d], np.nan)
        im = ax.pcolormesh(b, b, h.T, cmap="viridis", vmin=0, vmax=hmax)
        ax.set_aspect("equal")
        ax.set_title(f"disk {d}: {int(hs[d].sum()):,} decay candidates", fontsize=11, loc="left")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(False)
    fig.colorbar(im, ax=axs[2], label="delayed clusters / bin")
    fig.savefig(f"{out}/06_decay_dr_and_map.png")
    plt.close(fig)


def plot_clusters(H, run, out):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax = axs[0, 0]
    e = H["B_CLE"]
    for d in (0, 1):
        ax.stairs(H["cl_e"][d], e, color=C_DISK[d], lw=1.3, label=f"disk {d} ({H['cl_e'][d].sum():,.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("cluster energy [MeV]")
    ax.set_ylabel("clusters per MeV")
    ax.legend()
    stamp(ax, run, f"   |   {int(H['nev']):,} events")

    ax = axs[0, 1]
    e = H["B_SIZE"]
    for d in (0, 1):
        ax.stairs(H["cl_size"][d] / H["cl_size"][d].sum(), e, color=C_DISK[d], lw=1.3, label=f"disk {d}")
    ax.set_yscale("log")
    ax.set_xlabel("crystals per cluster")
    ax.set_ylabel("fraction of clusters")
    ax.set_xlim(0.5, 40)
    ax.legend()

    ax = axs[1, 0]
    e = H["B_NCL"]
    ax.stairs(H["cl_ncl"], e, color="k", lw=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("clusters per event (both disks)")
    ax.set_ylabel("events")
    f0 = H["cl_ncl"][0] / H["cl_ncl"].sum()
    ax.text(0.97, 0.95, f"{f0:.1%} of events have no cluster", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=INK2)

    ax = axs[1, 1]
    es, ss = H["B_CLE"], H["B_SIZE"]
    h = H["cl_e_size"].astype(float)
    h[h == 0] = np.nan
    im = ax.pcolormesh(es, ss, h.T, cmap="viridis", norm=matplotlib.colors.LogNorm())
    ax.set_xlim(0, 300)
    ax.set_ylim(0.5, 30)
    ax.set_xlabel("cluster energy [MeV]")
    ax.set_ylabel("crystals per cluster")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="clusters")
    fig.savefig(f"{out}/07_cluster_basics.png")
    plt.close(fig)


def plot_times(H, run, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    e = H["B_T"] / 1000
    ax.stairs(H["ht_t"], e, color="#888888", lw=1.2, label="hits")
    ax.stairs(H["cl_t"], e, color="k", lw=1.2, label="clusters")
    ax.set_yscale("log")
    ax.set_xlabel("time in readout window [µs]")
    ax.set_ylabel("entries per 200 ns")
    ax.legend()
    stamp(ax, run)
    fig.savefig(f"{out}/08_time_in_window.png")
    plt.close(fig)


def plot_occupancy(H, run, out):
    b = H["B_XY"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4))
    vmax = H["cl_xy"].max()
    for d, ax in enumerate(axs):
        h = H["cl_xy"][d].astype(float)
        h[h == 0] = np.nan
        im = ax.pcolormesh(b, b, h.T, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"disk {d}: cluster COG", fontsize=11, loc="left", pad=18)
        stamp(ax, run)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=axs, label="clusters / 10×10 mm²", shrink=0.85)
    fig.savefig(f"{out}/09_cluster_occupancy.png")
    plt.close(fig)


def echo_mask(n):
    """Bins of a 10 ns Δt histogram that belong to an echo spike (plus +-2 bins of tails)."""
    med = np.array([np.median(n[max(0, i - 25):i + 25]) for i in range(len(n))])
    sp = n > med + 4 * np.sqrt(med) + 30
    m = sp.copy()
    for s in (-2, -1, 1, 2):
        m |= np.roll(sp, s)
    return m


def plot_capture(P, f, run, out):
    """Search for a fast (mu- capture) component under the echo comb at Δt < 1.25 us."""
    sel = (P["dr"] < DR_CUT) & (P["ed"] < ED_MAX)
    bw = 0.01
    e = np.arange(0.02, 3.0 + 1e-9, bw)
    c = 0.5 * (e[1:] + e[:-1])
    n, _ = np.histogram(P["dt"][sel], e)
    good = ~echo_mask(n)
    n0, tau, b = f["m"].values
    s = bw / 0.25  # main fit is per 0.25 us bin

    def mup(x):
        return s * (n0 * np.exp(-x / tau) + b)

    def nll(A, tc, N0):
        mu = A * np.exp(-c / tc) + N0 * np.exp(-c / TAU_REF) + s * b
        return 2 * np.sum((mu - n * np.log(mu))[good])

    m = Minuit(nll, A=300, tc=0.2, N0=s * n0)
    m.errordef = 1
    m.limits["tc"] = (0.01, 1.5)
    m.limits["A"] = (0, None)
    m.migrad()
    m.hesse()

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axs[0]
    w = c < 1.6
    ax.stairs(np.where(good, np.nan, n)[w], e[:w.sum() + 1], color="#cccccc", lw=0.8,
              label="echo-spike bins (masked)")
    errbar(ax, c[good & w], n[good & w], np.sqrt(n[good & w]), ms=3, label="clean bins between spikes")
    x = np.linspace(0.02, 1.6, 400)
    ax.plot(x, mup(x), color=INK2, ls="--", lw=1.4, label=f"µ⁺ fit from {T_LO} µs, extrapolated")
    ax.plot(x, m.values["A"] * np.exp(-x / m.values["tc"]) + m.values["N0"] * np.exp(-x / TAU_REF) + s * b,
            color=C_FIT, lw=1.6,
            label=f"+ fast component: τ = {m.values['tc']*1000:.0f} ± {m.errors['tc']*1000:.0f} ns")
    ax.axvline(T_LO, color=INK2, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(300, 3e3)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("Δt [µs]")
    ax.set_ylabel("pairs per 10 ns")
    ax.legend(fontsize=9, loc="upper right")
    stamp(ax, run, f"   |   ΔR < {DR_CUT:.0f} mm, E$_{{del}}$ < {ED_MAX:.0f} MeV")

    ax = axs[1]
    ib = np.clip(np.digitize(P["dt"], e) - 1, 0, len(n) - 1)
    ing = (P["dt"] >= e[0]) & (P["dt"] < e[-1]) & good[ib]
    eb = np.arange(10, 101, 2.5)
    base = P["dr"] < DR_CUT
    for lab, msk, col, ls in [
        (f"Michel window {T_LO}–10 µs", base & (P["dt"] > T_LO) & (P["dt"] < 10), "k", "-"),
        ("clean gaps 0.08–0.35 µs", base & ing & (P["dt"] > 0.08) & (P["dt"] < 0.35), C_FIT, "-"),
        ("inside echo spike 0.16–0.19 µs", base & (P["dt"] > 0.16) & (P["dt"] < 0.19), "#888888", "--"),
    ]:
        h, _ = np.histogram(P["ed"][msk], eb)
        ax.stairs(h / h.sum() / 2.5, eb, color=col, ls=ls, lw=1.5, label=lab)
    ax.axvline(52.8, color=INK2, ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("delayed cluster energy [MeV]")
    ax.set_ylabel("fraction per MeV")
    ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    ax.set_title("No soft excess from capture γ/n; >53 MeV tail = echo leakage", fontsize=10, loc="left")
    fig.savefig(f"{out}/13_capture_search.png")
    plt.close(fig)
    return m


def gaus(x, a, mu, s):
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def crystal_mip_peaks(H):
    e = H["B_HTE"]
    c = 0.5 * (e[1:] + e[:-1])
    n = len(H["ht_e"])
    mu = np.full(n, np.nan)
    sig = np.full(n, np.nan)
    nh = H["ht_e"].sum(1)
    win = (c > 12) & (c < 40)
    for i in range(n):
        y = H["ht_e"][i]
        if y[win].sum() < 200:
            continue
        ys = np.convolve(y, np.ones(5) / 5, mode="same")
        m0 = c[win][np.argmax(ys[win])]
        r = (c > 0.8 * m0) & (c < 1.25 * m0)
        try:
            p, _ = curve_fit(gaus, c[r], y[r], p0=[y[r].max(), m0, 0.15 * m0],
                             sigma=np.sqrt(np.maximum(y[r], 1)))
            if 10 < p[1] < 45 and 0 < abs(p[2]) < 20:
                mu[i], sig[i] = p[1], abs(p[2])
        except RuntimeError:
            pass
    return mu, sig, nh


def crystal_map(ax, H, vals, mask, cmap, vmin, vmax):
    patches = [Rectangle((x - CRY_PITCH / 2, y - CRY_PITCH / 2), CRY_PITCH, CRY_PITCH)
               for x, y in zip(H["cry_x"][mask], H["cry_y"][mask])]
    pc = PatchCollection(patches, cmap=cmap, edgecolor="white", linewidth=0.3)
    v = vals[mask]
    pc.set_array(np.ma.masked_invalid(v))
    pc.set_clim(vmin, vmax)
    pc.cmap.set_bad("#bbbbbb")
    ax.add_collection(pc)
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    return pc


def plot_crystals(H, run, out):
    mu, sig, nh = crystal_mip_peaks(H)
    ncry = len(mu)
    disk = (np.arange(ncry) >= ncry // 2).astype(int)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    e = H["B_HTE"]
    for d in (0, 1):
        ax.stairs(H["ht_e"][disk == d].sum(0), e, color=C_DISK[d], lw=1.3, label=f"disk {d}")
    ax.set_yscale("log")
    ax.set_xlabel("crystal hit energy [MeV]")
    ax.set_ylabel("hits per 0.5 MeV")
    ax.legend()
    stamp(ax, run)
    ax = axs[1]
    bins = np.linspace(10, 35, 51)
    for d in (0, 1):
        v = mu[(disk == d) & np.isfinite(mu)]
        ax.hist(v, bins, histtype="step", color=C_DISK[d], lw=1.5,
                label=f"disk {d}: mean {v.mean():.2f}, RMS {v.std():.2f} MeV ({len(v)} crystals)")
    ax.set_xlabel("per-crystal MIP peak (Gaussian core fit) [MeV]")
    ax.set_ylabel("crystals")
    ax.legend(fontsize=9)
    fig.savefig(f"{out}/10_crystal_mip_spectra.png")
    plt.close(fig)

    fin = np.isfinite(mu)
    lo, hi = np.nanpercentile(mu, [1, 99])
    for name, vals, lab, cm, vr in [
        ("11_crystal_mip_peak_map", mu, "MIP peak [MeV]", "viridis", (lo, hi)),
        ("12_crystal_hit_count_map", nh.astype(float), "hits per crystal", "viridis",
         (0, np.nanpercentile(nh, 99.5))),
    ]:
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4))
        for d, ax in enumerate(axs):
            pc = crystal_map(ax, H, vals, disk == d, cm, *vr)
            ax.set_title(f"disk {d}", fontsize=11, loc="left", pad=18)
            stamp(ax, run, "   |   grey: no fit" if name.startswith("11") else "")
        fig.colorbar(pc, ax=axs, label=lab, shrink=0.85)
        fig.savefig(f"{out}/{name}.png")
        plt.close(fig)
    return mu, sig, nh, fin


if __name__ == "__main__":
    H = np.load(sys.argv[1])
    out = sys.argv[2]
    run = sys.argv[3] if len(sys.argv) > 3 else "125329"
    os.makedirs(out, exist_ok=True)
    P = {k[5:]: H[k] for k in H.files if k.startswith("pair_")}
    P["dt"] = P["dt"] / 1000.0  # ns -> us

    f = plot_lifetime(P, run, out)
    print(f"tau = {f['tau']:.4f} +- {f['etau']:.4f} us, chi2/ndf = {f['chi2']:.1f}/{f['ndf']}, "
          f"pairs in fit = {f['n'].sum()}")
    plot_dt_fine(P, run, out)
    mc = plot_capture(P, f, run, out)
    print(f"fast component: tau = {mc.values['tc']*1000:.0f} +- {mc.errors['tc']*1000:.0f} ns, "
          f"A = {mc.values['A']:.0f} +- {mc.errors['A']:.0f} per 10 ns at dt=0")
    plot_tau_stability(P, run, out)
    m2, frac = plot_tau_radius(P, run, out)
    print(f"two-component: tau2 = {m2.values['t2']:.2f} +- {m2.errors['t2']:.2f} us, fraction after {T_LO} us = {frac:.3f}")
    mn = plot_michel(P, run, out)
    print(f"Michel edge E0 = {mn.values['E0']:.2f} +- {mn.errors['E0']:.2f}, "
          f"sigma = {mn.values['s']:.2f} +- {mn.errors['s']:.2f}, chi2 = {mn.fval:.1f}")
    for d, m in plot_michel_disks(P, run, out):
        print(f"  disk {d}: E0 = {m.values['E0']:.2f} +- {m.errors['E0']:.2f}, "
              f"sigma = {m.values['s']:.2f} +- {m.errors['s']:.2f}")
    plot_decay_map_and_dr(P, run, out)
    plot_clusters(H, run, out)
    plot_times(H, run, out)
    plot_occupancy(H, run, out)
    mu, sig, nh, fin = plot_crystals(H, run, out)
    print(f"crystal MIP peaks fitted: {fin.sum()}/{len(mu)}, mean {np.nanmean(mu):.2f}, "
          f"RMS {np.nanstd(mu):.2f} MeV")
