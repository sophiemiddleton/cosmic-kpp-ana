"""Stopped-muon studies from fill_stops.py (and fill_wfcapture.py) output.

  20  stop selection: Michel fraction (stop purity) vs last-crystal energy and room ahead; Michel position bias
  21  muon lifetime for the stop sample
  22  Michel deposit spectrum down to 0.5 MeV (hit groups, not clusters)
  23  up/down Michel asymmetry: stop-position bias and its time dependence
  24  delayed soft activity (neutron capture gammas) after stop candidates
  25  second pulse in the stop crystal's waveform (mu- capture in CsI), if the fill_wfcapture.py output is given

Usage: python plot_stops.py <stops.npz> <outdir> [wfcapture.npz] [run]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares

from plot_calo import C_DISK, C_FIT, INK2, TAU_REF, edge, errbar, fit_edge, fit_lifetime, stamp

# selections (fill_wfcapture.py uses the same STOP cuts)
T_STOP = (2000.0, 65000.0)   # ns, stop time in the readout window
ELAST = (25.0, 70.0)         # MeV, energy of the last crystal
AHEAD_MIN = 40.0             # mm of disk ahead of the stop along the track
MICH = dict(e=(10.0, 60.0), dmin=80.0)
SIG = (1500.0, 9000.0)       # ns, Michel window (after the ~1 us post-muon burst)
SB = (15000.0, 22500.0)      # ns, sideband of the same width
EL_BINS = [0, 15, 25, 35, 45, 70]
C_CLS = ["#9aa7b4", "#5f87b0", "#2a78d6", "#eb6834", "#c0392b"]


def load(path):
    d = np.load(path)
    S = {k[2:]: d[k] for k in d.files if k.startswith("s_")}
    G = {k[2:]: d[k] for k in d.files if k.startswith("g_")}
    return S, G


def stop_sel(S):
    return ((S["elast"] > ELAST[0]) & (S["elast"] < ELAST[1]) & (S["ahead"] >= AHEAD_MIN)
            & (S["t"] > T_STOP[0]) & (S["t"] < T_STOP[1]))


def michel_groups(G, emin=MICH["e"][0], emax=MICH["e"][1]):
    return (G["e"] > emin) & (G["e"] < emax) & (G["dmin"] < MICH["dmin"])


def michel_count(G, gsel):
    """Background-subtracted Michel groups in SIG (sideband SB), and its error."""
    s = np.sum(gsel & (G["dt"] > SIG[0]) & (G["dt"] < SIG[1]))
    b = np.sum(gsel & (G["dt"] > SB[0]) & (G["dt"] < SB[1]))
    return s - b, np.sqrt(s + b)


def window_eff(lo=SIG[0], hi=SIG[1]):
    return np.exp(-lo / 1000 / TAU_REF) - np.exp(-hi / 1000 / TAU_REF)


# ---------------------------------------------------------------- 20
def plot_selection(S, G, run, out):
    gs = G["stop"]
    base = (S["t"] > T_STOP[0]) & (S["t"] < T_STOP[1])
    mg = michel_groups(G)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    ax = axs[0]
    eb = np.arange(0, 101, 5.0)
    c = 0.5 * (eb[1:] + eb[:-1])
    for lab, cut, col in [("all", base, "#888888"), (f"≥ {AHEAD_MIN:.0f} mm ahead", base & (S["ahead"] >= AHEAD_MIN), "k"),
                          ("≥ 80 mm ahead", base & (S["ahead"] >= 80), C_FIT)]:
        f, ef = [], []
        for lo, hi in zip(eb[:-1], eb[1:]):
            s = cut & (S["elast"] >= lo) & (S["elast"] < hi)
            n, en = michel_count(G, mg & s[gs])
            f.append(n / max(s.sum(), 1))
            ef.append(en / max(s.sum(), 1))
        errbar(ax, c, np.array(f) * 100, np.array(ef) * 100, color=col, label=lab)
    ax.axvspan(*ELAST, color="#f2e2b3", alpha=0.5, lw=0, label="stop selection")
    ax.set_xlabel("energy in the last crystal of the track [MeV]")
    ax.set_ylabel(f"Michel per candidate, {SIG[0]/1000:.1f}–{SIG[1]/1000:.0f} µs [%]")
    ax.legend(fontsize=9)
    stamp(ax, run)

    ax = axs[1]
    ab = np.arange(0, 201, 10.0)
    for lab, cut, col in [("all", base, "#888888"), ("E$_{last}$ 25–70 MeV", base & (S["elast"] > 25) & (S["elast"] < 70), "k")]:
        f, ef = [], []
        for lo, hi in zip(ab[:-1], ab[1:]):
            s = cut & (S["ahead"] >= lo) & (S["ahead"] < hi + (1 if hi == 200 else 0))
            n, en = michel_count(G, mg & s[gs])
            f.append(n / max(s.sum(), 1))
            ef.append(en / max(s.sum(), 1))
        errbar(ax, 0.5 * (ab[1:] + ab[:-1]), np.array(f) * 100, np.array(ef) * 100, color=col, label=lab)
    ax.set_xlabel("disk ahead of the stop along the track [mm] (200 = ≥ 200)")
    ax.set_ylabel("Michel per candidate [%]")
    ax.legend(fontsize=9)
    stamp(ax, run)

    ax = axs[2]
    b = np.arange(-6.5, 6.6, 1.0) * 34.3 / 2
    cb = 0.5 * (b[1:] + b[:-1])
    for (lo, hi), col in zip(zip(EL_BINS[:-1], EL_BINS[1:]), C_CLS):
        s = (base & (S["ahead"] >= AHEAD_MIN) & (S["elast"] >= lo) & (S["elast"] < hi))[gs] & michel_groups(G, 3, 60)
        hs, _ = np.histogram(G["y"][s & (G["dt"] > SIG[0]) & (G["dt"] < SIG[1])], b)
        hb, _ = np.histogram(G["y"][s & (G["dt"] > SB[0]) & (G["dt"] < SB[1])], b)
        y = hs - hb
        ax.plot(cb, y / max(y.sum(), 1), "o-", color=col, ms=3, lw=1, label=f"E$_{{last}}$ {lo}–{hi} MeV")
    ax.set_xlabel("Michel group y − stop crystal y [mm]  (+ = up)")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)
    stamp(ax, run)
    fig.savefig(f"{out}/20_stop_selection.png")
    plt.close(fig)


# ---------------------------------------------------------------- 21
def plot_lifetime_stops(S, G, run, out):
    gs = G["stop"]
    sel = stop_sel(S)
    mg = michel_groups(G)
    dt = G["dt"] / 1000
    f = fit_lifetime(dt[mg & sel[gs]], lo=1.5)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    ax = axs[0]
    c = 0.5 * (f["edges"][1:] + f["edges"][:-1])
    errbar(ax, c, f["n"], np.sqrt(f["n"]), label=f"stop sample ({f['n'].sum():,} groups)")
    x = np.linspace(1.5, 20, 300)
    n0, tau, bb = f["m"].values
    ax.plot(x, n0 * np.exp(-x / tau) + bb, color=C_FIT, lw=1.6, label="fit")
    ax.set_yscale("log")
    ax.set_xlabel("Δt Michel group − muon [µs]")
    ax.set_ylabel("groups per 0.25 µs")
    ax.text(0.97, 0.95, f"τ = {f['tau']:.3f} ± {f['etau']:.3f} µs\nχ²/ndf = {f['chi2']:.0f} / {f['ndf']}\n"
            f"PDG τ(µ⁺) = {TAU_REF:.3f} µs", transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(fc="white", ec="#cccccc"))
    ax.legend(loc="lower left", fontsize=9)
    stamp(ax, run, f"   |   E$_{{last}}$ {ELAST[0]:.0f}–{ELAST[1]:.0f} MeV, ≥ {AHEAD_MIN:.0f} mm ahead")

    ax = axs[1]
    base = (S["t"] > T_STOP[0]) & (S["t"] < T_STOP[1]) & (S["ahead"] >= AHEAD_MIN)
    for i, (lo, hi) in enumerate(zip(EL_BINS[:-1], EL_BINS[1:])):
        ff = fit_lifetime(dt[mg & (base & (S["elast"] >= lo) & (S["elast"] < hi))[gs]], lo=1.5)
        errbar(ax, 0.5 * (lo + hi), ff["tau"], ff["etau"], color=C_CLS[i])
    ax.axhline(TAU_REF, color=INK2, ls="--", lw=1)
    ax.set_xlabel("energy in the last crystal [MeV]")
    ax.set_ylabel("fitted τ [µs] (from 1.5 µs)")
    stamp(ax, run)

    ax = axs[2]
    los = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for lo in los:
        ff = fit_lifetime(dt[mg & sel[gs]], lo=lo)
        errbar(ax, lo, ff["tau"], ff["etau"])
    ax.axhline(TAU_REF, color=INK2, ls="--", lw=1)
    ax.set_xlabel("fit start [µs]")
    ax.set_ylabel("fitted τ [µs]")
    stamp(ax, run, "   |   stop sample")
    fig.savefig(f"{out}/21_stop_lifetime.png")
    plt.close(fig)
    return f


# ---------------------------------------------------------------- 22
def plot_michel_lowE(S, G, run, out):
    gs = G["stop"]
    sel = stop_sel(S)[gs] & (G["dmin"] < MICH["dmin"])
    b = np.r_[np.arange(0.5, 10, 0.5), np.arange(10, 81, 2.0)]
    hs, _ = np.histogram(G["e"][sel & (G["dt"] > SIG[0]) & (G["dt"] < SIG[1])], b)
    hb, _ = np.histogram(G["e"][sel & (G["dt"] > SB[0]) & (G["dt"] < SB[1])], b)
    y, ey = hs - hb, np.sqrt(hs + hb)
    w = np.diff(b)
    c = 0.5 * (b[1:] + b[:-1])
    tot = y[c < 70].sum()
    cum = np.cumsum(y) / tot
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    ax = axs[0]
    errbar(ax, c, y / w, ey / w, label=f"signal − sideband ({tot:,.0f} decays)")
    ax.set_yscale("log")
    ax.axvline(10, color=INK2, ls=":", lw=1)
    ax.text(10.5, ax.get_ylim()[1] * 0.3, "cluster\nthreshold", fontsize=8, color=INK2)
    # erfc edge on a 1 MeV binning, same range as plot_calo.py
    e1 = np.arange(0, 81, 1.0)
    h1s, _ = np.histogram(G["e"][sel & (G["dt"] > SIG[0]) & (G["dt"] < SIG[1])], e1)
    h1b, _ = np.histogram(G["e"][sel & (G["dt"] > SB[0]) & (G["dt"] < SB[1])], e1)
    mn, ndf = fit_edge(e1, h1s - h1b, np.sqrt(h1s + h1b))
    x = np.linspace(30, 52, 200)
    ax.plot(x, edge(x, *mn.values), color=C_FIT, lw=1.8, label="erfc edge fit 30–52 MeV")
    ax.text(0.03, 0.05, f"E₀ = {mn.values['E0']:.2f} ± {mn.errors['E0']:.2f} MeV\n"
            f"σ = {mn.values['s']:.2f} ± {mn.errors['s']:.2f} MeV\nχ²/ndf = {mn.fval:.0f} / {ndf}",
            transform=ax.transAxes, fontsize=10, bbox=dict(fc="white", ec="#cccccc"))
    ax.set_xlabel("delayed hit-group energy within 80 mm of the stop [MeV]")
    ax.set_ylabel("decays per MeV")
    ax.legend(fontsize=9, loc="upper right")
    stamp(ax, run, f"   |   stop sample, {SIG[0]/1000:.1f}–{SIG[1]/1000:.0f} µs")
    ax = axs[1]
    ax.plot(b[1:], cum * 100, "k-", lw=1.4)
    lines = []
    for thr in (3, 5, 10, 15):
        v = np.interp(thr, b[1:], cum) * 100
        ax.plot(thr, v, "o", color=C_FIT, ms=4)
        lines.append(f"below {thr:>2d} MeV: {v:4.1f}%")
    ax.text(0.04, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10, family="monospace",
            bbox=dict(fc="white", ec="#cccccc"))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 40)
    ax.set_xlabel("energy threshold [MeV]")
    ax.set_ylabel("decays below threshold [%]")
    stamp(ax, run)
    fig.savefig(f"{out}/22_michel_lowE.png")
    plt.close(fig)
    print(f"Michel edge (hit groups): E0 {mn.values['E0']:.2f} ± {mn.errors['E0']:.2f}, "
          f"sigma {mn.values['s']:.2f} ± {mn.errors['s']:.2f} MeV")
    return {thr: np.interp(thr, b[1:], cum) for thr in (1, 3, 5, 10)}


# ---------------------------------------------------------------- 23
def updown(G, gsel, lo, hi, emin=3.0):
    """Background-subtracted (up, down) Michel groups with dt in [lo, hi) ns, sideband scaled by width."""
    m = gsel & michel_groups(G, emin, 60)
    k = (hi - lo) / (SB[1] - SB[0])
    r = []
    for side in (G["y"] > 17.15, G["y"] < -17.15):
        s = np.sum(m & side & (G["dt"] >= lo) & (G["dt"] < hi))
        b = np.sum(m & side & (G["dt"] >= SB[0]) & (G["dt"] < SB[1]))
        r.append((s - k * b, s + k * k * b))
    (u, vu), (d, vd) = r
    a = (u - d) / (u + d)
    ea = 2 * np.sqrt(d * d * vu + u * u * vd) / (u + d) ** 2
    return a, ea


def plot_asymmetry(S, G, run, out):
    gs = G["stop"]
    base = (S["t"] > T_STOP[0]) & (S["t"] < T_STOP[1]) & (S["ahead"] >= AHEAD_MIN)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    ax = axs[0]
    eb = np.arange(0, 75, 5.0)
    a, ea = zip(*[updown(G, (base & (S["elast"] >= lo) & (S["elast"] < hi))[gs], *SIG)
                  for lo, hi in zip(eb[:-1], eb[1:])])
    errbar(ax, 0.5 * (eb[1:] + eb[:-1]), a, ea)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("energy in the last crystal [MeV]")
    ax.set_ylabel("(up − down) / (up + down)")
    ax.set_title("Michel position relative to the stop crystal", fontsize=11, loc="left", pad=16)
    stamp(ax, run)

    ax = axs[1]
    tb = np.array([1500, 2500, 3500, 5000, 7000, 10000, 14000])
    c = 0.5 * (tb[1:] + tb[:-1]) / 1000
    res = {}
    for lab, cut, col in [("E$_{last}$ 25–35 MeV", (S["elast"] >= 25) & (S["elast"] < 35), C_CLS[2]),
                          ("E$_{last}$ 35–70 MeV", (S["elast"] >= 35) & (S["elast"] < 70), C_CLS[4])]:
        s = (base & cut)[gs]
        a, ea = map(np.array, zip(*[updown(G, s, lo, hi) for lo, hi in zip(tb[:-1], tb[1:])]))
        errbar(ax, c, a, ea, color=col, label=lab)
        # a(t) = a0 + a1 (t - 4 us): any slope is a time-dependent (spin) effect, the bias is static
        mn = Minuit(LeastSquares(c, a, ea, lambda x, a0, a1: a0 + a1 * (x - 4.0)), a0=0, a1=0)
        mn.migrad()
        mn.hesse()
        res[lab] = mn
        x = np.linspace(c[0], c[-1], 50)
        ax.plot(x, mn.values["a0"] + mn.values["a1"] * (x - 4), color=col, lw=1)
        ax.text(0.02, 0.95 - 0.08 * len(res), f"{lab}: slope {mn.values['a1']:+.4f} ± {mn.errors['a1']:.4f} /µs",
                transform=ax.transAxes, fontsize=9, color=col)
    ax.set_xlabel("decay time [µs]")
    ax.set_ylabel("(up − down) / (up + down)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Time dependence (the position bias is static)", fontsize=11, loc="left", pad=16)
    stamp(ax, run)
    fig.savefig(f"{out}/23_michel_asymmetry.png")
    plt.close(fig)
    return res


# ---------------------------------------------------------------- 24
SOFT = (0.5, 8.0)       # MeV
B_N = np.arange(-20.0, 60.01, 1.0)  # us
READOUT = (1.0, 91.0)   # us, usable part of the readout window
VETO = (-0.1, 3.0)      # us around any group above 10 MeV (Michels, other cosmics)


def soft_rate(S, G, sel):
    """Soft-group rate per stop and us vs dt, live time from the readout window minus big-group vetoes."""
    gs = G["stop"]
    idx = np.flatnonzero(sel)
    ts = S["t"][idx] / 1000
    lo = np.maximum(B_N[0], READOUT[0] - ts)
    hi = np.minimum(B_N[-1], READOUT[1] - ts)
    nb = len(B_N) - 1
    live = np.zeros(nb)
    # bins fully inside [lo, hi]
    i0 = np.ceil(lo - B_N[0]).astype(int)
    i1 = np.floor(hi - B_N[0]).astype(int)
    d = np.zeros(nb + 1)
    ok = i1 > i0
    np.add.at(d, np.clip(i0[ok], 0, nb), 1)
    np.add.at(d, np.clip(i1[ok], 0, nb), -1)
    live = np.cumsum(d)[:nb]
    gsel = sel[gs]
    t = G["dt"] / 1000
    big = gsel & (G["e"] > 10) & (G["dmin"] < 150)
    # (stop, bin) pairs removed by vetoes
    vs, vb = [], []
    for k in range(int(np.floor(VETO[0])), int(np.ceil(VETO[1])) + 1):
        b = np.floor(t[big] - B_N[0]).astype(int) + k
        tb0 = B_N[0] + b
        hit = (tb0 < t[big] + VETO[1]) & (tb0 + 1 > t[big] + VETO[0]) & (b >= 0) & (b < nb)
        vs.append(gs[big][hit])
        vb.append(b[hit])
    vs, vb = np.concatenate(vs), np.concatenate(vb)
    key = np.unique(vs.astype(np.int64) * 1000 + vb)
    vs, vb = key // 1000, key % 1000
    # only bins that were live for that stop
    tsv = S["t"][vs] / 1000
    inl = (B_N[0] + vb >= np.maximum(B_N[0], READOUT[0] - tsv)) & (B_N[0] + vb + 1 <= np.minimum(B_N[-1], READOUT[1] - tsv))
    live -= np.bincount(vb[inl], minlength=nb)
    soft = gsel & (G["e"] >= SOFT[0]) & (G["e"] < SOFT[1]) & (G["dmin"] < 150)
    sb = np.floor(t[soft] - B_N[0]).astype(int)
    keep = (sb >= 0) & (sb < nb)
    ss, sb = gs[soft][keep], sb[keep]
    vetoed = np.isin(ss.astype(np.int64) * 1000 + sb, key)
    tss = S["t"][ss] / 1000
    inl = (B_N[0] + sb >= np.maximum(B_N[0], READOUT[0] - tss)) & (B_N[0] + sb + 1 <= np.minimum(B_N[-1], READOUT[1] - tss))
    n = np.bincount(sb[~vetoed & inl], minlength=nb)
    return n, live


def plot_soft(S, G, run, out):
    gs = G["stop"]
    base = (S["t"] > T_STOP[0]) & (S["t"] < T_STOP[1]) & (S["ahead"] >= AHEAD_MIN)
    # untagged: no Michel-like group 0.1-25 us (removes most mu+; mu- in CsI are almost never tagged)
    tag = np.zeros(len(S["t"]), bool)
    m = michel_groups(G, 3, 60) & (G["dt"] > 100) & (G["dt"] < 25000)
    tag[gs[m]] = True
    c = 0.5 * (B_N[1:] + B_N[:-1])
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    rates = {}
    ax = axs[0]
    for i, (lo, hi) in enumerate(zip(EL_BINS[:-1], EL_BINS[1:])):
        sel = base & ~tag & (S["elast"] >= lo) & (S["elast"] < hi)
        n, live = soft_rate(S, G, sel)
        r, er = n / np.maximum(live, 1), np.sqrt(n) / np.maximum(live, 1)
        rates[(lo, hi)] = (n, live)
        errbar(ax, c, r * 1e6, er * 1e6, color=C_CLS[i], ms=2.5, label=f"E$_{{last}}$ {lo}–{hi} MeV ({sel.sum():,})")
    ax.set_yscale("log")
    ax.axvspan(-1, 2, color="#eeeeee", lw=0)
    ax.set_xlabel("Δt soft group − muon [µs]")
    ax.set_ylabel(f"groups {SOFT[0]}–{SOFT[1]:.0f} MeV per 10⁶ stops per µs")
    ax.legend(fontsize=8)
    stamp(ax, run, "   |   no Michel tag")

    # purest minus least-pure class, scaled to equal pre-muon rate
    (nh, lh), (nl, ll) = rates[(45, 70)], rates[(0, 15)]
    pre = (c > -18) & (c < -2)
    k = (nh[pre].sum() / lh[pre].sum()) / (nl[pre].sum() / ll[pre].sum())
    rh, rl = nh / np.maximum(lh, 1), nl / np.maximum(ll, 1)
    y = rh - k * rl
    # Poisson errors with at least one count per bin
    ey = np.sqrt(np.maximum(nh, 1) / np.maximum(lh, 1) ** 2 + k * k * np.maximum(nl, 1) / np.maximum(ll, 1) ** 2)
    ax = axs[1]
    errbar(ax, c, y * 1e6, ey * 1e6)
    fitm = (c > 4) & (c < 58)    # after the post-muon burst and the soft Michel tail

    def model(x, a, tn, c0):
        return a * np.exp(-x / tn) + c0

    mn = Minuit(LeastSquares(c[fitm], y[fitm] * 1e6, ey[fitm] * 1e6, model), a=20, tn=15, c0=0)
    mn.limits["tn"] = (1, 500)
    mn.migrad()
    mn.hesse()
    x = np.linspace(4, 58, 300)
    ax.plot(x, model(x, *mn.values), color=C_FIT, lw=1.4, label="A e$^{-t/τ}$ + C, fit 4–58 µs")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvspan(-1, 4, color="#eeeeee", lw=0)
    ax.set_xlim(-20, 60)
    ax.set_ylim(-30, 150)
    ax.set_xlabel("Δt [µs]")
    ax.set_ylabel("excess per 10⁶ stops per µs")
    ax.text(0.97, 0.95, f"τ = {mn.values['tn']:.1f} ± {mn.errors['tn']:.1f} µs\n"
            f"A = {mn.values['a']:.1f} ± {mn.errors['a']:.1f}\nC = {mn.values['c0']:.2f} ± {mn.errors['c0']:.2f}\n"
            f"χ²/ndf = {mn.fval:.0f} / {fitm.sum() - 3}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox=dict(fc="white", ec="#cccccc"))
    ax.legend(fontsize=8, loc="lower right")
    stamp(ax, run, f"   |   E$_{{last}}$ 45–70 − {k:.2f} × (0–15) MeV")

    # excess 5-40 us per stop vs Michel fraction of the class
    ax = axs[2]
    mg = michel_groups(G)
    xs, exs, ys, eys = [], [], [], []
    win = (c > 5) & (c < 40)
    _, (nl, ll) = None, rates[(0, 15)]
    rl_pre = nl[pre].sum() / ll[pre].sum()
    for i, (lo, hi) in enumerate(zip(EL_BINS[:-1], EL_BINS[1:])):
        cls = base & (S["elast"] >= lo) & (S["elast"] < hi)
        nm, enm = michel_count(G, mg & cls[gs])
        xs.append(nm / cls.sum() * 100)
        exs.append(enm / cls.sum() * 100)
        n, live = rates[(lo, hi)]
        kk = (n[pre].sum() / live[pre].sum()) / rl_pre
        r = (n[win] / live[win] - kk * nl[win] / ll[win]).sum()
        er = np.sqrt((n[win] / live[win] ** 2).sum() + kk * kk * (nl[win] / ll[win] ** 2).sum())
        ys.append(r * 1e3)
        eys.append(er * 1e3)
        ax.errorbar(xs[-1], ys[-1], eys[-1], exs[-1], fmt="o", color=C_CLS[i], ms=5, label=f"E$_{{last}}$ {lo}–{hi} MeV")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Michel per candidate [%]  (∝ stop purity)")
    ax.set_ylabel("soft groups 5–40 µs per 10³ untagged candidates\n(above the 0–15 MeV class)")
    ax.legend(fontsize=8)
    stamp(ax, run)
    fig.savefig(f"{out}/24_delayed_soft.png")
    plt.close(fig)
    return mn, k


# ---------------------------------------------------------------- 25
WF_SIG = 25.0          # Delta chi2 / (chi2/ndf) for a second pulse
E2 = (1.0, 10.0)       # MeV
B_D = np.arange(20.0, 160.01, 10.0)


def load_wf(path):
    d = np.load(path)
    F = {k[2:]: d[k] for k in d.files if k.startswith("f_")}
    M = {k[2:]: d[k] for k in d.files if k.startswith("m_")}
    L = {k[2:]: d[k] for k in d.files if k.startswith("l_")}
    return F, M, L


def second_pulse(F, e2=E2):
    sig = (F["chi1"] - F["chi2"]) / (F["chi2"] / (F["ndf1"] - 1))
    both = (F["nsipm"] == 2) & (F["e2a"] > 0) & (F["e2b"] > 0)
    return (sig > WF_SIG) & both & (F["e2"] >= e2[0]) & (F["e2"] < e2[1])


def plot_capture_wf(S, G, wfpath, run, out):
    F, M, L = load_wf(wfpath)
    gs = G["stop"]
    st = F["stop"].astype(np.int64)
    tag = np.zeros(len(S["t"]), bool)
    tag[gs[michel_groups(G) & (G["dt"] > 500) & (G["dt"] < 10000)]] = True       # mu+ stops (decay after the digi)
    any3 = np.zeros(len(S["t"]), bool)
    any3[gs[michel_groups(G, 3, 60) & (G["dt"] > 100) & (G["dt"] < 25000)]] = True
    sp = second_pulse(F)
    el = S["elast"][st]
    c = 0.5 * (B_D[1:] + B_D[:-1])

    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
    ax = axs[0, 0]
    rows = {}
    for lab, m, col in [("Michel-tagged (µ⁺ stops)", tag[st], "k"),
                        ("untagged, E$_{last}$ 25–35", ~any3[st] & (el < 35), C_CLS[2]),
                        ("untagged, E$_{last}$ 35–70", ~any3[st] & (el >= 35), C_CLS[4])]:
        h, _ = np.histogram(F["delta"][m & sp], B_D)
        rows[lab] = (h, m.sum())
        errbar(ax, c, h / m.sum() * 1e3, np.sqrt(h) / m.sum() * 1e3, color=col, label=f"{lab} ({m.sum():,})")
    ax.set_xlabel("second pulse − muon pulse [ns]")
    ax.set_ylabel("second pulses per 10³ candidates per 10 ns")
    ax.legend(fontsize=8)
    stamp(ax, run, f"   |   {E2[0]:.0f}–{E2[1]:.0f} MeV, both SiPMs")

    # why the subtraction fails: stopping-muon pulses give longer digis than through-going ones at equal energy
    ax = axs[0, 1]
    nb = np.arange(40, 81, 2)
    e1w = (F["e1"] >= 30) & (F["e1"] < 45)
    for lab, m, col in [("Michel-tagged (µ⁺ stops)", tag[st], "k"), ("untagged", ~any3[st], C_CLS[2])]:
        s = m & e1w
        h, _ = np.histogram(F["nsamp"][s], nb)
        ax.stairs(h / s.sum(), nb, color=col, lw=1.4, label=f"{lab}, median {np.median(F['nsamp'][s]):.0f}")
    ax2 = ax.twinx()
    pk = sp & (F["delta"] >= 85) & (F["delta"] < 115)
    fr = [pk[(F["nsamp"] >= lo) & (F["nsamp"] < hi)].mean() if ((F["nsamp"] >= lo) & (F["nsamp"] < hi)).sum() > 100
          else np.nan for lo, hi in zip(nb[:-1], nb[1:])]
    ax2.plot(0.5 * (nb[1:] + nb[:-1]), np.array(fr) * 100, "o-", color=C_FIT, ms=3, lw=1)
    ax2.set_ylabel("second pulse at 85–115 ns [%] (all)", color=C_FIT)
    ax2.grid(False)
    ax.set_xlabel("digi length of the stop crystal [samples]")
    ax.set_ylabel("fraction (main pulse 30–45 MeV)")
    ax.legend(fontsize=8, loc="upper left")
    stamp(ax, run)
    mn = None

    # embedding efficiency
    ax = axs[1, 0]
    ms = (M["chi1"] - M["chi2"]) / (M["chi2"] / (M["ndf1"] - 1))
    rec = (ms > WF_SIG) & (np.abs(M["delta"] - M["delta_in"]) < 15) & (M["e2"] >= E2[0]) & (M["e2"] < E2[1])
    db = np.arange(10, 301, 20.0)
    for (lo, hi), col in zip([(0.3, 1), (1, 2), (2, 4), (4, 10)], C_CLS[1:]):
        eff = []
        for a0, a1 in zip(db[:-1], db[1:]):
            s = (M["e_in"] >= lo) & (M["e_in"] < hi) & (M["delta_in"] >= a0) & (M["delta_in"] < a1)
            eff.append(rec[s].mean() if s.sum() else np.nan)
        ax.plot(0.5 * (db[1:] + db[:-1]), eff, "o-", color=col, ms=3, label=f"{lo}–{hi} MeV injected")
    ax.set_xlabel("injected delay [ns]")
    ax.set_ylabel("found (same cuts, |Δδ| < 15 ns)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    stamp(ax, run, f"   |   embedding on {len(M['delta_in']):,} candidates")

    # early-pulse excess (20-80 ns) vs stop purity, in E_last bins
    ax = axs[1, 1]
    mg = michel_groups(G)
    early = sp & (F["delta"] >= 20) & (F["delta"] < 80)
    late_ = sp & (F["delta"] >= 100) & (F["delta"] < 160)
    tg_e = early[tag[st]].mean() - late_[tag[st]].mean() * 60 / 60
    for i, (lo, hi) in enumerate(zip(EL_BINS[2:-1], EL_BINS[3:])):
        cls = (S["t"] > T_STOP[0]) & (S["t"] < T_STOP[1]) & (S["ahead"] >= AHEAD_MIN) & (S["elast"] >= lo) & (S["elast"] < hi)
        nm, enm = michel_count(G, mg & cls[gs])
        u = ~any3[st] & cls[st]
        v = early[u].mean() - late_[u].mean() - tg_e
        ev = np.sqrt(early[u].sum() + late_[u].sum()) / u.sum()
        ax.errorbar(nm / cls.sum() * 100, v * 1e3, ev * 1e3, enm / cls.sum() * 100, fmt="o", ms=5,
                    color=C_CLS[i + 2], label=f"E$_{{last}}$ {lo}–{hi} MeV")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Michel per candidate [%]  (∝ stop purity)")
    ax.set_ylabel("(20–80 ns) − (100–160 ns) pulses per 10³,\nuntagged minus tagged")
    ax.set_title("No capture-like excess growing with purity", fontsize=11, loc="left", pad=16)
    ax.legend(fontsize=8)
    stamp(ax, run)
    fig.savefig(f"{out}/25_capture_waveform.png")
    plt.close(fig)
    return mn


if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    wf = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].endswith(".npz") else None
    run = sys.argv[-1] if len(sys.argv) > 3 and not sys.argv[-1].endswith(".npz") else "125329"
    os.makedirs(out, exist_ok=True)
    S, G = load(path)
    sel = stop_sel(S)
    nm, enm = michel_count(G, michel_groups(G) & sel[G["stop"]])
    print(f"candidates {len(S['t']):,}, stop sample {sel.sum():,}, Michel groups {nm:,.0f} ± {enm:.0f} "
          f"({nm / sel.sum():.2%}; / window eff {nm / sel.sum() / window_eff():.2%})")
    plot_selection(S, G, run, out)
    f = plot_lifetime_stops(S, G, run, out)
    print(f"lifetime {f['tau']:.4f} ± {f['etau']:.4f} us, chi2 {f['chi2']:.0f}/{f['ndf']}")
    fr = plot_michel_lowE(S, G, run, out)
    print("Michel fraction below threshold:", {k: round(v, 4) for k, v in fr.items()})
    res = plot_asymmetry(S, G, run, out)
    for k, m in res.items():
        print(f"asymmetry {k}: a0 {m.values['a0']:+.4f} ± {m.errors['a0']:.4f}, slope {m.values['a1']:+.4f} ± {m.errors['a1']:.4f} /us")
    mn, k = plot_soft(S, G, run, out)
    print("soft excess fit:", {p: (round(mn.values[p], 3), round(mn.errors[p], 3)) for p in mn.parameters}, "k", round(k, 3))
    if wf:
        plot_capture_wf(S, G, wf, run, out)
