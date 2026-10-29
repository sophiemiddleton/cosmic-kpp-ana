"""Echo clusters = the two SiPMs of a crystal read out with a time offset of k x ~169 ns.

Input: sipm_offsets.py output (per-crystal dt histograms) and the ntuple (for an example waveform pair).
Usage: python echo_study.py <ntuple.root> <sipm_offsets.npz> <calo.npz> <outdir> [run]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from plot_calo import C_DISK, INK2, crystal_map, stamp

STEP = 169.0  # ns, spacing of the offsets
NCRY = 1348


def offsets(S):
    c = 0.5 * (S["B"][1:] + S["B"][:-1])
    n = S["hist"].sum(1)
    pk = np.where(n > 20, c[np.argmax(S["hist"], 1)], np.nan)
    return c, pk, np.round(pk / STEP)


def find_example(path, k_cry, kwant=5, emin=20.0, max_events=200000):
    """First crystal with offset kwant where both SiPMs fired as single hits kwant*STEP apart."""
    t = uproot.open(path)["EventNtuple/ntuple"]
    H, R, D = "calohits/calohits.", "calorecodigis/calorecodigis.", "calodigis/calodigis."
    for a in t.iterate([H + x for x in ("crystalId_", "time_", "eDep_", "nSiPMs_", "recoDigis_")]
                       + [R + "SiPMID_", R + "caloDigiIdx_", D + "t0_", D + "waveform_"],
                       entry_stop=max_events, step_size=20000):
        for ev in a:
            cry = ev[H + "crystalId_"].to_numpy()
            tt = ev[H + "time_"].to_numpy()
            e = ev[H + "eDep_"].to_numpy()
            ns = ev[H + "nSiPMs_"].to_numpy()
            for i in np.flatnonzero((ns == 1) & (e > emin)):
                if k_cry[cry[i]] != kwant:
                    continue
                r_i = ev[H + "recoDigis_"][i][0]
                if ev[R + "SiPMID_"][r_i] % 2:
                    continue
                for j in np.flatnonzero((cry == cry[i]) & (ns == 1)):
                    r_j = ev[H + "recoDigis_"][j][0]
                    if ev[R + "SiPMID_"][r_j] % 2 and abs(tt[j] - tt[i] - kwant * STEP) < 20:
                        out = []
                        for r in (r_i, r_j):
                            d = ev[R + "caloDigiIdx_"][r]
                            w = ev[D + "waveform_"][d].to_numpy()
                            out.append((int(ev[R + "SiPMID_"][r]), 5.0 * (ev[D + "t0_"][d] + np.arange(len(w))), w))
                        return int(cry[i]), (tt[i], tt[j]), (e[i], e[j]), out
    return None


if __name__ == "__main__":
    path, sfile, cfile, out = sys.argv[1:5]
    run = sys.argv[5] if len(sys.argv) > 5 else "125329"
    os.makedirs(out, exist_ok=True)
    S = np.load(sfile)
    Cc = np.load(cfile)
    c, pk, k = offsets(S)
    ncry = len(k)
    disk = (np.arange(ncry) >= ncry // 2).astype(int)

    fig = plt.figure(figsize=(17, 9.5), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    # (a) dt between the two SiPMs of the same crystal
    ax = fig.add_subplot(gs[0, 0:2])
    tot = S["hist"].sum(0)
    ax.stairs(tot, S["B"], color="k", lw=0.9)
    for kk in range(-7, 8):
        if kk:
            ax.axvline(kk * STEP, color=C_DISK[1], lw=0.7, ls=":", zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(-1400, 1400)
    ax.set_xlabel("t(odd SiPM) − t(even SiPM) of the same crystal, separate single-SiPM hits [ns]")
    ax.set_ylabel("pairs per 5 ns")
    ax.text(0.01, 0.95, f"dotted: k × {STEP:.0f} ns", transform=ax.transAxes, color=C_DISK[1], fontsize=9, va="top")
    stamp(ax, run)

    # (b) 2-SiPM fraction vs offset class
    ax = fig.add_subplot(gs[0, 2])
    nb, nsg = S["nboth"], S["nsingle"].sum(1)
    f2 = nb / np.maximum(nb + nsg, 1)
    cls = np.where(~np.isfinite(k), 0, np.where(k == 0, 1, 2))
    labels = ["no split pairs\n(merging OK)", "split by ~10 ns\n(k = 0)", "split by k × 169 ns\n(|k| ≥ 1)"]
    data = [f2[cls == i] for i in range(3)]
    ax.boxplot(data, showfliers=False, widths=0.5)
    ax.set_xticks([1, 2, 3], [f"{l}\n{len(d)} crystals" for l, d in zip(labels, data)], fontsize=9)
    ax.set_ylabel("fraction of hits with both SiPMs")
    ax.set_ylim(0, 1)

    # (c, d) maps of k per disk
    cmap = LinearSegmentedColormap.from_list("div", [C_DISK[0], "#e8e8e8", C_DISK[1]])
    bounds = np.arange(-7.5, 8.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    H = {"cry_x": Cc["cry_x"], "cry_y": Cc["cry_y"]}
    for d in (0, 1):
        ax = fig.add_subplot(gs[1, d])
        pc = crystal_map(ax, H, k.astype(float), disk == d, cmap, -7.5, 7.5)
        pc.set_norm(norm)
        ax.set_title(f"disk {d}: SiPM offset k  ({int(np.sum(np.abs(k[disk == d]) >= 1))} crystals with |k| ≥ 1)",
                     fontsize=10, loc="left")
    fig.colorbar(pc, ax=fig.axes[-1], ticks=range(-7, 8), label="k = offset / 169 ns  (grey: not measured)",
                 shrink=0.9)

    # (e) example waveforms
    ax = fig.add_subplot(gs[1, 2])
    ex = find_example(path, k, kwant=5)
    if ex:
        cry, (ti, tj), (ei, ej), wfs = ex
        dtk = tj - ti
        (s0, x0, w0), (s1, x1, w1) = wfs
        ax.plot(x0 - ti, w0 - w0[:4].mean(), color=C_DISK[0], lw=1.6, label=f"SiPM {s0} (even): hit {ei:.1f} MeV")
        ax.plot(x1 - ti, w1 - w1[:4].mean(), color=C_DISK[1], lw=1.6,
                label=f"SiPM {s1} (odd): hit {ej:.1f} MeV, +{dtk:.0f} ns")
        ax.plot(x1 - ti - dtk, w1 - w1[:4].mean(), color=C_DISK[1], lw=1.2, ls="--",
                label=f"odd SiPM shifted back by {dtk:.0f} ns")
        ax.set_xlabel("time − even-SiPM hit time [ns]")
        ax.set_ylabel("ADC − pedestal")
        ax.set_title(f"crystal {cry}: one pulse, two SiPMs, k = 5", fontsize=10, loc="left")
        ax.legend(fontsize=8.5, loc="upper right")
    fig.savefig(f"{out}/15_sipm_time_offsets.png")
    plt.close(fig)

    print(f"crystals measured: {np.isfinite(k).sum()}/{ncry}; |k|>=1: {int(np.sum(np.abs(k) >= 1))}; k=0: {int(np.sum(k == 0))}")
    for i, lab in enumerate(["not measured", "k=0", "|k|>=1"]):
        print(f"  {lab:13s} median 2-SiPM fraction {np.median(data[i]):.2f}")
    np.savetxt(f"{out}/sipm_offsets_k.txt", np.column_stack([np.arange(ncry), pk, k]), fmt=["%d", "%.1f", "%.0f"],
               header="crystalId modal_dt_odd_minus_even_ns k")
    if ex:
        print(f"example: crystal {ex[0]}, dt = {ex[1][1] - ex[1][0]:.1f} ns, E = {ex[2][0]:.1f} / {ex[2][1]:.1f} MeV")
