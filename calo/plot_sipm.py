"""Plots from fill_sipm.py: SiPM timing/gain balance, shifted-SiPM side list, run stability.

Usage: python plot_sipm.py <sipm.npz> <sipm_offsets_k.txt> <outdir> [run]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from plot_calo import C_DISK, C_FIT, INK2, errbar, stamp


def gaus(x, a, m, s):
    return a * np.exp(-0.5 * ((x - m) / s) ** 2)


def core_fit(c, y, half):
    i = np.argmax(y)
    r = (c > c[i] - half) & (c < c[i] + half)
    p, cov = curve_fit(gaus, c[r], y[r], p0=[y.max(), c[i], half / 2], sigma=np.sqrt(np.maximum(y[r], 1)))
    return p, np.sqrt(np.diag(cov))


if __name__ == "__main__":
    D = np.load(sys.argv[1])
    kt = np.loadtxt(sys.argv[2])
    out = sys.argv[3]
    run = sys.argv[4] if len(sys.argv) > 4 else "125329"
    os.makedirs(out, exist_ok=True)
    k = kt[:, 2]
    cls = np.where(~np.isfinite(k), 0, np.where(k == 0, 1, 2))
    cls_lab = ["SiPMs merge fine", "k = 0 (~10 ns split)", "|k| ≥ 1 (k × 169 ns)"]

    # ---------------- timing and gain
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    ax = axs[0]
    B, EH = D["B_DTS"], D["B_EH"]
    c = 0.5 * (B[1:] + B[:-1])
    ec, sig, esig = [], [], []
    for j in range(len(EH) - 1):
        p, e = core_fit(c, D["dt_e"][:, j], 3.0)
        ec.append(np.sqrt(EH[j] * EH[j + 1]))
        sig.append(abs(p[2]))
        esig.append(e[2])
    ec, sig = np.array(ec), np.array(sig)
    errbar(ax, ec, sig, esig, label="σ core of t(odd) − t(even)")
    errbar(ax, ec, sig / 2, np.array(esig) / 2, color=C_FIT, label="→ crystal time resolution (σ/2)")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50, 100], ["3", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("hit energy [MeV]")
    ax.set_ylabel("σ [ns]")
    ax.set_ylim(0, None)
    ax.legend(fontsize=9)
    stamp(ax, run, "   |   hits with both SiPMs")
    i_mip = np.searchsorted(EH, 20) - 1
    ax.text(0.97, 0.6, f"at {EH[i_mip]:.0f}–{EH[i_mip + 1]:.0f} MeV:\nσ/2 = {sig[i_mip] / 2:.2f} ns",
            transform=ax.transAxes, ha="right", fontsize=10, color=C_FIT)

    ax = axs[1]
    dm = D["dtmean"][:, 0] / np.maximum(D["dtmean"][:, 1], 1)
    nm = D["dtmean"][:, 1]
    bins = np.linspace(-12, 12, 97)
    for i, col in [(0, "k"), (1, C_DISK[1])]:
        v = dm[(cls == i) & (nm > 100)]
        ax.hist(v, bins, histtype="step", lw=1.5, color=col,
                label=f"{cls_lab[i]}\n{len(v)} crystals, RMS {v.std():.2f} ns")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 600)
    ax.set_xlabel("per-crystal mean t(odd) − t(even), merged hits 15–40 MeV [ns]")
    ax.set_ylabel("crystals")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("~10 ns SiPM-to-SiPM offsets stop the hit builder merging", fontsize=10, loc="left")

    ax = axs[2]
    A, Ba = D["asy"], D["B_ASY"]
    ca = 0.5 * (Ba[1:] + Ba[:-1])
    na = A.sum(1)
    ma = (A * ca).sum(1) / np.maximum(na, 1)
    ok = na > 200
    ax.hist(ma[ok], np.linspace(-0.4, 0.4, 81), histtype="stepfilled", color="#888888", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-crystal mean (E$_{odd}$ − E$_{even}$)/(E$_{odd}$ + E$_{even}$), MIP hits")
    ax.set_ylabel("crystals")
    ax.set_title(f"SiPM gain balance: spread {ma[ok].std():.3f}, {np.sum(np.abs(ma[ok]) > 0.1)} crystals beyond ±0.1",
                 fontsize=10, loc="left")
    fig.savefig(f"{out}/17_sipm_timing_gain.png")
    plt.close(fig)

    # ---------------- which SiPM is off for shifted crystals
    on = D["ontime"]
    tot = on.sum(1)
    fe = on[:, 0] / np.maximum(tot, 1)
    sh = cls == 2
    wrong = np.full(len(k), -1)
    wrong[sh & (tot >= 20) & (fe > 0.9)] = 1   # even on time -> odd SiPM shifted
    wrong[sh & (tot >= 20) & (fe < 0.1)] = 0   # odd on time -> even SiPM shifted
    rows = [(cid, 2 * cid + wrong[cid], kt[cid, 1], k[cid], int(on[cid, 0]), int(on[cid, 1]))
            for cid in np.flatnonzero(sh)]
    with open(f"{out}/shifted_sipms.txt", "w") as f:
        f.write("# crystalId  shifted_SiPMID(-1 = ambiguous)  dt_odd_minus_even_ns  k  n_even_ontime  n_odd_ontime\n")
        for cid, s, dt, kk, ne, no in rows:
            f.write(f"{cid:5d} {s if wrong[cid] >= 0 else -1:6d} {dt:9.1f} {int(kk):+3d} {ne:7d} {no:7d}\n")

    # ---------------- run stability
    S, SE, BE = D["sub_n"], D["sub_e"], D["B_ESUB"]
    used = np.flatnonzero(S[:, 0] > 0)
    ce = 0.5 * (BE[1:] + BE[:-1])
    mip, emip = np.full(len(used), np.nan), np.full(len(used), np.nan)
    for i, s in enumerate(used):
        try:
            p, e = core_fit(ce, SE[s], 4.0)
            mip[i], emip[i] = p[1], e[1]
        except RuntimeError:
            pass
    fig, axs = plt.subplots(5, 1, figsize=(12, 11.5), sharex=True, layout="constrained")
    series = [
        (mip, emip, "MIP peak, merged hits [MeV]"),
        (S[used, 1] / S[used, 0], np.sqrt(S[used, 1]) / S[used, 0], "clusters / event"),
        (S[used, 3] / S[used, 2], None, "single-SiPM hit fraction"),
        (S[used, 0], None, "events / subrun"),
        (S[used, 1], None, "clusters / subrun"),
    ]
    for ax, (y, ey, lab) in zip(axs, series):
        if ey is None:
            ax.plot(used, y, ".", ms=3, color="k")
        else:
            errbar(ax, used, y, ey, ms=2)
        med = np.nanmedian(y)
        ax.axhline(med, color=C_FIT, lw=0.8)
        ax.set_ylabel(lab, fontsize=9)
        ax.text(0.995, 0.93, f"median {med:.4g}, RMS/median {np.nanstd(y) / med:.2%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=INK2,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    axs[-1].set_xlabel("subrun (≈ 1 min each over the 10.8 h run)")
    stamp(axs[0], run)
    fig.savefig(f"{out}/18_run_stability.png")
    plt.close(fig)

    print(f"time resolution sigma/2 at {EH[i_mip]:.0f}-{EH[i_mip + 1]:.0f} MeV: {sig[i_mip] / 2:.2f} ns; "
          f">40 MeV: {sig[-2] / 2:.2f} ns")
    print(f"shifted crystals: {sh.sum()}, even SiPM shifted {np.sum(wrong[sh] == 0)}, "
          f"odd SiPM shifted {np.sum(wrong[sh] == 1)}, ambiguous {np.sum(wrong[sh] < 0)}")
    print(f"MIP per subrun: median {np.nanmedian(mip):.2f}, RMS {np.nanstd(mip):.3f} MeV, "
          f"min {np.nanmin(mip):.2f}, max {np.nanmax(mip):.2f}")
