"""Calo event displays of stopped-muon decay candidates selected from fill_calo.py pairs.

Usage: python event_display.py <ntuple.root> <calo.npz> <outdir> [n=10] [seed=1] [run]
"""
import os
import sys

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

from fill_calo import CL, CL_BR, HT, HT_BR, TREE
from plot_calo import CRY_PITCH, DR_CUT, INK2, stamp

C_DEL = "#d62728"
EMAX = 60.0          # MeV, colour scale top (as on the slide)
PROMPT_WIN = 25.0    # ns, hits within this of the prompt cluster time are drawn
DT_SEL = (1.25, 10.0)
ED_SEL = (15.0, 55.0)


def pick(P, n, seed):
    dt = P["dt"] / 1000.0
    s = ((P["dr"] < DR_CUT) & (dt > DT_SEL[0]) & (dt < DT_SEL[1])
         & (P["ed"] > ED_SEL[0]) & (P["ed"] < ED_SEL[1]) & (P["ep"] > 15))
    idx = np.flatnonzero(s)
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def read_event(t, entry):
    a = t.arrays([CL + v for v in CL_BR.values()] + [HT + v for v in HT_BR.values()]
                 + ["evtinfo/run", "evtinfo/subrun", "evtinfo/event"],
                 entry_start=entry, entry_stop=entry + 1)
    c = {k: a[CL + v][0].to_numpy() for k, v in CL_BR.items()}
    h = {k: a[HT + v][0].to_numpy() for k, v in HT_BR.items()}
    eid = tuple(int(a[f"evtinfo/{k}"][0]) for k in ("run", "subrun", "event"))
    return c, h, eid


def prompt_index(c, di):
    """Same choice as fill_calo.py: closest earlier cluster on the same disk."""
    dt = c["t"][di] - c["t"]
    dr = np.hypot(c["x"] - c["x"][di], c["y"] - c["y"][di])
    ok = (c["disk"] == c["disk"][di]) & (dt > 20) & (dt < 25000) & (dr < 150)
    return np.flatnonzero(ok)[np.argmin(dr[ok])]


def draw(ax, H, c, h, di, pi, eid, cmap, norm):
    disk = c["disk"][di]
    ncry = len(H["cry_x"])
    hdisk = (h["cry"] >= ncry // 2).astype(int) == disk
    cm = (np.arange(ncry) >= ncry // 2).astype(int) == disk
    ax.add_collection(PatchCollection(
        [Rectangle((x - CRY_PITCH / 2, y - CRY_PITCH / 2), CRY_PITCH, CRY_PITCH)
         for x, y in zip(H["cry_x"][cm], H["cry_y"][cm])],
        facecolor="none", edgecolor="#d0d0d0", linewidth=0.4))

    # muon (prompt-time) hits filled by energy; the delayed cluster is outline-only so the
    # crystal where the muon stopped, often the same one, stays visible underneath
    prompt = hdisk & (np.abs(h["t"] - c["t"][pi]) < PROMPT_WIN)
    pc = PatchCollection(
        [Rectangle((x - CRY_PITCH / 2, y - CRY_PITCH / 2), CRY_PITCH, CRY_PITCH)
         for x, y in zip(h["x"][prompt], h["y"][prompt])],
        cmap=cmap, norm=norm, edgecolor="#333333", linewidth=0.4, zorder=2)
    pc.set_array(h["e"][prompt])
    ax.add_collection(pc)
    s = CRY_PITCH - 6
    ax.add_collection(PatchCollection(
        [Rectangle((x - s / 2, y - s / 2), s, s) for x, y in zip(h["x"][h["cl"] == di], h["y"][h["cl"] == di])],
        facecolor="none", edgecolor=C_DEL, linewidth=1.6, zorder=3))

    ax.plot(c["x"][pi], c["y"][pi], "x", color="k", ms=9, mew=2.2, zorder=4)
    ax.add_patch(Circle((c["x"][di], c["y"][di]), 60, fill=False, ec=C_DEL, lw=1.8, zorder=4))
    dt = (c["t"][di] - c["t"][pi]) / 1000.0
    dr = np.hypot(c["x"][di] - c["x"][pi], c["y"][di] - c["y"][pi])
    ax.text(0.5, 0.5,
            f"disk {disk}\nΔt = {dt:.2f} µs\nΔR = {dr:.0f} mm\n"
            f"E$_{{prompt}}$ = {c['e'][pi]:.1f} MeV\nE$_{{delayed}}$ = {c['e'][di]:.1f} MeV",
            transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#222222")
    ax.set_title(f"{eid[0]}:{eid[1]}:{eid[2]}", fontsize=9, loc="left", color=INK2)
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_aspect("equal")
    ax.grid(False)


if __name__ == "__main__":
    path, npz, out = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    run = sys.argv[6] if len(sys.argv) > 6 else "125329"
    os.makedirs(out, exist_ok=True)
    H = np.load(npz)
    P = {k[5:]: H[k] for k in H.files if k.startswith("pair_")}
    sel = pick(P, n, seed)
    t = uproot.open(path)[TREE]
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(0, EMAX)

    ncol = 5
    nrow = int(np.ceil(n / ncol))
    grid, gaxs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow), layout="constrained")
    gaxs = np.atleast_1d(gaxs).ravel()
    for k, i in enumerate(sel):
        entry, di = int(P["evt"][i]), int(P["di"][i])
        c, h, eid = read_event(t, entry)
        pi = prompt_index(c, di)
        print(f"entry {entry} {eid[0]}:{eid[1]}:{eid[2]}  delayed cl {di} prompt cl {pi}  "
              f"dt {(c['t'][di] - c['t'][pi]) / 1000:.2f} us  Ed {c['e'][di]:.1f}  Ep {c['e'][pi]:.1f}")

        fig, ax = plt.subplots(figsize=(6.6, 5.6), layout="constrained")
        draw(ax, H, c, h, di, pi, eid, cmap, norm)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm] (up)")
        ax.set_title("", loc="left")
        stamp(ax, run, f"   |   event {eid[1]}:{eid[2]}   |   entry {entry}")
        fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=ax, label="crystal energy [MeV]")
        fig.savefig(f"{out}/decay_{k:02d}_evt{eid[2]}.png")
        plt.close(fig)

        draw(gaxs[k], H, c, h, di, pi, eid, cmap, norm)
        gaxs[k].set_xticks([])
        gaxs[k].set_yticks([])
    for ax in gaxs[len(sel):]:
        ax.set_visible(False)
    grid.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=gaxs.tolist(), shrink=0.6,
                  label="crystal energy [MeV]")
    grid.suptitle(f"Stopped-muon decay candidates, run {run}:  prompt-time hits (±{PROMPT_WIN:.0f} ns), "
                  f"delayed-cluster crystals outlined in red, × = prompt-cluster COG", fontsize=11)
    grid.savefig(f"{out}/decay_grid.png", dpi=130)
