"""Why reco clusters can't define a stopped muon: one cosmic is split into several same-time clusters.

For every cluster above 15 MeV, the time difference to every other cluster on the same disk (left) and, for
those within 30 ns, their distance (right).

Usage: python plot_fragmentation.py <ntuple.root> <outdir> [max_entries] [run]
"""
import sys

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from fill_calo import CL, TREE
from plot_calo import C_FIT, INK2, stamp

if __name__ == "__main__":
    path, out = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000
    run = sys.argv[4] if len(sys.argv) > 4 else "125329"
    a = uproot.open(path)[TREE].arrays([CL + v for v in ("time_", "energyDep_", "diskID_", "cog_.fCoordinates.fX",
                                                         "cog_.fCoordinates.fY")], entry_stop=nmax)
    c = ak.zip({"t": a[CL + "time_"], "e": a[CL + "energyDep_"], "d": a[CL + "diskID_"],
                "x": a[CL + "cog_.fCoordinates.fX"], "y": a[CL + "cog_.fCoordinates.fY"]})
    ij = ak.argcartesian([c.t, c.t])
    p, q = c[ij["0"]], c[ij["1"]]
    m = (ij["0"] != ij["1"]) & (p.d == q.d) & (p.e > 15)
    dt = ak.flatten((q.t - p.t)[m]).to_numpy()
    dr = ak.flatten(np.hypot(q.x - p.x, q.y - p.y)[m]).to_numpy()
    frac = ak.sum(ak.any(m & (abs(q.t - p.t) < 30), axis=-1)) / ak.sum(c.e > 15)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    ax = axs[0]
    b = np.arange(-300, 301, 5.0)
    ax.stairs(np.histogram(dt, b)[0], b, color="k", lw=1.2)
    ax.axvspan(-30, 30, color="#f2c4b3", alpha=0.5, lw=0, label="|Δt| < 30 ns")
    ax.set_yscale("log")
    ax.set_xlabel("t(other cluster) − t(muon cluster) [ns], same disk")
    ax.set_ylabel("cluster pairs per 5 ns")
    ax.text(0.03, 0.95, f"{frac:.0%} of clusters > 15 MeV have\na same-disk partner within 30 ns",
            transform=ax.transAxes, va="top", fontsize=10, bbox=dict(fc="white", ec="#cccccc"))
    ax.legend(loc="upper right", fontsize=9)
    stamp(ax, run, f"   |   {nmax:,} events")
    ax = axs[1]
    s = np.abs(dt) < 30
    b = np.arange(0, 801, 20.0)
    ax.stairs(np.histogram(dr[s], b)[0], b, color=C_FIT, lw=1.4)
    ax.axvline(34.3, color=INK2, ls=":", lw=1)
    ax.text(38, ax.get_ylim()[1] * 0.9, "1 crystal", fontsize=9, color=INK2)
    ax.set_xlabel("distance between the two cluster centres [mm]")
    ax.set_ylabel("pairs with |Δt| < 30 ns per 20 mm")
    stamp(ax, run)
    fig.savefig(f"{out}/26_cluster_fragmentation.png")
    plt.close(fig)
    print(f"fraction with a partner within 30 ns: {frac:.3f}; median distance {np.median(dr[s]):.0f} mm")
