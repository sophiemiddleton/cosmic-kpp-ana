"""One pass over a calo EventNtuple: cluster/hit histograms + stopped-muon pair candidates.

Usage: python fill_calo.py <ntuple.root> <out.npz> [nworkers] [max_entries]
"""
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np
import uproot

TREE = "EventNtuple/ntuple"
CL = "caloclusters/caloclusters."
HT = "calohits/calohits."
CL_BR = {"disk": "diskID_", "t": "time_", "e": "energyDep_", "x": "cog_.fCoordinates.fX",
         "y": "cog_.fCoordinates.fY", "size": "size_", "e1": "e1_", "e9": "e9_"}
HT_BR = {"cry": "crystalId_", "e": "eDep_", "t": "time_", "cl": "clusterIdx_",
         "x": "crystalPos_.fCoordinates.fX", "y": "crystalPos_.fCoordinates.fY"}

NCRY = 1348
# pair candidate window (wider than the analysis cuts so they can be varied later)
DT_MIN, DT_MAX, DR_MAX = 20.0, 25000.0, 150.0  # ns, ns, mm

# histogram binnings
B_CLE = np.linspace(0, 500, 501)
B_HTE = np.linspace(0, 100, 201)
B_SIZE = np.arange(0.5, 60.5, 1)
B_NCL = np.arange(-0.5, 80.5, 1)
B_T = np.linspace(0, 100000, 501)
B_XY = np.linspace(-700, 700, 141)
B_DTALL = np.linspace(-25000, 25000, 1001)


def empty():
    return {
        "nev": 0,
        "cl_e": np.zeros((2, len(B_CLE) - 1)),
        "cl_size": np.zeros((2, len(B_SIZE) - 1)),
        "cl_ncl": np.zeros(len(B_NCL) - 1),
        "cl_t": np.zeros(len(B_T) - 1),
        "cl_xy": np.zeros((2, len(B_XY) - 1, len(B_XY) - 1)),
        "cl_e_size": np.zeros((len(B_CLE) - 1, len(B_SIZE) - 1)),
        "ht_e": np.zeros((NCRY, len(B_HTE) - 1)),
        "ht_t": np.zeros(len(B_T) - 1),
        "cry_x": np.full(NCRY, np.nan),
        "cry_y": np.full(NCRY, np.nan),
        "pairs": [],
    }


def process(rng):
    start, stop, path = rng
    h = empty()
    t = uproot.open(path)[TREE]
    a = t.arrays([CL + v for v in CL_BR.values()] + [HT + v for v in HT_BR.values()],
                 entry_start=start, entry_stop=stop)
    c = ak.zip({k: a[CL + v] for k, v in CL_BR.items()})
    hh = ak.zip({k: a[HT + v] for k, v in HT_BR.items()})
    h["nev"] = len(c)

    # ---- cluster-level
    f = {k: ak.flatten(c[k]).to_numpy() for k in CL_BR}
    for d in (0, 1):
        m = f["disk"] == d
        h["cl_e"][d] = np.histogram(f["e"][m], B_CLE)[0]
        h["cl_size"][d] = np.histogram(f["size"][m], B_SIZE)[0]
        h["cl_xy"][d] = np.histogram2d(f["x"][m], f["y"][m], [B_XY, B_XY])[0]
    h["cl_ncl"] = np.histogram(ak.num(c).to_numpy(), B_NCL)[0]
    h["cl_t"] = np.histogram(f["t"], B_T)[0]
    h["cl_e_size"] = np.histogram2d(f["e"], f["size"], [B_CLE, B_SIZE])[0]

    # ---- hit-level: per-crystal energy spectra
    g = {k: ak.flatten(hh[k]).to_numpy() for k in HT_BR}
    ok = (g["cry"] >= 0) & (g["cry"] < NCRY)
    np.add.at(h["ht_e"], (g["cry"][ok], np.clip(np.digitize(g["e"][ok], B_HTE) - 1, 0, len(B_HTE) - 2)),
              (g["e"][ok] < B_HTE[-1]))
    h["ht_t"] = np.histogram(g["t"], B_T)[0]
    u, i = np.unique(g["cry"][ok], return_index=True)
    h["cry_x"][u] = g["x"][ok][i]
    h["cry_y"][u] = g["y"][ok][i]

    # ---- prompt/delayed pairs on the same disk
    ij = ak.argcartesian([c.t, c.t])
    p, d = c[ij["0"]], c[ij["1"]]
    dt = d.t - p.t
    dr = np.hypot(d.x - p.x, d.y - p.y)
    m = (d.disk == p.disk) & (dt > DT_MIN) & (dt < DT_MAX) & (dr < DR_MAX)
    evt = ak.broadcast_arrays(ak.local_index(c.t, axis=0) + start, ij["0"])[0]
    cols = {
        "evt": evt[m], "di": ij["1"][m], "dt": dt[m], "dr": dr[m], "disk": d.disk[m],
        "ed": d.e[m], "ep": p.e[m], "sd": d.size[m], "sp": p.size[m], "xd": d.x[m], "yd": d.y[m],
    }
    cols = {k: ak.flatten(v).to_numpy() for k, v in cols.items()}
    # each delayed cluster is kept once, paired with its closest (in space) earlier cluster
    o = np.lexsort((cols["dr"], cols["di"], cols["evt"]))
    cols = {k: v[o] for k, v in cols.items()}
    key = cols["evt"].astype(np.int64) * 1000 + cols["di"]
    first = np.r_[True, key[1:] != key[:-1]]
    h["pairs"] = {k: v[first] for k, v in cols.items()}
    # all same-disk dt (both signs) without the uniqueness step: accidental-rate reference
    mall = (d.disk == p.disk) & (dr < 80) & (ij["0"] != ij["1"])
    h["dt_all"] = np.histogram(ak.flatten(dt[mall]).to_numpy(), B_DTALL)[0]
    return h


def merge(hs):
    out = empty()
    out["dt_all"] = 0
    for h in hs:
        for k, v in h.items():
            if k == "pairs":
                continue
            if k.startswith("cry_"):
                out[k] = np.where(np.isnan(out[k]), v, out[k])
            else:
                out[k] = out[k] + v
    pairs = {k: np.concatenate([h["pairs"][k] for h in hs]) for k in hs[0]["pairs"]}
    del out["pairs"]
    out.update({"pair_" + k: v for k, v in pairs.items()})
    return out


if __name__ == "__main__":
    path, outp = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n = uproot.open(path)[TREE].num_entries
    if len(sys.argv) > 4:
        n = min(n, int(sys.argv[4]))
    step = 200_000
    ranges = [(s, min(s + step, n), path) for s in range(0, n, step)]
    hs = []
    with Pool(nw) as pool:
        for k, h in enumerate(pool.imap_unordered(process, ranges)):
            hs.append(h)
            print(f"{k + 1}/{len(ranges)} chunks", flush=True)
    out = merge(hs)
    np.savez_compressed(outp, **out,
                        B_CLE=B_CLE, B_HTE=B_HTE, B_SIZE=B_SIZE, B_NCL=B_NCL, B_T=B_T, B_XY=B_XY,
                        B_DTALL=B_DTALL)
    print("events", out["nev"], "pairs", len(out["pair_dt"]))
