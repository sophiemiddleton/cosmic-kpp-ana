"""Hit-level search for mu- capture in CsI: delayed crystal hits near a prompt (muon) cluster.

For each hit j on the same disk and not in the cluster, the prompt cluster is the one (E > PROMPT_EMIN)
with a crystal closest to j. dt = t_hit - t_cluster, d = distance from j's crystal to that nearest
prompt crystal. Histograms in dt x d-class x E_hit x nSiPMs x (j in any cluster).

Usage: python fill_capture.py <ntuple.root> <out.npz> [nworkers] [max_entries]
"""
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np
import uproot

from fill_calo import CL, HT, TREE

PROMPT_EMIN = 15.0  # MeV
NCRY = 1348

B_DT = np.arange(-5000, 25000 + 1, 10.0)                                   # ns
B_D = np.array([-0.5, 0.5, 50, 100, 200, 400])                             # same / adjacent / 2nd ring / gap / far
B_E = np.array([0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 60, 1e4])             # MeV
B_NS = np.array([0.5, 1.5, 4.5])                                           # 1 SiPM / 2 SiPMs
B_IN = np.array([-0.5, 0.5, 1.5])                                          # not in a cluster / in a cluster
BINS = [B_DT, B_D, B_E, B_NS, B_IN]


def process(rng):
    start, stop, path = rng
    t = uproot.open(path)[TREE]
    a = t.arrays([CL + "time_", CL + "energyDep_"]
                 + [HT + v for v in ("crystalId_", "eDep_", "time_", "clusterIdx_", "nSiPMs_",
                                     "crystalPos_.fCoordinates.fX", "crystalPos_.fCoordinates.fY")],
                 entry_start=start, entry_stop=stop)
    ct, ce = a[CL + "time_"], a[CL + "energyDep_"]
    h = ak.zip({"cry": a[HT + "crystalId_"], "e": a[HT + "eDep_"], "t": a[HT + "time_"],
                "cl": a[HT + "clusterIdx_"], "ns": a[HT + "nSiPMs_"],
                "x": a[HT + "crystalPos_.fCoordinates.fX"], "y": a[HT + "crystalPos_.fCoordinates.fY"]})
    nprompt = int(ak.sum(ce > PROMPT_EMIN))
    # pad so index 0 exists in events without clusters (unclustered hits, cl = -1, point there)
    ctp = ak.fill_none(ak.pad_none(ct, 1), np.nan)
    cep = ak.fill_none(ak.pad_none(ce, 1), 0.0)
    clsafe = ak.where(h.cl >= 0, h.cl, 0)
    h = ak.with_field(h, ak.where(h.cl >= 0, ctp[clsafe], np.nan), "tc")
    h = ak.with_field(h, (h.cl >= 0) & (cep[clsafe] > PROMPT_EMIN), "prompt")
    h = ak.with_field(h, h.cry >= NCRY // 2, "disk")

    ij = ak.argcartesian([h.e, h.e])
    p, q = h[ij["0"]], h[ij["1"]]
    dt = q.t - p.tc
    m = p.prompt & (q.cl != p.cl) & (q.disk == p.disk) & (dt > B_DT[0]) & (dt < B_DT[-1])
    d = np.hypot(q.x - p.x, q.y - p.y)
    evt = ak.broadcast_arrays(ak.local_index(h.e, axis=0), ij["0"])[0]
    cols = {"evt": evt[m], "j": ij["1"][m], "d": d[m], "dt": dt[m], "e": q.e[m], "ns": q.ns[m],
            "inc": ak.values_astype(q.cl[m] >= 0, np.int8)}
    cols = {k: ak.flatten(v).to_numpy() for k, v in cols.items()}
    # each delayed hit once, with its nearest prompt-cluster crystal
    o = np.lexsort((cols["d"], cols["j"], cols["evt"]))
    cols = {k: v[o] for k, v in cols.items()}
    key = cols["evt"].astype(np.int64) * 100000 + cols["j"]
    first = np.r_[True, key[1:] != key[:-1]]
    x = np.column_stack([cols["dt"][first], cols["d"][first], cols["e"][first],
                         cols["ns"][first], cols["inc"][first]])
    hist, _ = np.histogramdd(x, BINS)
    return hist.astype(np.int64), nprompt


if __name__ == "__main__":
    path, outp = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    n = uproot.open(path)[TREE].num_entries
    if len(sys.argv) > 4:
        n = min(n, int(sys.argv[4]))
    step = 100_000
    ranges = [(s, min(s + step, n), path) for s in range(0, n, step)]
    tot, npr = 0, 0
    with Pool(nw) as pool:
        for k, (hh, nn) in enumerate(pool.imap_unordered(process, ranges)):
            tot = tot + hh
            npr += nn
            print(f"{k + 1}/{len(ranges)} chunks", flush=True)
    np.savez_compressed(outp, hist=tot, nprompt=npr, nev=n,
                        B_DT=B_DT, B_D=B_D, B_E=B_E, B_NS=B_NS, B_IN=B_IN)
    print("events", n, "prompt clusters", npr, "delayed hits histogrammed", tot.sum())
