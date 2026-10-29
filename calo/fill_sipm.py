"""Per-SiPM checks and run stability in one pass.

- 2-SiPM hits: dt(odd - even) vs hit energy, per-crystal energy asymmetry (Eodd - Eeven)/(Eodd + Eeven)
- shifted crystals: which SiPM side is on time (single-SiPM hit sharing a cluster with a merged 2-SiPM hit
  of a crystal that is not shifted)
- per subrun: events, clusters, hits, single-SiPM hits, 2-SiPM hit-energy spectrum

Usage: python fill_sipm.py <ntuple.root> <sipm_offsets_k.txt> <out.npz> [nworkers] [max_entries]
"""
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np
import uproot

from fill_calo import TREE

H, R, C = "calohits/calohits.", "calorecodigis/calorecodigis.", "caloclusters/caloclusters."
NCRY = 1348
NSUB = 1000
B_DTS = np.arange(-30, 30.01, 0.25)                        # ns
B_EH = np.array([2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 60, 100])  # MeV, for dt vs E
B_ASY = np.linspace(-1, 1, 101)
B_ESUB = np.arange(0, 60.01, 0.5)                           # MeV, per-subrun hit spectrum
KFILE = None


def process(rng):
    start, stop, path, kfile = rng
    k = np.loadtxt(kfile)[:, 2]
    shifted = np.isfinite(k) & (k != 0)
    t = uproot.open(path)[TREE]
    a = t.arrays([H + x for x in ("crystalId_", "eDep_", "nSiPMs_", "recoDigis_", "clusterIdx_")]
                 + [R + x for x in ("SiPMID_", "time_", "eDep_")] + [C + "energyDep_", "evtinfo/subrun"],
                 entry_start=start, entry_stop=stop)
    out = {
        "dt_e": np.zeros((len(B_DTS) - 1, len(B_EH) - 1)),
        "asy": np.zeros((NCRY, len(B_ASY) - 1)),
        "dtmean": np.zeros((NCRY, 2)),  # sum dt, n  (2-SiPM hits, 15-40 MeV)
        "ontime": np.zeros((NCRY, 2)),  # counts: even on time, odd on time
        "sub_n": np.zeros((NSUB, 5)),   # events, clusters, hits, single-SiPM hits, 2-SiPM hits
        "sub_e": np.zeros((NSUB, len(B_ESUB) - 1)),
    }
    sub = a["evtinfo/subrun"].to_numpy()
    nh = ak.num(a[H + "eDep_"]).to_numpy()
    ns = a[H + "nSiPMs_"]
    np.add.at(out["sub_n"][:, 0], sub, 1)
    np.add.at(out["sub_n"][:, 1], sub, ak.num(a[C + "energyDep_"]).to_numpy())
    np.add.at(out["sub_n"][:, 2], sub, nh)
    np.add.at(out["sub_n"][:, 3], sub, ak.sum(ns == 1, axis=1).to_numpy())
    np.add.at(out["sub_n"][:, 4], sub, ak.sum(ns == 2, axis=1).to_numpy())

    # ---- 2-SiPM hits
    rd = ak.pad_none(a[H + "recoDigis_"], 2, axis=2)
    r0, r1 = ak.fill_none(rd[:, :, 0], 0), ak.fill_none(rd[:, :, 1], 0)
    sip, rt, re = a[R + "SiPMID_"], a[R + "time_"], a[R + "eDep_"]
    two = ns == 2
    s0, s1 = sip[r0][two], sip[r1][two]
    odd_first = (s0 % 2) == 1
    t_odd = ak.where(odd_first, rt[r0][two], rt[r1][two])
    t_even = ak.where(odd_first, rt[r1][two], rt[r0][two])
    e_odd = ak.where(odd_first, re[r0][two], re[r1][two])
    e_even = ak.where(odd_first, re[r1][two], re[r0][two])
    cry2 = ak.flatten(a[H + "crystalId_"][two]).to_numpy()
    eh2 = ak.flatten(a[H + "eDep_"][two]).to_numpy()
    dts = ak.flatten(t_odd - t_even).to_numpy()
    eo, ee = ak.flatten(e_odd).to_numpy(), ak.flatten(e_even).to_numpy()
    out["dt_e"] = np.histogram2d(dts, eh2, [B_DTS, B_EH])[0]
    mip = (eh2 > 15) & (eh2 < 40)
    asy = (eo - ee) / np.maximum(eo + ee, 1e-6)
    np.add.at(out["asy"], (cry2[mip], np.clip(np.digitize(asy[mip], B_ASY) - 1, 0, len(B_ASY) - 2)), 1)
    np.add.at(out["dtmean"][:, 0], cry2[mip], dts[mip])
    np.add.at(out["dtmean"][:, 1], cry2[mip], 1)
    sub2 = np.repeat(sub, ak.sum(two, axis=1).to_numpy())
    ok = eh2 < B_ESUB[-1]
    np.add.at(out["sub_e"], (sub2[ok], np.digitize(eh2[ok], B_ESUB) - 1), 1)

    # ---- on-time side for shifted crystals
    cry = a[H + "crystalId_"]
    cl = a[H + "clusterIdx_"]
    side = sip[ak.fill_none(rd[:, :, 0], 0)] % 2
    is_sh = ak.unflatten(shifted[ak.flatten(cry).to_numpy()], ak.num(cry))
    good_anchor = (ns == 2) & ~is_sh & (cl >= 0)
    single_sh = (ns == 1) & is_sh & (cl >= 0)
    ij = ak.argcartesian([cl, cl])
    p, q = ij["0"], ij["1"]
    m = single_sh[p] & good_anchor[q] & (cl[p] == cl[q])
    # one entry per single hit: has any anchor in its cluster
    evt = ak.broadcast_arrays(ak.local_index(cl, axis=0), p)[0]
    key = ak.flatten(evt[m]).to_numpy().astype(np.int64) * 100000 + ak.flatten(p[m]).to_numpy()
    _, first = np.unique(key, return_index=True)
    c_on = ak.flatten(cry[p][m]).to_numpy()[first]
    s_on = ak.flatten(side[p][m]).to_numpy()[first]
    np.add.at(out["ontime"], (c_on, s_on), 1)
    return out


if __name__ == "__main__":
    path, kfile, outp = sys.argv[1:4]
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    n = uproot.open(path)[TREE].num_entries
    if len(sys.argv) > 5:
        n = min(n, int(sys.argv[5]))
    step = 200_000
    ranges = [(s, min(s + step, n), path, kfile) for s in range(0, n, step)]
    tot = None
    with Pool(nw) as pool:
        for i, h in enumerate(pool.imap_unordered(process, ranges)):
            tot = h if tot is None else {kk: tot[kk] + v for kk, v in h.items()}
            print(f"{i + 1}/{len(ranges)} chunks", flush=True)
    np.savez_compressed(outp, **tot, B_DTS=B_DTS, B_EH=B_EH, B_ASY=B_ASY, B_ESUB=B_ESUB)
    print("events", int(tot["sub_n"][:, 0].sum()), "2-SiPM hits", int(tot["sub_n"][:, 4].sum()))
