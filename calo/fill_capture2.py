"""Hit-level mu- capture search with the SiPM time offsets corrected (see echo_study.py).

Per hit: the absolute offset of its SiPM (single-SiPM hit) or of the crystal (merged hit), measured against
reference crystals by sipm_abs_offsets.py, is removed. Opposite-side single-SiPM hits of one crystal within MERGE_NS are then merged. Clusters with more than half
their energy in shifted-SiPM hits are echo clusters and are not used as prompts; prompt times are the
corrected time of the cluster's most energetic hit. Histogram axes as in fill_capture.py.

Usage: python fill_capture2.py <ntuple.root> <sipm_abs_offsets.npz> <out.npz> [nworkers] [max]
"""
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np
import uproot

from fill_calo import CL, HT, TREE
from fill_capture import BINS, B_D, B_DT, B_E, B_IN, B_NS, NCRY, PROMPT_EMIN

R = "calorecodigis/calorecodigis."
MERGE_NS = 25.0


def tables(absfile):
    """Time correction per (crystal, side), side 0/1 = single even/odd SiPM, 2 = merged hit (sipm_abs_offsets.py)."""
    mode = np.load(absfile)["mode"]
    return -np.nan_to_num(mode)


def process(rng):
    start, stop, path, corr = rng
    t = uproot.open(path)[TREE]
    a = t.arrays([CL + "energyDep_"]
                 + [HT + v for v in ("crystalId_", "eDep_", "time_", "clusterIdx_", "nSiPMs_", "recoDigis_",
                                     "crystalPos_.fCoordinates.fX", "crystalPos_.fCoordinates.fY")]
                 + [R + "SiPMID_"], entry_start=start, entry_stop=stop)
    nev = len(a)
    nh = ak.num(a[HT + "eDep_"]).to_numpy()
    ncl = ak.num(a[CL + "energyDep_"]).to_numpy()
    evt = np.repeat(np.arange(nev), nh)
    f = lambda b: ak.flatten(a[HT + b]).to_numpy()
    cry, e, tt, cl, ns = f("crystalId_"), f("eDep_"), f("time_").astype(np.float64), f("clusterIdx_"), f("nSiPMs_")
    x, y = f("crystalPos_.fCoordinates.fX"), f("crystalPos_.fCoordinates.fY")
    first = ak.fill_none(ak.firsts(a[HT + "recoDigis_"], axis=2), 0)
    side = ak.flatten(a[R + "SiPMID_"][first] % 2).to_numpy()

    # ---- time correction
    single = ns == 1
    sidx = np.where(single, side, 2)
    tc = tt + corr[cry, sidx]
    shifted_hit = np.abs(corr[cry, sidx]) > 84

    # ---- echo clusters (by energy share of shifted-SiPM hits, before merging)
    cl_off = np.r_[0, np.cumsum(ncl)[:-1]]
    g = np.where(cl >= 0, cl_off[evt] + cl, -1)
    ing = g >= 0
    ncl_tot = ncl.sum()
    esum = np.bincount(g[ing], weights=e[ing], minlength=ncl_tot)
    esh = np.bincount(g[ing], weights=(e * shifted_hit)[ing], minlength=ncl_tot)
    cl_e = ak.flatten(a[CL + "energyDep_"]).to_numpy()
    prompt_cl = (cl_e > PROMPT_EMIN) & (esh <= 0.5 * np.maximum(esum, 1e-9))
    # corrected cluster time = corrected time of its most energetic hit
    o = np.lexsort((-e[ing], g[ing]))
    gi, ti = g[ing][o], tc[ing][o]
    firstg = np.r_[True, gi[1:] != gi[:-1]]
    cl_t = np.full(ncl_tot, np.nan)
    cl_t[gi[firstg]] = ti[firstg]

    # ---- merge opposite-side single hits of one crystal
    o = np.lexsort((tc, cry, evt))
    ev_s, cry_s, tc_s, ns_s, sd_s = evt[o], cry[o], tc[o], ns[o], side[o]
    dup = np.r_[False, (ev_s[1:] == ev_s[:-1]) & (cry_s[1:] == cry_s[:-1]) & (ns_s[1:] == 1) & (ns_s[:-1] == 1)
                & (sd_s[1:] != sd_s[:-1]) & (tc_s[1:] - tc_s[:-1] < MERGE_NS)]
    dup &= ~np.r_[False, dup[:-1]]  # pairs only, no chains
    lead = np.r_[dup[1:], False]
    t_m, e_m, g_m, ns_m = tc[o].copy(), e[o].copy(), g[o].copy(), ns[o].copy()
    i = np.flatnonzero(lead)
    t_m[i] = 0.5 * (t_m[i] + t_m[i + 1])
    e_m[i] = 0.5 * (e_m[i] + e_m[i + 1])
    g_m[i] = np.where(g_m[i] >= 0, g_m[i], g_m[i + 1])
    ns_m[i] = 2
    k = ~dup
    ev_k, cry_k, t_k, e_k, g_k, ns_k = ev_s[k], cry_s[k], t_m[k], e_m[k], g_m[k], ns_m[k]
    x_k, y_k = x[o][k], y[o][k]
    nmerged = int(dup.sum())

    # ---- back to jagged and pair as in fill_capture.py
    cnt = np.bincount(ev_k, minlength=nev)
    U = lambda v: ak.unflatten(v, cnt)
    h = ak.zip({"g": U(g_k), "x": U(x_k), "y": U(y_k), "t": U(t_k), "e": U(e_k), "ns": U(ns_k),
                "disk": U(cry_k >= NCRY // 2),
                "prompt": U((g_k >= 0) & prompt_cl[np.maximum(g_k, 0)]),
                "tcl": U(np.where(g_k >= 0, cl_t[np.maximum(g_k, 0)], np.nan))})
    ij = ak.argcartesian([h.t, h.t])
    p, q = h[ij["0"]], h[ij["1"]]
    dt = q.t - p.tcl
    m = p.prompt & (q.g != p.g) & (q.disk == p.disk) & (dt > B_DT[0]) & (dt < B_DT[-1])
    d = np.hypot(q.x - p.x, q.y - p.y)
    ev2 = ak.broadcast_arrays(ak.local_index(h.t, axis=0), ij["0"])[0]
    cols = {"evt": ev2[m], "j": ij["1"][m], "d": d[m], "dt": dt[m], "e": q.e[m], "ns": q.ns[m],
            "inc": ak.values_astype(q.g[m] >= 0, np.int8)}
    cols = {kk: ak.flatten(v).to_numpy() for kk, v in cols.items()}
    oo = np.lexsort((cols["d"], cols["j"], cols["evt"]))
    cols = {kk: v[oo] for kk, v in cols.items()}
    key = cols["evt"].astype(np.int64) * 100000 + cols["j"]
    fst = np.r_[True, key[1:] != key[:-1]]
    xx = np.column_stack([cols["dt"][fst], cols["d"][fst], cols["e"][fst], np.minimum(cols["ns"][fst], 2),
                          cols["inc"][fst]])
    hist, _ = np.histogramdd(xx, BINS)
    return hist.astype(np.int64), int(prompt_cl.sum()), int((cl_e > PROMPT_EMIN).sum()), nmerged, len(o)


if __name__ == "__main__":
    path, absfile, outp = sys.argv[1:4]
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    n = uproot.open(path)[TREE].num_entries
    if len(sys.argv) > 5:
        n = min(n, int(sys.argv[5]))
    corr = tables(absfile)
    step = 100_000
    ranges = [(s, min(s + step, n), path, corr) for s in range(0, n, step)]
    tot, npr, nall, nmer, nhit = 0, 0, 0, 0, 0
    with Pool(nw) as pool:
        for i, (hh, a1, a2, a3, a4) in enumerate(pool.imap_unordered(process, ranges)):
            tot = tot + hh
            npr, nall, nmer, nhit = npr + a1, nall + a2, nmer + a3, nhit + a4
            print(f"{i + 1}/{len(ranges)} chunks", flush=True)
    np.savez_compressed(outp, hist=tot, nprompt=npr, nev=n, B_DT=B_DT, B_D=B_D, B_E=B_E, B_NS=B_NS, B_IN=B_IN)
    print(f"events {n}  clusters >{PROMPT_EMIN} MeV {nall}, used as prompt (not echo) {npr}  "
          f"hits {nhit}, merged pairs {nmer}")
