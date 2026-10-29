"""Stopped-muon candidates and the hit activity around them (input to plot_stops.py and fill_wfcapture.py).

Hit times are corrected with the per-SiPM offsets of sipm_abs_offsets.py. Track = hits on the same disk chained in
time (gaps < TRK_GAP); reco clusters are not used, since the clustering splits one muon into several same-time
pieces. Muon = track above PROMPT_EMIN with no earlier muon track on the same disk within 150 mm and
25 us (this drops Michel and echo clusters as prompts). Its hits are fitted with a line (energy-weighted
principal axis), oriented downwards; the stop crystal is the hit furthest along it. Cosmics travel downwards, so
the stop is the lowest end of the track.

Delayed (and early) activity: hits on the same disk within GRP_R of the stop crystal, -20 to +60 us around the
muon, not in the muon track, grouped in time (gaps > GRP_GAP start a new group). For |dt| < 1.3 us the
muon's own crystals are dropped: their echoes (SiPM time offsets, see echo_study.py) sit there.

Usage: python fill_stops.py <ntuple.root> <out.npz> [nworkers] [max_entries] [calo.npz] [sipm_abs_offsets.npz]
"""
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np
import uproot
from scipy.spatial import cKDTree

from fill_calo import HT, TREE
from fill_capture2 import tables

R = "calorecodigis/calorecodigis."

NCRY = 1348
PITCH = 34.3                    # mm
PROMPT_EMIN = 15.0              # MeV
TRK_GAP = 30.0                  # ns, same-disk hits closer than this in (corrected) time form one track
HIT_EMIN = 1.0                  # MeV, hits used for the track line
PRE_DT, PRE_DR = 25000.0, 150.0  # earlier muon within this -> not a prompt
GRP_R = 150.0                   # mm
GRP_T = (-20000.0, 60000.0)     # ns
GRP_GAP = 30.0                  # ns
ECHO_T = 1300.0                 # ns
OUTSIDE = 26.0                  # mm, a point further than this from every crystal centre is outside the disk


def geometry(calo_npz):
    d = np.load(calo_npz)
    xy = np.column_stack([d["cry_x"][:NCRY // 2], d["cry_y"][:NCRY // 2]])  # both disks share the layout
    tree = cKDTree(xy)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rad = np.arange(5.0, 300.0, 5.0)
    depth = np.full(len(xy), np.nan)
    for i, p in enumerate(xy):
        for r in rad:
            q = p + r * np.column_stack([np.cos(ang), np.sin(ang)])
            if tree.query(q)[0].max() > OUTSIDE:
                depth[i] = r
                break
    return xy, tree, np.r_[depth, depth]


def ahead(tree, x, y, ux, uy, step=5.0, rmax=200.0):
    """Distance from (x, y) along u to the disk edge (rmax if further)."""
    out = np.full(len(x), rmax)
    for r in np.arange(step, rmax, step)[::-1]:
        d, _ = tree.query(np.column_stack([x + r * ux, y + r * uy]))
        out[d > OUTSIDE] = r
    return out


def process(args):
    start, stop, path, tree, depth, corr = args
    t = uproot.open(path)[TREE]
    a = t.arrays([HT + v for v in ("crystalId_", "eDep_", "time_", "nSiPMs_", "recoDigis_",
                                   "crystalPos_.fCoordinates.fX", "crystalPos_.fCoordinates.fY")]
                 + [R + "SiPMID_"], entry_start=start, entry_stop=stop)
    nh = ak.num(a[HT + "eDep_"]).to_numpy()
    hoff = np.r_[0, np.cumsum(nh)]
    H = {k: ak.flatten(a[HT + v]).to_numpy() for k, v in
         (("cry", "crystalId_"), ("e", "eDep_"), ("t", "time_"), ("ns", "nSiPMs_"),
          ("x", "crystalPos_.fCoordinates.fX"), ("y", "crystalPos_.fCoordinates.fY"))}
    hev = np.repeat(np.arange(len(nh)), nh)
    H["disk"] = (H["cry"] >= NCRY // 2).astype(np.int32)
    # SiPM time offsets (sipm_abs_offsets.py): per (crystal, side), side 0/1 = single SiPM, 2 = merged hit
    first = ak.fill_none(ak.firsts(a[HT + "recoDigis_"], axis=2), 0)
    side = ak.flatten(a[R + "SiPMID_"][first] % 2).to_numpy()
    H["t"] = H["t"].astype(np.float64) + corr[H["cry"], np.where(H["ns"] == 1, side, 2)]

    # ---- tracks: hits on the same disk chained in (corrected) time. Reco clusters are not used: the clustering
    # splits one muon into several same-time pieces, and every upper piece then looks like a stop.
    o = np.lexsort((H["t"], H["disk"], hev))
    brk = np.r_[True, (hev[o][1:] != hev[o][:-1]) | (H["disk"][o][1:] != H["disk"][o][:-1])
                | (np.diff(H["t"][o]) > TRK_GAP)] if len(o) else np.zeros(0, bool)
    htr = np.empty(len(o), np.int64)
    htr[o] = np.cumsum(brk) - 1
    n = int(brk.sum())
    tev = hev[o][brk]
    tdisk = H["disk"][o][brk]
    # energy per crystal = mean of its hits (the two single-SiPM hits of a split crystal each estimate it)
    key = htr * 2000 + H["cry"]
    uk, inv, kc = np.unique(key, return_inverse=True, return_counts=True)
    ecry = np.bincount(inv, H["e"]) / kc
    te = np.bincount(uk // 2000, ecry, n)
    tnc = np.bincount(uk // 2000, None, n).astype(np.int32)
    oo = np.lexsort((-H["e"], htr))
    f0 = np.r_[True, htr[oo][1:] != htr[oo][:-1]]
    tt = np.empty(n)
    tt[htr[oo][f0]] = H["t"][oo][f0]                                  # time of the most energetic hit

    # ---- track line (energy-weighted principal axis of the hits above HIT_EMIN)
    hi = np.flatnonzero(H["e"] > HIT_EMIN)
    g = htr[hi]
    w = H["e"][hi]
    sw = np.bincount(g, w, n)
    nth = np.bincount(g, None, n)
    mx = np.bincount(g, w * H["x"][hi], n) / np.maximum(sw, 1e-9)
    my = np.bincount(g, w * H["y"][hi], n) / np.maximum(sw, 1e-9)
    dx, dy = H["x"][hi] - mx[g], H["y"][hi] - my[g]
    sxx = np.bincount(g, w * dx * dx, n)
    syy = np.bincount(g, w * dy * dy, n)
    sxy = np.bincount(g, w * dx * dy, n)
    phi = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    ux, uy = np.cos(phi), np.sin(phi)
    flip = uy > 0
    ux[flip], uy[flip] = -ux[flip], -uy[flip]
    single = (sxx + syy) < 1.0                                         # one crystal: direction unknown
    ux[single], uy[single] = 0.0, -1.0
    tr = sxx + syy
    det = sxx * syy - sxy ** 2
    lam1 = 0.5 * (tr + np.sqrt(np.maximum(tr ** 2 - 4 * det, 0)))
    elong = np.where(sw > 0, np.sqrt(np.maximum(lam1, 0) / np.maximum(sw, 1e-9)), 0)
    s = dx * ux[g] + dy * uy[g]
    # endpoint = max projection per track; also the track extent
    o = np.lexsort((-s, g))
    first = np.r_[True, g[o][1:] != g[o][:-1]]
    endh = np.full(n, -1)
    endh[g[o][first]] = hi[o][first]
    o2 = np.lexsort((s, g))
    first2 = np.r_[True, g[o2][1:] != g[o2][:-1]]
    smin = np.zeros(n)
    smin[g[o2][first2]] = s[o2][first2]
    smax = np.zeros(n)
    smax[g[o][first]] = s[o][first]

    # ---- muon tracks with no earlier muon on the same disk nearby (drops Michels and echoes as prompts)
    mu = np.flatnonzero((te > PROMPT_EMIN) & (endh >= 0))
    keep = np.ones(len(mu), bool)
    b = np.r_[0, np.flatnonzero(np.diff(tev[mu])) + 1, len(mu)]
    for s0, e0 in zip(b[:-1], b[1:]):
        if e0 - s0 < 2:
            continue
        idx = mu[s0:e0]
        dt = tt[idx][:, None] - tt[idx][None, :]                     # [i, j] = t_i - t_j
        dr = np.hypot(mx[idx][:, None] - mx[idx][None, :], my[idx][:, None] - my[idx][None, :])
        same = tdisk[idx][:, None] == tdisk[idx][None, :]
        keep[s0:e0] = ~np.any(same & (dt > TRK_GAP) & (dt < PRE_DT) & (dr < PRE_DR), axis=1)
    mu = mu[keep]
    eh = endh[mu]
    elast = ecry[inv[eh]]
    S = {
        "evt": tev[mu] + start, "hit": eh - hoff[hev[eh]],
        "disk": tdisk[mu], "t": tt[mu], "ecl": te[mu], "size": tnc[mu],
        "ntrk": nth[mu].astype(np.int32), "len": (smax - smin)[mu], "elong": elong[mu],
        "ux": ux[mu], "uy": uy[mu],
        "cry": H["cry"][eh], "x": H["x"][eh], "y": H["y"][eh], "elast": elast, "tlast": H["t"][eh],
        "nslast": H["ns"][eh], "depth": depth[H["cry"][eh]],
    }
    S["ahead"] = ahead(tree, S["x"], S["y"], S["ux"], S["uy"])

    # ---- activity around each stop: (stop, hit) pairs in the same event
    sidx = np.arange(len(mu))
    sev = tev[mu]
    cnt = nh[sev]
    ps = np.repeat(sidx, cnt)
    ph = np.repeat(hoff[sev], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    dt = H["t"][ph] - S["t"][ps]
    dr = np.hypot(H["x"][ph] - S["x"][ps], H["y"][ph] - S["y"][ps])
    m = (H["disk"][ph] == S["disk"][ps]) & (dt > GRP_T[0]) & (dt < GRP_T[1]) & (dr < GRP_R)
    m &= htr[ph] != mu[ps]
    # echo veto for SiPMs without an offset measurement: muon's own crystals near the muon time
    mucry = np.unique(htr.astype(np.int64) * 2000 + H["cry"])
    own = np.isin(mu[ps].astype(np.int64) * 2000 + H["cry"][ph], mucry)
    m &= ~(own & (np.abs(dt) < ECHO_T))
    ps, ph, dt, dr = ps[m], ph[m], dt[m], dr[m]
    o = np.lexsort((dt, ps))
    ps, ph, dt, dr = ps[o], ph[o], dt[o], dr[o]
    newg = np.r_[True, (ps[1:] != ps[:-1]) | (np.diff(dt) > GRP_GAP)] if len(ps) else np.zeros(0, bool)
    gb = np.flatnonzero(newg)
    e = H["e"][ph]
    G = {"stop": ps[gb] if len(gb) else np.zeros(0, np.int64)}
    if len(gb):
        gid = np.cumsum(newg) - 1
        # a crystal with several hits in the group (the two single-SiPM hits of a split crystal) counts once
        _, inv, kc = np.unique(gid * 2000 + H["cry"][ph], return_inverse=True, return_counts=True)
        e = e / kc[inv]
        G["e"] = np.add.reduceat(e, gb)
        G["nhit"] = np.diff(np.r_[gb, len(e)])
        G["x"] = np.add.reduceat(e * H["x"][ph], gb) / G["e"] - S["x"][G["stop"]]
        G["y"] = np.add.reduceat(e * H["y"][ph], gb) / G["e"] - S["y"][G["stop"]]
        G["dmin"] = np.minimum.reduceat(dr, gb)
        G["emax"] = np.maximum.reduceat(e, gb)
        # time of the most energetic hit of the group
        oo = np.lexsort((-e, gid))
        ff = np.r_[True, gid[oo][1:] != gid[oo][:-1]]
        G["dt"] = dt[oo][ff]
        G["instop"] = np.add.reduceat((H["cry"][ph] == S["cry"][ps]).astype(np.int32), gb) > 0
    else:
        for k in ("e", "nhit", "x", "y", "dmin", "emax", "dt", "instop"):
            G[k] = np.zeros(0)
    return S, G, len(nh)


def merge(res):
    S = {k: np.concatenate([r[0][k] for r in res]) for k in res[0][0]}
    off = np.cumsum([0] + [len(r[0]["evt"]) for r in res[:-1]])
    G = {k: np.concatenate([r[1][k] for r in res]) for k in res[0][1] if k != "stop"}
    G["stop"] = np.concatenate([r[1]["stop"] + o for r, o in zip(res, off)])
    return S, G, sum(r[2] for r in res)


if __name__ == "__main__":
    path, outp = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    calo = sys.argv[5] if len(sys.argv) > 5 else "calo_run125329.npz"
    corr = tables(sys.argv[6] if len(sys.argv) > 6 else "sipm_abs_offsets_run125329.npz")
    n = uproot.open(path)[TREE].num_entries
    if len(sys.argv) > 4 and int(sys.argv[4]) > 0:
        n = min(n, int(sys.argv[4]))
    xy, tree, depth = geometry(calo)
    step = 100_000
    ranges = [(s, min(s + step, n), path, tree, depth, corr) for s in range(0, n, step)]
    res = []
    with Pool(nw) as pool:
        for k, r in enumerate(pool.imap(process, ranges)):
            res.append(r)
            print(f"{k + 1}/{len(ranges)} chunks", flush=True)
    S, G, nev = merge(res)
    np.savez_compressed(outp, nev=nev, cry_depth=depth, cry_x=np.r_[xy[:, 0], xy[:, 0]],
                        cry_y=np.r_[xy[:, 1], xy[:, 1]],
                        **{"s_" + k: v for k, v in S.items()}, **{"g_" + k: v for k, v in G.items()})
    print("events", nev, "stops", len(S["evt"]), "groups", len(G["e"]))
