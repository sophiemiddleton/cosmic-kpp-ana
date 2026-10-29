"""Absolute per-SiPM time offsets relative to well-behaved reference crystals, propagated iteratively.

Pass 1 references ("anchors"): 2-SiPM hits with E > 5 MeV in crystals whose SiPMs merge fine (k not
measured in sipm_offsets_k.txt). For every other hit within 120 mm on the same disk, histogram
t_hit - t_anchor per (crystal, side), side 0/1 = single even/odd SiPM, 2 = merged hit; the mode is the
offset. Later passes also use every hit whose (crystal, side) offset is already known as an anchor, with
that offset removed from the anchor time, so the calibration spreads into regions without good crystals.

Usage: python sipm_abs_offsets.py <ntuple.root> <sipm_offsets_k.txt> <out.npz> [max_entries] [passes]
"""
import sys

import awkward as ak
import numpy as np
import uproot

from fill_calo import TREE

H, R = "calohits/calohits.", "calorecodigis/calorecodigis."
NCRY = 1348
B = np.arange(-1500, 1501, 5.0)
C = 0.5 * (B[1:] + B[:-1])
NMIN = 30


def one_pass(t, N, good, corr):
    """corr: (NCRY, 3) known offsets (nan = unknown). Returns histogram of t_hit(raw) - t_anchor(corrected)."""
    hist = np.zeros((NCRY, 3, len(B) - 1), np.int64)
    known = np.isfinite(corr)
    c0 = np.nan_to_num(corr)
    for a in t.iterate([H + v for v in ("crystalId_", "eDep_", "time_", "nSiPMs_", "recoDigis_",
                                        "crystalPos_.fCoordinates.fX", "crystalPos_.fCoordinates.fY")]
                       + [R + "SiPMID_"], entry_stop=N, step_size=100_000):
        first = ak.fill_none(ak.firsts(a[H + "recoDigis_"], axis=2), 0)
        cry = a[H + "crystalId_"]
        side = ak.where(a[H + "nSiPMs_"] >= 2, 2, a[R + "SiPMID_"][first] % 2)
        cf, sf = ak.flatten(cry).to_numpy(), ak.flatten(side).to_numpy()
        cnt = ak.num(cry)
        isgood = ak.unflatten(good[cf], cnt)
        isknown = ak.unflatten(known[cf, sf], cnt)
        off = ak.unflatten(c0[cf, sf], cnt)
        anchor = (a[H + "eDep_"] > 5) & (((a[H + "nSiPMs_"] >= 2) & isgood) | isknown)
        h = ak.zip({"cry": cry, "t": a[H + "time_"], "side": side, "x": a[H + "crystalPos_.fCoordinates.fX"],
                    "y": a[H + "crystalPos_.fCoordinates.fY"], "anchor": anchor,
                    "ta": a[H + "time_"] - ak.where(isknown, off, 0.0)})
        ij = ak.argcartesian([h.t, h.t])
        p, q = h[ij["0"]], h[ij["1"]]
        dt = q.t - p.ta
        m = (p.anchor & (p.cry != q.cry) & ((p.cry >= NCRY // 2) == (q.cry >= NCRY // 2))
             & (np.hypot(q.x - p.x, q.y - p.y) < 120) & (abs(dt) < B[-1]))
        c_, s_, d_ = (ak.flatten(v[m]).to_numpy() for v in (q.cry, q.side, dt))
        np.add.at(hist, (c_, s_, np.digitize(d_, B) - 1), 1)
    return hist


if __name__ == "__main__":
    path, kfile, outp = sys.argv[1:4]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 2_000_000
    passes = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    good = ~np.isfinite(np.loadtxt(kfile)[:, 2])
    t = uproot.open(path)[TREE]
    corr = np.full((NCRY, 3), np.nan)
    corr[good, 2] = 0.0  # reference crystals define t = 0
    for it in range(passes):
        hist = one_pass(t, N, good, corr)
        n = hist.sum(2)
        mode = np.where(n > NMIN, C[np.argmax(hist, 2)], np.nan)
        mode[good, 2] = 0.0
        print(f"pass {it + 1}: measured (even, odd, merged) = {np.isfinite(mode).sum(0)}", flush=True)
        corr = mode
    np.savez_compressed(outp, hist=hist, B=B, mode=mode, n=n)
    for s, lab in enumerate(["single even", "single odd", "merged"]):
        v = mode[:, s][np.isfinite(mode[:, s])]
        big = np.abs(v) > 84
        print(f"{lab:12s}: {len(v)} crystals measured, |offset| > 84 ns: {big.sum()},"
              f"  steps present: {sorted(set(np.round(v[big] / 169).astype(int)))}")
    with open(outp.replace(".npz", "_shifted.txt"), "w") as f:
        f.write("# SiPMs (or whole crystals, side 'merged') whose time is off by > 84 ns w.r.t. reference crystals\n")
        f.write("# crystalId  SiPMID(-1 = merged hit, both SiPMs)  side  offset_ns  k=offset/169  n_entries\n")
        for cid in range(NCRY):
            for s, lab in enumerate(["even", "odd", "merged"]):
                if np.isfinite(mode[cid, s]) and abs(mode[cid, s]) > 84:
                    sip = 2 * cid + s if s < 2 else -1
                    f.write(f"{cid:5d} {sip:6d} {lab:7s} {mode[cid, s]:8.1f} {int(np.round(mode[cid, s] / 169)):+3d}"
                            f" {int(n[cid, s]):7d}\n")
