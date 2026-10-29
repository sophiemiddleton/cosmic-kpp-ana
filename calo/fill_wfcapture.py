"""Second pulse in the stop crystal's raw waveforms (mu- capture search), with an embedding efficiency.

For each selected stop candidate (fill_stops.py) both SiPMs of the stop crystal are taken at the raw digitizer
time of the stop hit, so the reco-level SiPM time offsets (echo_study.py) don't enter. The digis are fitted
jointly (shared times, free pedestal and amplitudes per SiPM) with
  1 pulse:  ped + A1 T(t - t1) + B T'(t - t1) + C T''(t - t1)
  2 pulses: the same + A2 T(t - t1 - delta),  delta from DELTA_MIN to the end of the digi
T is the data template (wf_template.py). Energies use each SiPM's gain A1 / E_recodigi.
Later digis on the same two SiPMs (up to LATE_MAX after the stop) are listed separately.

Embedding: on a random EMBED_FRAC of the candidates the fit is repeated after adding a template pulse of known
delta and energy to the waveforms; the recovered pulse gives the efficiency vs (delta, E).

Usage: python fill_wfcapture.py <ntuple.root> <stops.npz> <template.npz> <out.npz> [nworkers] [max_entries]
"""
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np
import uproot

from fill_calo import TREE

H, R, D = "calohits/calohits.", "calorecodigis/calorecodigis.", "calodigis/calodigis."
TICK, PEAK = 5.0, 17
ADC_SAT = 4090
# candidate selection (same as plot_stops.py)
ELAST = (25.0, 70.0)
AHEAD_MIN = 40.0
T_STOP = (2000.0, 65000.0)
# fit
T1_GRID = np.arange(-8.0, 8.01, 0.25)
DELTA_MIN, DELTA_STEP = 20.0, 2.5
PAIR_NS = 50.0          # other-side digi within this raw time of the stop hit's digi
LATE_MAX = 2000.0       # ns
EMBED_FRAC = 0.25
EMBED_DELTA = (10.0, 300.0)
EMBED_E = (0.3, 10.0)

TMPL = None


def tmpl(x, k=1):
    """Template (k=1) and its first/second time derivatives (k=2, 3), scaled by 5 ns and 25 ns^2."""
    return np.interp(x, TMPL[0], TMPL[k], left=0.0, right=0.0)


def basis(tau, t1):
    """Main-pulse columns [1, T, T', T''] for every t1 (t1: (k,)) -> (k, 4, n)."""
    x = tau[None, :] - t1[:, None]
    return np.stack([np.ones_like(x), tmpl(x), tmpl(x, 2), tmpl(x, 3)], 1)


def lsq(w, X):
    """Batched linear least squares, X: (..., ncol, n). Returns chi2 (...) and coefficients (..., ncol)."""
    G = X @ np.swapaxes(X, -1, -2)
    b = X @ w
    G = G + 1e-9 * np.eye(G.shape[-1])
    c = np.linalg.solve(G, b[..., None])[..., 0]
    return (w * w).sum() - (b * c).sum(-1), c


def fit(ws, taus):
    """Joint 1- and 2-pulse fits to the waveforms of one crystal. taus: sample times relative to the reference.
    The main pulse has a free amplitude, T' and T'' term per SiPM (absorbing small time and width differences);
    the times t1 and t2 are shared by the SiPMs."""
    chi1 = 0.0
    c1 = []
    for w, tau in zip(ws, taus):
        c, cc = lsq(w, basis(tau, T1_GRID))
        chi1 = chi1 + c
        c1.append(cc)
    i = int(np.argmin(chi1))
    t1 = T1_GRID[i]
    ndf1 = sum(len(w) for w in ws) - 4 * len(ws) - 1
    out = {"t1": t1, "chi1": chi1[i], "ndf1": ndf1, "A1": np.array([c[i, 1] for c in c1])}
    tend = min(tau[-1] for tau in taus) - 10.0
    deltas = np.arange(DELTA_MIN, tend - t1 + 1e-9, DELTA_STEP)
    if len(deltas) == 0:
        out.update(chi2=out["chi1"], delta=np.nan, A2=np.full(len(ws), np.nan))
        return out
    chi2 = 0.0
    a2 = []
    for w, tau in zip(ws, taus):
        B = basis(tau, np.array([t1]))[0]                                     # (4, n)
        T2 = tmpl(tau[None, :] - t1 - deltas[:, None])                        # (nd, n)
        X = np.concatenate([np.broadcast_to(B, (len(deltas),) + B.shape), T2[:, None, :]], 1)
        c, cc = lsq(w, X)
        chi2 = chi2 + c
        a2.append(cc[:, 4])
    k = int(np.argmin(chi2))
    out.update(chi2=chi2[k], delta=deltas[k], A2=np.array([a[k] for a in a2]))
    return out


def process(args):
    start, stop, path, cand, seed = args
    rng = np.random.default_rng(seed)
    t = uproot.open(path)[TREE]
    a = t.arrays([H + "recoDigis_", R + "caloDigiIdx_", R + "eDep_", D + "SiPMID_", D + "t0_", D + "waveform_",
                  D + "peakval_", D + "caloRecoDigiIdx_"], entry_start=start, entry_stop=stop)
    rows, late, emb = [], [], []
    for ci, (ev, hit, cry) in enumerate(zip(cand["evt"], cand["hit"], cand["cry"])):
        e = a[int(ev) - start]
        rds = np.asarray(e[H + "recoDigis_"][int(hit)])
        dsipm = e[D + "SiPMID_"].to_numpy()
        dt0 = e[D + "t0_"].to_numpy()
        rd_d = e[R + "caloDigiIdx_"].to_numpy()
        if len(rds) == 0:
            continue
        tref = np.mean([TICK * (dt0[rd_d[r]] + PEAK) for r in rds])
        ws, taus, gains, sides, sat = [], [], [], [], False
        for side in (0, 1):
            s = 2 * int(cry) + side
            cand_d = np.flatnonzero(dsipm == s)
            if len(cand_d) == 0:
                continue
            traw = TICK * (dt0[cand_d] + PEAK)
            k = np.argmin(np.abs(traw - tref))
            if abs(traw[k] - tref) > PAIR_NS:
                continue
            d = cand_d[k]
            w = e[D + "waveform_"][d].to_numpy().astype(float)
            sat |= e[D + "peakval_"][d] >= ADC_SAT
            ws.append(w)
            taus.append(TICK * (dt0[d] + np.arange(len(w))) - tref)
            r = e[D + "caloRecoDigiIdx_"][d]
            gains.append(e[R + "eDep_"][r] if r >= 0 else np.nan)
            sides.append(side)
            # later digis on this SiPM
            for dl in cand_d[(traw > traw[k] + TICK * (len(w) - PEAK)) & (traw < traw[k] + LATE_MAX)]:
                wl = e[D + "waveform_"][dl].to_numpy().astype(float)
                late.append((ci, side, TICK * (dt0[dl] + PEAK) - tref, wl.max() - np.median(wl[:4])))
        if not ws or sat:
            continue
        f = fit(ws, taus)
        g = f["A1"] / np.array(gains)                                        # ADC per MeV of the main pulse
        g = np.where(np.isfinite(g) & (g > 5) & (g < 60), g, np.nan)
        e2 = np.nanmean(f["A2"] / g) if np.any(np.isfinite(g)) else np.nan
        e1 = np.nanmean(f["A1"] / g) if np.any(np.isfinite(g)) else np.nan
        e2s = f["A2"] / g
        rows.append((ci, len(ws), f["t1"], f["chi1"], f["chi2"], f["delta"], e1, e2,
                     e2s[0], e2s[-1] if len(ws) > 1 else np.nan, max(len(w) for w in ws), f["ndf1"]))
        if rng.random() < EMBED_FRAC and np.all(np.isfinite(g)):
            dl = rng.uniform(*EMBED_DELTA)
            ei = rng.uniform(*EMBED_E)
            wi = [w + ei * gg * tmpl(tau - f["t1"] - dl) for w, tau, gg in zip(ws, taus, g)]
            fi = fit(wi, taus)
            emb.append((ci, dl, ei, fi["chi1"], fi["chi2"], fi["delta"], np.nanmean(fi["A2"] / g),
                        max(len(w) for w in ws), fi["ndf1"]))
    return np.array(rows, float).reshape(-1, 12), np.array(late, float).reshape(-1, 4), \
        np.array(emb, float).reshape(-1, 9), cand["idx"]


def select(S):
    return np.flatnonzero((S["s_elast"] > ELAST[0]) & (S["s_elast"] < ELAST[1]) & (S["s_ahead"] >= AHEAD_MIN)
                          & (S["s_t"] > T_STOP[0]) & (S["s_t"] < T_STOP[1]))


def init(tm):
    global TMPL
    TMPL = tm


if __name__ == "__main__":
    path, stops, tpath, outp = sys.argv[1:5]
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    nmax = int(sys.argv[6]) if len(sys.argv) > 6 else None
    S = np.load(stops)
    S = {k: S[k] for k in ("s_evt", "s_hit", "s_cry", "s_elast", "s_ahead", "s_t")}
    sel = select(S)
    if nmax:
        sel = sel[S["s_evt"][sel] < nmax]
    T = np.load(tpath)
    dT = np.gradient(T["tmpl"], T["grid"])
    tm = (T["grid"], T["tmpl"], 5.0 * dT, 25.0 * np.gradient(dT, T["grid"]))
    ev = S["s_evt"][sel]
    step = 50_000
    lo, hi = int(ev.min()), int(ev.max()) + 1
    jobs = []
    for k, s in enumerate(range(lo, hi, step)):
        m = (ev >= s) & (ev < s + step)
        if m.any():
            jobs.append((s, min(s + step, hi), path,
                         {"evt": ev[m], "hit": S["s_hit"][sel][m], "cry": S["s_cry"][sel][m], "idx": sel[m]}, k))
    rows, late, emb = [], [], []
    with Pool(nw, initializer=init, initargs=(tm,)) as pool:
        for k, (r, l, em, idx) in enumerate(pool.imap_unordered(process, jobs)):
            # local candidate index -> global stop index
            r[:, 0] = idx[r[:, 0].astype(int)]
            l[:, 0] = idx[l[:, 0].astype(int)] if len(l) else l[:, 0]
            em[:, 0] = idx[em[:, 0].astype(int)] if len(em) else em[:, 0]
            rows.append(r)
            late.append(l)
            emb.append(em)
            print(f"{k + 1}/{len(jobs)} chunks", flush=True)
    r, l, em = np.concatenate(rows), np.concatenate(late), np.concatenate(emb)
    cols = ["stop", "nsipm", "t1", "chi1", "chi2", "delta", "e1", "e2", "e2a", "e2b", "nsamp", "ndf1"]
    np.savez_compressed(outp, **{"f_" + c: r[:, i] for i, c in enumerate(cols)},
                        **{"l_" + c: l[:, i] for i, c in enumerate(["stop", "side", "dt", "adc"])},
                        **{"m_" + c: em[:, i] for i, c in enumerate(["stop", "delta_in", "e_in", "chi1", "chi2",
                                                                     "delta", "e2", "nsamp", "ndf1"])},
                        ncand=len(sel))
    print("candidates", len(sel), "fitted", len(r), "late digis", len(l), "embedded", len(em))
