"""Average SiPM pulse shape from data, on a 0.5 ns grid (input to fill_wfcapture.py).

Isolated, unsaturated single-pulse digis (one recodigi, not pile-up, E_SEL MeV) are fitted with the current
template (time, amplitude, pedestal); their samples are then put on a common time axis and averaged. Three
iterations, starting from the peak-aligned mean. Templates are also made in amplitude bins to check whether the
shape depends on the pulse height.

Usage: python wf_template.py <ntuple.root> <out.npz> [max_entries]
"""
import sys

import awkward as ak
import numpy as np
import uproot

from fill_calo import TREE

R, D = "calorecodigis/calorecodigis.", "calodigis/calodigis."
TICK = 5.0                     # ns per sample
PEAK = 17                      # the digitizer puts the peak at this sample
GRID = np.arange(-90.0, 450.0, 0.5)  # ns relative to the template peak
E_SEL = (8.0, 80.0)            # MeV per SiPM
ADC_SAT = 4090
AMP_BINS = [(100, 300), (300, 700), (700, 2000)]  # ADC


def load(path, nmax):
    t = uproot.open(path)[TREE]
    a = t.arrays([R + "eDep_", R + "pileUp_", R + "caloDigiIdx_", D + "waveform_", D + "peakval_"],
                 entry_stop=nmax)
    out = []
    for ev in a:
        nd = len(ev[D + "peakval_"])
        n_rd = np.bincount(ev[R + "caloDigiIdx_"].to_numpy(), minlength=nd)
        e = ev[R + "eDep_"].to_numpy()
        ok = (~ev[R + "pileUp_"].to_numpy()) & (e > E_SEL[0]) & (e < E_SEL[1])
        for k in np.flatnonzero(ok):
            di = ev[R + "caloDigiIdx_"][k]
            if n_rd[di] != 1 or ev[D + "peakval_"][di] >= ADC_SAT:
                continue
            out.append(ev[D + "waveform_"][di].to_numpy().astype(float))
    return out


def fit_one(w, tmpl, tgrid=np.arange(-6.0, 6.01, 0.25), ped=None):
    """Best (t, A, ped, chi2) for one waveform; t is the template-peak time relative to sample PEAK.
    With ped given, only t and A are fitted."""
    tau = (np.arange(len(w)) - PEAK) * TICK
    T = np.interp(tau[None, :] - tgrid[:, None], GRID, tmpl, left=0.0, right=0.0)  # (nt, ns)
    if ped is not None:
        A = (T @ (w - ped)) / (T * T).sum(1)
        chi = ((w[None, :] - ped - A[:, None] * T) ** 2).sum(1)
        i = np.argmin(chi)
        return tgrid[i], A[i], ped, chi[i]
    n = len(w)
    st, stt = T.sum(1), (T * T).sum(1)
    sw, stw = w.sum(), T @ w
    det = n * stt - st ** 2
    A = (n * stw - st * sw) / det
    p = (stt * sw - st * stw) / det
    chi = ((w[None, :] - p[:, None] - A[:, None] * T) ** 2).sum(1)
    i = np.argmin(chi)
    return tgrid[i], A[i], p[i], chi[i]


def build(wfs, tmpl):
    acc = np.zeros(len(GRID))
    cnt = np.zeros(len(GRID))
    amps = []
    for w in wfs:
        t, A, p, _ = fit_one(w, tmpl, ped=np.median(w[:4]))  # baseline from the pre-samples
        amps.append(A)
        tau = (np.arange(len(w)) - PEAK) * TICK - t
        j = np.round((tau - GRID[0]) / 0.5).astype(int)
        ok = (j >= 0) & (j < len(GRID))
        np.add.at(acc, j[ok], (w[ok] - p) / A)
        np.add.at(cnt, j[ok], 1)
    good = cnt > 20
    m = np.interp(GRID, GRID[good], acc[good] / cnt[good])
    # light smoothing (5 ns box) on the tail where the statistics are thin
    k = np.ones(11) / 11
    sm = np.convolve(m, k, mode="same")
    m = np.where(GRID > 60, sm, m)
    m[GRID > GRID[good].max()] = 0.0
    return m / m.max(), cnt, np.array(amps)


if __name__ == "__main__":
    path, outp = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000
    wfs = load(path, nmax)
    print("pulses", len(wfs))
    # start: peak-aligned mean on the 5 ns grid
    L = 40
    s = np.array([w[:L] - np.median(w[:4]) for w in wfs if len(w) >= L])
    s /= s[:, PEAK][:, None]
    tmpl = np.interp(GRID, (np.arange(L) - PEAK) * TICK, s.mean(0), left=0.0, right=0.0)
    for it in range(3):
        tmpl, cnt, amps = build(wfs, tmpl)
        print("iteration", it, "peak at", GRID[np.argmax(tmpl)], "ns")
    by_amp = []
    for lo, hi in AMP_BINS:
        sel = [w for w, A in zip(wfs, amps) if lo <= A < hi]
        by_amp.append(build(sel, tmpl)[0])
        print("amp", lo, hi, len(sel))
    np.savez_compressed(outp, grid=GRID, tmpl=tmpl, cnt=cnt, by_amp=np.array(by_amp), amp_bins=np.array(AMP_BINS),
                        tick=TICK, peak=PEAK)
