"""Characterise echo pairs (dt in a comb spike) vs off-spike pairs at digi level."""
import sys
import awkward as ak
import numpy as np
import uproot

PATH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
t = uproot.open(PATH)["EventNtuple/ntuple"]
C, Hh, R, D = "caloclusters/caloclusters.", "calohits/calohits.", "calorecodigis/calorecodigis.", "calodigis/calodigis."
a = t.arrays([C + x for x in ("time_", "energyDep_", "diskID_", "cog_.fCoordinates.fX", "cog_.fCoordinates.fY")]
             + [Hh + x for x in ("crystalId_", "time_", "eDep_", "clusterIdx_", "recoDigis_")]
             + [R + x for x in ("caloDigiIdx_", "SiPMID_", "time_")]
             + [D + x for x in ("SiPMID_", "t0_", "waveform_")], entry_stop=N)
rows = []
for ev in a:
    ct = ev[C + "time_"].to_numpy()
    if len(ct) < 2:
        continue
    disk, x, y = (ev[C + k].to_numpy() for k in ("diskID_", "cog_.fCoordinates.fX", "cog_.fCoordinates.fY"))
    hcl = ev[Hh + "clusterIdx_"].to_numpy()
    hcry = ev[Hh + "crystalId_"].to_numpy()
    rdig = ev[Hh + "recoDigis_"]
    rsipm = ev[R + "SiPMID_"].to_numpy()
    rd2d = ev[R + "caloDigiIdx_"].to_numpy()
    dt0 = ev[D + "t0_"].to_numpy()
    wf = ev[D + "waveform_"]
    for p in range(len(ct)):
        for q in range(len(ct)):
            dt = ct[q] - ct[p]
            if not (100 < dt < 1300) or disk[p] != disk[q] or np.hypot(x[q] - x[p], y[q] - y[p]) > 80:
                continue
            spike = any(abs(dt - s) < 12 for s in (172, 335, 505, 675, 845, 1007, 1177))
            pc = set(hcry[hcl == p])
            qh = np.flatnonzero(hcl == q)
            same_cry = np.mean([hcry[h] in pc for h in qh])
            # per SiPM in the echo cluster: does the prompt cluster have a digi on the same SiPM?
            psipm = {}
            for h in np.flatnonzero(hcl == p):
                for r in rdig[h]:
                    psipm[rsipm[r]] = rd2d[r]
            gaps, base, same_sipm = [], [], []
            for h in qh:
                for r in rdig[h]:
                    d = rd2d[r]
                    w = wf[d].to_numpy()
                    base.append(w[:3].mean() - w[-3:].mean())  # >0 if waveform starts above its own end
                    if rsipm[r] in psipm:
                        dp = psipm[rsipm[r]]
                        same_sipm.append(1)
                        gaps.append(dt0[d] - (dt0[dp] + len(wf[dp])))
                    else:
                        same_sipm.append(0)
            rows.append((spike, dt, same_cry, np.mean(same_sipm) if same_sipm else np.nan,
                         np.median(gaps) if gaps else np.nan, np.median(base)))
r = np.array(rows, float)
for lab, m in (("SPIKE pairs", r[:, 0] == 1), ("off-spike pairs", r[:, 0] == 0)):
    s = r[m]
    print(f"{lab}: n={m.sum()}")
    print(f"   frac echo crystals also in prompt cluster: mean {np.nanmean(s[:,2]):.2f}")
    print(f"   frac echo SiPMs also in prompt cluster:    mean {np.nanmean(s[:,3]):.2f}")
    g = s[:, 4][np.isfinite(s[:, 4])]
    if len(g):
        print(f"   echo t0 - (prompt t0 + nsamp) [samples]: pct 5/25/50/75/95 {np.percentile(g,[5,25,50,75,95])}")
        print(f"   hist of gap:", np.unique(g.astype(int), return_counts=True))
    print(f"   waveform start-minus-end ADC: median {np.median(s[:,5]):.1f}")
