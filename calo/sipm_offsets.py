"""Per-crystal time offset between the two SiPMs, from crystals whose two SiPMs fired as separate hits.

Usage: python sipm_offsets.py <ntuple.root> <out.npz> [max_entries]
"""
import sys

import awkward as ak
import numpy as np
import uproot

PATH, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
NCRY = 1348
B = np.arange(-1500, 1501, 5.0)
t = uproot.open(PATH)["EventNtuple/ntuple"]
H, R = "calohits/calohits.", "calorecodigis/calorecodigis."
hist = np.zeros((NCRY, len(B) - 1))
nboth = np.zeros(NCRY)
nsingle = np.zeros((NCRY, 2))
for a in t.iterate([H + "crystalId_", H + "time_", H + "eDep_", H + "nSiPMs_", H + "recoDigis_", R + "SiPMID_"],
                   entry_stop=N, step_size=100000):
    first = ak.firsts(a[H + "recoDigis_"], axis=2)
    sipm = a[R + "SiPMID_"][ak.fill_none(first, 0)]
    h = ak.zip({"cry": a[H + "crystalId_"], "t": a[H + "time_"], "e": a[H + "eDep_"], "ns": a[H + "nSiPMs_"],
                "side": sipm % 2})
    hf = {k: ak.flatten(h[k]).to_numpy() for k in h.fields}
    np.add.at(nboth, hf["cry"][hf["ns"] == 2], 1)
    s1 = hf["ns"] == 1
    np.add.at(nsingle, (hf["cry"][s1], hf["side"][s1]), 1)
    one = h[(h.ns == 1) & (h.e > 3)]
    ij = ak.argcartesian([one.t, one.t])
    p, q = one[ij["0"]], one[ij["1"]]
    m = (p.cry == q.cry) & (p.side == 0) & (q.side == 1)
    cry = ak.flatten(p.cry[m]).to_numpy()
    dt = ak.flatten(q.t[m] - p.t[m]).to_numpy()
    ok = np.abs(dt) < B[-1]
    np.add.at(hist, (cry[ok], np.digitize(dt[ok], B) - 1), 1)
np.savez_compressed(OUT, hist=hist, B=B, nboth=nboth, nsingle=nsingle)
c = 0.5 * (B[1:] + B[:-1])
tot = hist.sum(0)
print("all crystals, dt(odd SiPM - even SiPM) peaks:")
for i in np.argsort(tot)[::-1][:12]:
    print(f"   {c[i]:7.1f} ns  {int(tot[i])}")
pk = np.where(hist.sum(1) > 20, c[np.argmax(hist, 1)], np.nan)
print("per-crystal modal offset, unique values (rounded to 5 ns) and counts:")
u, n = np.unique(np.round(pk[np.isfinite(pk)] / 5) * 5, return_counts=True)
print(list(zip(u, n)))
print("crystals with enough pairs:", np.isfinite(pk).sum(), "/", NCRY)
print("hits with both SiPMs:", int(nboth.sum()), " single even:", int(nsingle[:, 0].sum()), " single odd:", int(nsingle[:, 1].sum()))
