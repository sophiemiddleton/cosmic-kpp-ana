const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/nashome/s/sophie/.claude/skills/synced/43cbf15e-def1-45cd-8635-69bd16cb0773_35891fb4-20ea-4ea6-864e-ef1f5820f97f/pptx/scripts/apply_theme.js");

const PLOTS = "/exp/mu2e/app/users/sophie/newOffline/cosmic-studies/calo/plots_run125329/";
const OUT = "/exp/mu2e/app/users/sophie/newOffline/cosmic-studies/calo/stopped_muons_run125329.pptx";

const THEME = {
  name: "Mu2e cosmics",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2A36", lt1: "FFFFFF", dk2: "4A5563", lt2: "F2F4F7",
    accent1: "C0392B", accent2: "2A78D6", accent3: "EB6834", accent4: "5F87B0",
    accent5: "9AA7B4", accent6: "2E7D32", hlink: "2A78D6", folHlink: "6D2E46",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Stopped cosmic muons in the calorimeter";
const C = pres.SchemeColor;

const W = 13.333, MX = 0.6, CW = W - 2 * MX;
const FOOT = "Mu2e calorimeter · cosmic run 125329";

// ---------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 1.9, w: CW, h: 1.6, fontSize: 40, bold: true,
      color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 3.75, w: CW, h: 2.2, fontSize: 20,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.3, w: CW, h: 0.85, fontSize: 36, bold: true,
      color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.3, w: CW, h: 0.85, fontSize: 36, bold: true,
      color: C.background1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------------- helpers
function img(slide, file, x, y, w, aspect, name) {
  slide.addImage({ path: PLOTS + file, x, y, w, h: w / aspect, objectName: name, altText: name });
  return y + w / aspect;
}
function bullets(slide, items, opts) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    Object.assign({ isTextBox: true, fontSize: 15, color: C.text1, paraSpaceAfter: 6, valign: "top", margin: 0 }, opts));
}
function card(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 },
    line: { color: C.background2 }, objectName: name });
}

// ================= 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Stopped cosmic muons in the calorimeter", { placeholder: "title" });
s.addText([
  { text: "Muon lifetime and Michel spectrum, cosmic run 125329", options: { breakLine: true } },
  { text: "Extracted position, no field · calorimeter only, no MC · 10.3 M events, 10.8 h", options: { breakLine: true, fontSize: 16 } },
  { text: " ", options: { breakLine: true, fontSize: 12 } },
  { text: "29 October 2026", options: { fontSize: 16 } },
], { placeholder: "body" });
s.addNotes("Studies of cosmic muons that stop in the calorimeter disks, using run 125329 in the extracted position (no solenoid field). Everything is data-driven: there is no cosmic MC yet. Main topics: the muon lifetime, which was low in the first pass, and the Michel spectrum.");

// ================= 2. summary
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Summary", { placeholder: "title" });
const stats = [
  ["2.201 ± 0.015", "µs: muon lifetime of the stop sample", "PDG τ(µ⁺) = 2.197 µs"],
  ["44.2 ± 0.2", "MeV: Michel edge E₀ from hit groups", "38.8 MeV from reco clusters"],
  ["8.4%", "of Michel decays deposit < 10 MeV", "lost by the cluster threshold"],
];
const cw3 = (CW - 2 * 0.4) / 3;
stats.forEach(([big, lab, sub], i) => {
  const x = MX + i * (cw3 + 0.4);
  card(s, x, 1.5, cw3, 2.4, `stat card ${i + 1}`);
  s.addText(big, { x: x + 0.25, y: 1.65, w: cw3 - 0.5, h: 0.95, fontSize: 30, bold: true, color: C.accent1,
    fontFace: THEME.headFontFace, isTextBox: true, margin: 0, valign: "middle" });
  s.addText(lab, { x: x + 0.25, y: 2.65, w: cw3 - 0.5, h: 0.7, fontSize: 15, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
  s.addText(sub, { x: x + 0.25, y: 3.4, w: cw3 - 0.5, h: 0.4, fontSize: 13, color: C.text2, italic: true, isTextBox: true, margin: 0 });
});
bullets(s, [
  "The first-pass τ = 2.117 µs and its 1.4 µs second component came from how the prompt muon was defined, not from physics",
  "Reco clusters split one cosmic into several same-time pieces, and SiPM time offsets add echo clusters: tracks appear to end early",
  "Tracks built from SiPM-corrected hits fix both, and the lifetime agrees with PDG. Hit groups that count each crystal once give a Michel spectrum ending at the 52.8 MeV endpoint",
], { x: MX, y: 4.35, w: CW, h: 2.3 });
s.addNotes("Three numbers to take away. The lifetime now agrees with PDG. The Michel edge from hit groups is about 5 MeV higher and sharper than from clusters. And the 10 MeV cluster threshold loses 8% of the decays. The rest of the talk is how we got there.");

// ================= 3. starting point
pres.addSection({ title: "Diagnosis" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Diagnosis" });
s.addText("Where we started: cluster pairs", { placeholder: "title" });
img(s, "01_muon_lifetime.png", MX, 1.4, 5.6, 1.193, "First-pass lifetime fit");
img(s, "04_michel_spectrum.png", 6.6, 1.4, CW - 6.0, 2.426, "First-pass Michel spectrum");
bullets(s, [
  "Prompt = any earlier cluster on the same disk within 80 mm; each delayed cluster paired with the closest one",
  "τ = 2.117 ± 0.008 µs from 1.25 µs, rising to 2.148 ± 0.018 µs from 4 µs",
  "Two-component fit: 9.5% with τ₂ = 1.44 ± 0.22 µs, read as µ⁻ stopping in light materials",
  "Michel edge E₀ = 38.8 ± 0.2 MeV, σ = 6.2 MeV (clusters ≥ 10 MeV)",
], { x: 6.6, y: 4.15, w: CW - 6.0, h: 2.6, fontSize: 14 });
s.addNotes("This is the first pass from the README, sections 2 and 3. The lifetime was 4% low and grew when the fit started later, which is the signature of a short-lived component. We interpreted it as mu-minus in light materials. The Michel spectrum from clusters had its edge at 38.8 MeV.");

// ================= 4. asymmetry diagnostic
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Diagnosis" });
s.addText("Decay electrons sat ahead of the stop", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{
  name: "asymmetry",
  labels: ["Reco clusters", "Clusters merged in time", "SiPM-corrected hit tracks"],
  values: [-0.62, -0.29, -0.04],
}], {
  x: MX, y: 1.4, w: 6.6, h: 4.9, barDir: "col", objectName: "Up-down asymmetry by method",
  chartColors: [THEME.colors.accent1],
  showTitle: true, title: "(up − down) / (up + down) of the Michel position",
  titleFontSize: 14, titleColor: THEME.colors.dk1, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 14,
  dataLabelColor: THEME.colors.dk1, dataLabelFontFace: "+mn-lt",
  valAxisMinVal: -0.8, valAxisMaxVal: 0.1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
  valAxisLabelColor: THEME.colors.dk2, valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt",
  catAxisLabelColor: THEME.colors.dk1, catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt",
  catAxisLabelPos: "low",
  valGridLine: { color: "E3E3E3", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false,
});
bullets(s, [
  "A muon at rest emits its positron isotropically: up and down should match to within the polarization effect (≲ 0.1)",
  "With reco clusters, decay electrons were 4× more often below the \"stop\" than above it: the track went on below the crystal we called its end",
  "The same in every decay-time bin and for single-crystal clusters, so not a channel-recovery effect",
], { x: 7.6, y: 1.5, w: CW - 7.0, h: 4.2 });
s.addText("All candidates; Michel groups > 3 MeV within 80 mm, 1.5–9 µs, sideband-subtracted. Middle bar from a 400 k-event test sample.",
  { x: 7.6, y: 5.85, w: CW - 7.0, h: 0.6, fontSize: 11, color: C.text2, italic: true, isTextBox: true, margin: 0 });
s.addNotes("The key diagnostic. If the stop crystal is right, Michel deposits are symmetric around it. They were not: minus 0.62 overall and minus 0.69 in the purest selection. Merging same-time clusters halves it; correcting SiPM times and building tracks from hits removes it. The left-right control was zero in the first and final versions.");

// ================= 5. two causes
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Diagnosis" });
s.addText("Two ways a track ends early", { placeholder: "title" });
const y5 = img(s, "26_cluster_fragmentation.png", MX + 0.35, 1.3, CW - 0.7, 2.782, "Cluster fragmentation");
const half = (CW - 0.4) / 2;
[["Split clusters", "21% of clusters above 15 MeV have a same-disk partner within 30 ns, typically 1–4 crystals away. The upper piece looked like a stop."],
 ["SiPM time offsets", "The ±170 ns peaks are echo clusters from SiPMs shifted by k × 169 ns (283 crystals). A shifted crystal drops out of the track."]]
  .forEach(([h, t], i) => {
    s.addText([{ text: h, options: { bold: true, breakLine: true, color: C.accent1 } }, { text: t }],
      { x: MX + i * (half + 0.4), y: y5 + 0.2, w: half, h: 6.85 - y5 - 0.2, fontSize: 15, color: C.text1, isTextBox: true,
        margin: 0, valign: "top" });
  });
s.addNotes("Left: time difference between a muon cluster and every other cluster on the same disk. The central peak is the same cosmic split into pieces. The side peaks at plus or minus 170 ns are the echo clusters from the SiPM time-calibration offsets. Right: the split pieces are mostly one to four crystals apart, so they are fragments of one track, not separate muons.");

// ================= 6. new method
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("New stop finder: tracks from corrected hits", { placeholder: "title" });
const steps = [
  ["Correct hit times", "Per-SiPM offsets from sipm_abs_offsets.py, for single-SiPM and merged hits"],
  ["Chain hits into tracks", "Same disk, time gaps < 30 ns; reco clusters are not used"],
  ["Pick the stop", "Lowest end of the energy-weighted track axis; reject if an earlier muon is within 25 µs and 150 mm"],
  ["Collect delayed hits", "Within 150 mm, −20 to +60 µs, grouped in time; a split crystal counts once"],
];
const sw = (CW - 3 * 0.3) / 4;
steps.forEach(([h, t], i) => {
  const x = MX + i * (sw + 0.3);
  card(s, x, 1.6, sw, 3.4, `step ${i + 1}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.85, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 },
    objectName: `step ${i + 1} marker` });
  s.addText(String(i + 1), { x: x + 0.3, y: 1.85, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1,
    align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText(h, { x: x + 0.3, y: 2.65, w: sw - 0.6, h: 0.75, fontSize: 18, bold: true, color: C.text1,
    fontFace: THEME.headFontFace, isTextBox: true, margin: 0, valign: "top" });
  s.addText(t, { x: x + 0.3, y: 3.45, w: sw - 0.6, h: 1.45, fontSize: 14, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
});
s.addText([
  { text: "6.53 M muon candidates", options: { bold: true } },
  { text: "   ·   stop sample (last-crystal energy 25–70 MeV, ≥ 40 mm of disk ahead): " },
  { text: "662 k", options: { bold: true } },
], { x: MX, y: 5.4, w: CW, h: 0.6, fontSize: 16, color: C.text1, isTextBox: true, margin: 0 });
s.addText("fill_stops.py: about 2 min for the run on 4 cores", { x: MX, y: 6.0, w: CW, h: 0.4, fontSize: 12,
  color: C.text2, italic: true, isTextBox: true, margin: 0 });
s.addNotes("The new stop finder never uses reco clusters. Hit times are corrected with the per-SiPM offset table measured in section 1. Hits on the same disk with gaps under 30 ns form a track. The stop is the lowest end of the track, since cosmics go down. A track with another muon shortly before it nearby is rejected, which removes Michel electrons and leftover echoes as prompts.");

// ================= 7. purity
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Most track ends are not stops", { placeholder: "title" });
const y7 = img(s, "20_stop_selection.png", MX, 1.3, CW, 3.483, "Stop selection");
bullets(s, [
  "The calorimeter can't see a muon leave through a disk face: the Michel rate per candidate is 0.4% at low last-crystal energy and 7–8% at the Bragg peak (40–50 MeV) with ≥ 80 mm of disk ahead",
  "Stop sample: 3.7% Michel per candidate, so roughly 14% real stops (window efficiency 0.49, f(µ⁺) ≈ 0.55)",
  "Right: the Michel position still shifts with last-crystal energy; the stop is known only to about one crystal",
], { x: MX, y: y7 + 0.3, w: CW, h: 6.85 - y7 - 0.3, fontSize: 14 });
s.addNotes("Purity is the main limitation of a calorimeter-only stop. A muon that exits through the front or back face looks just like a stop. Last-crystal energy is the best handle: a Bragg peak in the last crystal means the muon really ended there. The purity matters for the charge ratio, but not for the lifetime, which only needs the Michel-tagged events.");

// ================= 8. lifetime
pres.addSection({ title: "Lifetime" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Lifetime" });
s.addText("Lifetime agrees with PDG", { placeholder: "title" });
const y8 = img(s, "21_stop_lifetime.png", MX, 1.3, CW, 3.63, "Stop-sample lifetime");
const third = (CW - 2 * 0.3) / 3;
[["Fit 1.5–20 µs", "Michel groups 10–60 MeV within 80 mm: τ = 2.201 ± 0.015 µs, χ²/ndf = 66/66"],
 ["By last-crystal energy", "Split by the stop-purity proxy: no trend, all within about 1σ of 2.197 µs"],
 ["By fit start", "Flat within errors (points share data), so no short-lived second component"]]
  .forEach(([h, t], i) => {
    s.addText([{ text: h, options: { bold: true, breakLine: true, color: C.accent1 } }, { text: t }],
      { x: MX + i * (third + 0.3), y: y8 + 0.3, w: third, h: 1.5, fontSize: 14, color: C.text1, isTextBox: true,
        margin: 0, valign: "top" });
  });
s.addNotes("Left: the stop-sample fit, starting at 1.5 microseconds to stay clear of the post-muon soft-hit burst and the echo region. Middle: if mu-minus in light material were pulling tau down, the low last-crystal-energy bins would sit low; they don't. Right: a second component would make tau rise with the fit start; it is flat within errors. The digitizer clock scale is fine.");

// ================= 9. method vs sample
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Lifetime" });
s.addText("It was the method, not the sample", { placeholder: "title" });
const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } });
const rows9 = [
  [hdr("Sample"), hdr("τ from 1.25 µs"), hdr("2nd component")],
  ["Cluster pairs (first pass)", "2.117 ± 0.008 µs", "9.5%, τ₂ = 1.44 µs"],
  ["New: all candidates, no purity cut", "2.209 ± 0.011 µs", "none"],
  ["New: stop sample", "2.218 ± 0.015 µs", "none"],
  ["New: stops < 40 mm from the disk edge", "2.12 ± 0.04 µs", "not significant"],
];
s.addTable(rows9, { x: MX, y: 1.5, w: 7.4, colW: [3.5, 2.0, 1.9], fontSize: 15, color: C.text1,
  border: { type: "solid", pt: 0.75, color: "D5DAE1" }, rowH: 0.55, valign: "middle", objectName: "Lifetime comparison" });
bullets(s, [
  "Even with no purity cut the new method gives the PDG value, so the purity cuts are not hiding anything",
  "The bias sits in the first-pass prompt definition (fragments, echoes, Michels as prompts); which of these produced the 1.4 µs component is not yet isolated",
  "Stops near the disk edge are 2σ low: possibly µ⁻ in the outer support, still open",
], { x: 8.5, y: 1.5, w: CW - 7.9, h: 4.9, fontSize: 15 });
s.addNotes("Same data, different prompt definition. Without any purity cut the new method already gives 2.21 microseconds with no second component, so the low first-pass value was a reconstruction effect. Stops near the edge are about 2 sigma low, which could be real mu-minus capture in the outer support ring, but it is only 15% of the decays. Next step: rerun the old pair method adding fragments, echoes and Michel prompts back one at a time.");

// ================= 10. Michel spectrum
pres.addSection({ title: "Michel spectrum" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Michel spectrum" });
s.addText("Michel spectrum from hit groups", { placeholder: "title" });
const y10 = img(s, "22_michel_lowE.png", MX + 0.35, 1.3, CW - 0.7, 2.669, "Michel spectrum from hit groups");
bullets(s, [
  "Hit groups sum all hits within 150 mm and 30 ns, down to 0.5 MeV, with each crystal counted once; background from the 15–22.5 µs sideband",
  "The spectrum ends at the 52.8 MeV endpoint; 8.4% of decays fall below the 10 MeV cluster threshold, 0.3% below 3 MeV",
], { x: MX, y: y10 + 0.2, w: CW, h: 6.85 - y10 - 0.2, fontSize: 14 });
s.addNotes("Using hit groups instead of clusters lowers the threshold to about half an MeV. Each crystal counts once: the two single-SiPM hits of a split crystal each estimate the full crystal energy, and summing both had pushed the spectrum past the endpoint. The right panel gives the fraction of decays below a given threshold, which is useful for any cluster-based Michel analysis.");

// ================= 11. edge comparison
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Michel spectrum" });
s.addText("Michel edge: clusters vs hit groups", { placeholder: "title" });
const rows11 = [
  [hdr("Method"), hdr("Decays"), hdr("E₀ [MeV]"), hdr("σ [MeV]")],
  ["Reco clusters ≥ 10 MeV, ΔR < 80 mm (first pass)", "90 k", "38.80 ± 0.23", "6.18 ± 0.13"],
  ["22/10/26 slides", "–", "42.80 ± 0.12", "5.13 ± 0.10"],
  ["Hit groups, stop sample, fit 30–52 MeV", "26.9 k", "44.16 ± 0.16", "4.28 ± 0.13"],
  ["Hit groups, stop sample, fit 35–55 MeV", "26.9 k", "45.04 ± 0.16", "3.50 ± 0.10"],
];
s.addTable(rows11, { x: MX, y: 1.5, w: CW, colW: [6.0, 1.6, 2.25, 2.28], fontSize: 15, color: C.text1,
  border: { type: "solid", pt: 0.75, color: "D5DAE1" }, rowH: 0.55, valign: "middle", objectName: "Michel edge comparison" });
const half11 = (CW - 0.4) / 2;
card(s, MX, 4.6, half11, 2.05, "edge interpretation");
s.addText([
  { text: "Why the edge moves up", options: { bold: true, breakLine: true, color: C.accent1 } },
  { text: "Hit groups keep the energy that split clusters lose, so the edge rises by about 5 MeV and sharpens. This is the likely cause; it is not checked against MC." },
], { x: MX + 0.3, y: 4.8, w: half11 - 0.6, h: 1.7, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
card(s, MX + half11 + 0.4, 4.6, half11, 2.05, "edge caveats");
s.addText([
  { text: "What it can't tell us yet", options: { bold: true, breakLine: true, color: C.accent1 } },
  { text: "E₀ moves by about 0.9 MeV with the fit range. Without MC, E₀ can't be converted to an absolute electron energy scale: leakage and threshold effects are unknown." },
], { x: MX + half11 + 0.7, y: 4.8, w: half11 - 0.6, h: 1.7, fontSize: 14, color: C.text1, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("The cluster edge is low and broad because clusters miss energy: the clustering splits showers just as it splits muon tracks, and there is a 10 MeV threshold. Hit groups recover that energy. The fit-range dependence is about 0.9 MeV, larger than the statistical error. To turn this into an energy calibration we need cosmic MC through the same chain.");

// ================= 12. conclusions
pres.addSection({ title: "Conclusions" });
s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Conclusions" });
s.addText("Conclusions and next steps", { placeholder: "title" });
const colW = (CW - 0.6) / 2;
s.addText("Conclusions", { x: MX, y: 1.45, w: colW, h: 0.5, fontSize: 22, bold: true, color: C.accent3,
  fontFace: THEME.headFontFace, isTextBox: true, margin: 0 });
bullets(s, [
  "Reco clusters can't define a stopped muon in cosmics: tracks are split in time and by SiPM offsets",
  "With SiPM-corrected hit tracks, τ = 2.201 ± 0.015 µs, consistent with PDG; the clock scale is fine",
  "Michel edge E₀ = 44.2 ± 0.2 (±0.9 from fit range) MeV, σ = 4.3 MeV; 8.4% of decays fall below 10 MeV",
], { x: MX, y: 2.05, w: colW, h: 4.6, color: C.background1, fontSize: 16, paraSpaceAfter: 10 });
s.addText("Next steps", { x: MX + colW + 0.6, y: 1.45, w: colW, h: 0.5, fontSize: 22, bold: true, color: C.accent3,
  fontFace: THEME.headFontFace, isTextBox: true, margin: 0 });
bullets(s, [
  "Isolate which part of the first-pass prompt definition produced the 1.4 µs component",
  "Fix the per-SiPM time constants and check the clustering time window against them",
  "Follow up the edge stops (2.12 ± 0.04 µs): µ⁻ in support material?",
  "Cosmic MC through the same chain, to turn E₀ into an electron energy scale",
], { x: MX + colW + 0.6, y: 2.05, w: colW, h: 4.6, color: C.background1, fontSize: 16, paraSpaceAfter: 10 });
s.addNotes("The clustering and SiPM-time issues affect any cosmic analysis that relies on reco clusters, so they are worth fixing upstream. The Michel edge is now a usable, self-consistent number, but it needs MC before it becomes a calibration.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
