# Mid-Sem Progress Presentation — Track Reconstruction (TrackML)

Format: exactly per `Presentation_Format.pdf` (6 sections, 12-15 slides,
10-12 min + 3-5 min Q&A). Hardware sections substituted with software
equivalents per direct confirmation this is a dataset-only project (see
`FACULTY_FORMAT_CHECKLISTS.md`).

Every fact below traces to `PROJECT_CONTENT_MASTER.md`. `[SYNTHETIC DATA]`
and `[TO BE COMPLETED]` labels are load-bearing — do not remove them when
turning this into slides.

---

## Slide 1 — Title

**Content:**
- Track Reconstruction (TrackML)
- Pinnacle 3: Machine Learning — Particle Physics
- [Your name, roll no., branch/section, faculty mentor — from Content Master Section A]

**Visual:** University/school header per report template's first page.

---

## Slide 2 — Project Introduction

**Content:**
- Particle colliders (e.g. LHC) smash protons together; each collision
  ("event") produces thousands of new particles.
- A layered detector records each particle's position as a "hit" —
  a 3D point — every time it crosses a layer.
- **Problem:** the detector returns an unordered pile of hit points with no
  labels. We must figure out which points belong to which particle.

**Visual:** A simple diagram — collision point at center, particles
radiating outward through concentric detector layers, each leaving a dot.
(*If a real figure isn't available yet, describe verbally over a simple
hand-drawn/labeled schematic — do not fabricate a "real detector" image.*)

**Speaker guidance embedded:** Say this is fundamentally a **clustering**
problem, not classification — there's no fixed number of "classes."

---

## Slide 3 — Problem Statement and Scope

**Content:**
- **Problem statement:** Group ~100,000 unlabeled 3D hits per event into
  ~10,000 particle tracks, using only geometric information (never given
  particle identity).
- **Why it matters:** Future colliders (HL-LHC) will produce far denser
  events; current combinatorial algorithms scale worse than quadratically.
  Faster ML-based grouping is an active research need.
- **Scope:** Laptop-CPU feasible. Ready-made tools only (pandas,
  scikit-learn). No novel neural architecture. Dataset-only project — no
  hardware.

**Visual:** Simple text/bullet slide, possibly a "scale" callout
(100,000 hits → 10,000 tracks) for impact.

---

## Slide 4 — Requirement Analysis *(software/dataset, adapted from hardware template)*

**Content:**

*Software requirements:*
| Component | Purpose |
|---|---|
| Python 3.12 | Base language |
| pandas, numpy, scipy | Data handling, numerics |
| scikit-learn | Clustering (DBSCAN) |
| matplotlib | Visualization |
| Official `trackml` library (from GitHub, **not** PyPI) | Correct, verified CSV parsing — a real trap discovered and avoided in this project |

*Dataset "components":*
| File | Contains |
|---|---|
| `hits.csv` | 3D hit positions + detector location IDs |
| `cells.csv` | Raw charge-deposition detail (not yet used) |
| `particles.csv` | True particle momentum, charge, vertex |
| `truth.csv` | hit → particle mapping + scoring weight (**never given to the clustering algorithm — evaluation only**) |

**Visual:** Table as above; optionally a small icon/logo row for the
libraries used.

**Speaker guidance:** Flag explicitly that there's no physical hardware —
this project uses a *public dataset* as its "component," and be ready to
explain the PyPI-name-collision trap as a concrete example of engineering
diligence (shows real depth for the viva).

---

## Slide 5 — System Architecture & Methodology

**Content:**

*Pipeline (data-flow) diagram, adapted from "circuit diagram":*
```
TrackML hits (x, y, z)
      |
Data validation & truth sanity checks   <- COMPLETED (synthetic), IN PROGRESS (real)
      |
Coordinate transformation                <- PROPOSED
      |
DBSCAN clustering                        <- PROPOSED
      |
Track candidates (hit -> predicted track_id)
      |
TrackML weighted scoring (truth used ONLY here)  <- PROPOSED
      |
Efficiency analysis (vs. pT, vs. occupancy)      <- PROPOSED
```

*Working principle (brief):*
- Charged particles bend in the detector's magnetic field, following
  helical paths. A coordinate transform makes these approximately
  straight, so density-based clustering (DBSCAN) can group hits by
  proximity.
- Truth labels are used **only for evaluation**, never fed into the
  clustering step itself — this boundary is enforced in code.

**Visual:** The pipeline diagram above, color-coded by status
(green=done, yellow=in progress, grey=proposed).

**Speaker guidance:** This is the slide to spend the most time on. Walk
through each stage, being explicit about what's real vs. planned.

---

## Slide 6 — Conclusion & Discussion (Progress + Remaining Work)

**Content:**

*What's genuinely done:*
- Repository architecture built, reproducible environment verified.
- Data loader, physics utilities, truth-consistency checks — all
  implemented, all covered by **39 passing unit tests**.
- Full exploration notebook, executed end-to-end, 0 errors.
- Correct dataset downloaded and confirmed (`train_sample.zip`, all four
  file types present).

*Important honesty note for this slide:*
- All numeric results shown anywhere in this project so far are
  **`[SYNTHETIC DATA]`** — generated by a small synthetic-event script
  built specifically to test the pipeline before real data was available.
  **No real event has been loaded by our code yet.**

*Remaining work (next steps, in order):*
1. Load and validate real event(s) — immediate next step.
2. Implement coordinate transformation (candidate identified from
   official starting kit).
3. Implement DBSCAN baseline.
4. Implement + unit-test the official TrackML scoring metric.
5. Run baseline experiment on real events; compare to random baseline.
6. Efficiency analysis vs. pT and occupancy.

*Proposed research question (not a novelty claim):*
> How strongly does coordinate representation affect density-based track
> reconstruction quality across different momentum and occupancy regimes?

**Visual:** A simple progress table (see below) — this is the single most
important visual for demonstrating credible, honest progress.

| Component | Status |
|---|---|
| Dataset selected & downloaded | COMPLETE |
| Data loading pipeline (code) | COMPLETE |
| Data validation (code) | COMPLETE |
| Data loading/validation (on real data) | IN PROGRESS |
| EDA / exploration notebook | COMPLETE (synthetic) |
| Coordinate transformation | NOT STARTED |
| Baseline clustering | NOT STARTED |
| Evaluation metric | NOT STARTED (formula + toy example derived) |
| pT / occupancy analysis | NOT STARTED |
| Optimization | NOT STARTED |

---

## Notes for building the actual PPT file

- Every `[SYNTHETIC DATA]` or `NOT STARTED` label above must survive into
  the final slides verbatim, or as an equally clear visual cue (e.g. a
  grey/red status chip) — do not soften this into ambiguous language.
- If your mentor asks "what's your biggest risk right now," the honest
  answer is: the TrackML scoring metric is explicitly documented (by the
  challenge organizers themselves) as easy to implement incorrectly — this
  is why our plan is to unit-test it against a hand-constructed example
  before trusting any real score.
