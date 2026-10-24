# Faculty Format Checklists

## PPT Checklist (from Presentation_Format.pdf)

| Requirement | Applies as-is? | Our adaptation |
|---|---|---|
| 10-12 min + 3-5 min Q&A | Yes | — |
| 12-15 slides | Yes | — |
| Page numbers on every slide | Yes | Must add when finalizing in PPT software |
| Slide 1: title + group members | Yes | — |
| Slide 2: intro, background, problem area | Yes | — |
| Slide 3: importance + scope | Yes | — |
| Slide 4: hardware components + pictures | **No hardware exists** (confirmed: software-only project) | Substituted with **software/dataset requirements**: libraries, official trackml library, dataset access |
| Slide 5: block diagram, circuit diagram, working principle, flowchart | Circuit diagram N/A | Substituted **circuit diagram → pipeline/data-flow diagram**; block diagram, working principle, flowchart all directly applicable to our software pipeline |
| Slide 6: conclusion & discussion | Yes | — |

**Note on the substitution**: confirmed with the student directly that this
is a dataset-only project with no hardware component (per the official
Project 35 specification: "No hardware, no cost"). The substitution
(software/library requirements instead of hardware; pipeline diagram
instead of circuit diagram) preserves the *intent* of each faculty section
(what does the project need to run? how does it work internally?) while
fitting a software-only project. This should still be mentioned briefly to
the mentor if asked, but does not require prior sign-off per the student.

## Report Checklist (from Table_of_content__Pinnacle_.pdf)

| Chapter | Applies as-is? | Our adaptation |
|---|---|---|
| First page (title, group members) | Yes | — |
| Dedication, Abstract, Acknowledgements, Preface | Yes | Standard front matter |
| Table of Contents, List of Figures, List of Tables | Yes | Generate once report is finalized |
| I. Introduction (Purpose, Conceptual Framework, Problem Definition, Significance, Scope) | Yes | Directly maps to Content Master sections B-F |
| II. Review of Related Literature (5-10 papers) | Yes | 7 sources identified (Content Master Section G); can extend toward 10 |
| III. Requirement Analysis (3.1 Hardware+software, 3.2 hardware component list, 3.3 component explanations) | Hardware sub-sections N/A | 3.1 → **software/library requirements** (Python 3.12, pandas, numpy, scipy, scikit-learn, matplotlib, official trackml library); 3.2/3.3 → **dataset as the "component"**: TrackML hits/cells/particles/truth files, explained individually |
| IV. System Architecture & Methodology (4.1 block diagram, 4.2 circuit diagram explanation, 4.3 working principle, 4.4 flowchart/algorithm) | 4.2 circuit diagram N/A | 4.1 → repository/pipeline architecture diagram; 4.2 → **data-flow diagram explanation** (raw hits → transform → cluster → score); 4.3/4.4 directly applicable |
| V. Discussions | Yes | Honest discussion of what's done vs. pending, per Content Master |
| VI. Conclusion | Yes | — |
| VII. Bibliography, VIII. References | Yes | From Content Master Section G/U; citation style **TO BE COMPLETED** — confirm with mentor |
| IX. Appendix incl. coding sheet | Yes | Real code from `src/`, `tests/` |

## Open items requiring your mentor's input (not resolved by us)

1. Exact citation/reference style expected (APA, IEEE, etc.) — not specified in either uploaded format document.
2. Whether the 5-10 paper literature review should lean more physics-methodology or more ML-methodology in emphasis.
3. Confirm the substitution approach above is acceptable, if asked directly.
