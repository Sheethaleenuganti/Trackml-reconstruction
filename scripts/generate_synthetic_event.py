#!/usr/bin/env python3
"""Generate a small, CLEARLY SYNTHETIC event for exercising the loader,
physics module, and tests when real TrackML data is not available.

============================================================================
IMPORTANT — READ BEFORE USING
============================================================================
This is NOT real TrackML data. It is NOT downloaded from anywhere. It is a
simple geometric simulation written for this project, invented specifically
to have the right SHAPE (same column names, same dtypes, same file-naming
convention as real TrackML events) so that infrastructure code can be
exercised end-to-end before real data is available.

It must NEVER be used to report "real" profiling numbers, physics
conclusions, or anything that goes into the course report as if it came
from the actual TrackML dataset. Every artifact derived from this generator
should be labeled '[SYNTHETIC DATA]' wherever it is displayed, exactly as
done in notebooks/01_dataset_exploration.ipynb.

Physical model used (deliberately simple, NOT the real detector geometry):
  - A fixed number of straight/helical "particles" originate near (0,0,0).
  - Each particle is assigned a random direction and a charge.
  - Each particle is stepped through a small number of concentric
    cylindrical "layers" (a crude stand-in for the real 10-layer silicon
    detector described in Step 2), leaving one hit per layer it crosses.
  - A configurable fraction of additional pure-noise hits (particle_id = 0)
    is added at random positions, echoing the ~15% noise fraction reported
    in the literature (Step 2, Part J) — but this specific number is a
    stylistic echo, not a claim that it matches real TrackML noise
    statistics.
============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYNTHETIC_TAG = "SYNTHETIC"  # embedded in output for traceability


def _generate_particles(n_particles: int, rng: np.random.Generator) -> pd.DataFrame:
    # Random directions on a sphere, random momentum magnitude in a
    # physically-plausible-looking but arbitrary GeV/c range.
    phi = rng.uniform(-np.pi, np.pi, n_particles)
    cos_theta = rng.uniform(-0.9, 0.9, n_particles)
    sin_theta = np.sqrt(1 - cos_theta**2)

    p_mag = rng.uniform(0.2, 10.0, n_particles)  # GeV/c, arbitrary spread
    px = p_mag * sin_theta * np.cos(phi)
    py = p_mag * sin_theta * np.sin(phi)
    pz = p_mag * cos_theta

    particle_id = np.arange(1, n_particles + 1, dtype=np.int64)
    q = rng.choice([-1, 1], size=n_particles).astype(np.int32)

    particles = pd.DataFrame(
        {
            "particle_id": particle_id,
            "particle_type": np.full(n_particles, -1, dtype=np.int32),  # not a real PDG code
            "vx": np.zeros(n_particles, dtype=np.float32),
            "vy": np.zeros(n_particles, dtype=np.float32),
            "vz": rng.normal(0, 5, n_particles).astype(np.float32),
            "px": px.astype(np.float32),
            "py": py.astype(np.float32),
            "pz": pz.astype(np.float32),
            "q": q,
            "nhits": np.zeros(n_particles, dtype=np.int32),  # filled in below
        }
    )
    return particles


def _generate_hits_and_truth(
    particles: pd.DataFrame,
    n_layers: int,
    layer_radii: np.ndarray,
    n_noise_hits: int,
    rng: np.random.Generator,
):
    """Crude helical stepping through concentric cylindrical layers.

    This is NOT a physically accurate magnetic-field simulation. It is a
    simplified geometric stand-in sufficient to produce hits that (a) lie
    on a curve per particle, (b) span multiple layers, and (c) exercise the
    full loader/physics/test pipeline. Do not use it to draw physics
    conclusions.
    """
    hit_rows = []
    truth_rows = []
    hit_id_counter = 1

    B_STRENGTH = 0.002  # arbitrary curvature scale, not a real field value

    for _, particle in particles.iterrows():
        pt = np.hypot(particle["px"], particle["py"])
        phi0 = np.arctan2(particle["py"], particle["px"])
        curvature = B_STRENGTH * particle["q"] / max(pt, 1e-3)

        n_hits_this_particle = 0
        for layer_idx in range(n_layers):
            r = layer_radii[layer_idx]
            # crude helical deflection growing with radius, so low-pT
            # (small pt -> large curvature) particles bend more, echoing
            # the qualitative real-physics relationship without claiming
            # to be an accurate simulation of it.
            phi = phi0 + curvature * r
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = particle["vz"] + (particle["pz"] / max(pt, 1e-3)) * r

            hit_rows.append(
                {
                    "hit_id": hit_id_counter,
                    "x": np.float32(x),
                    "y": np.float32(y),
                    "z": np.float32(z),
                    "volume_id": np.int32(1),  # single fake "volume" for simplicity
                    "layer_id": np.int32(layer_idx + 1),
                    "module_id": np.int32(1),
                }
            )
            truth_rows.append(
                {
                    "hit_id": hit_id_counter,
                    "particle_id": np.int64(particle["particle_id"]),
                    "tx": np.float32(x),
                    "ty": np.float32(y),
                    "tz": np.float32(z),
                    "tpx": np.float32(particle["px"]),
                    "tpy": np.float32(particle["py"]),
                    "tpz": np.float32(particle["pz"]),
                    "weight": np.float32(0.0),  # filled in below, uniform placeholder
                }
            )
            hit_id_counter += 1
            n_hits_this_particle += 1

        particles.loc[particles["particle_id"] == particle["particle_id"], "nhits"] = (
            n_hits_this_particle
        )

    # Pure noise hits: random positions, particle_id = 0, per the documented
    # convention (Step 2, Part C): "A value of 0 means that the hit did not
    # originate from a reconstructible particle."
    for _ in range(n_noise_hits):
        r = rng.uniform(layer_radii.min(), layer_radii.max())
        phi = rng.uniform(-np.pi, np.pi)
        z = rng.normal(0, 50)
        x, y = r * np.cos(phi), r * np.sin(phi)
        hit_rows.append(
            {
                "hit_id": hit_id_counter,
                "x": np.float32(x),
                "y": np.float32(y),
                "z": np.float32(z),
                "volume_id": np.int32(1),
                "layer_id": np.int32(rng.integers(1, n_layers + 1)),
                "module_id": np.int32(1),
            }
        )
        truth_rows.append(
            {
                "hit_id": hit_id_counter,
                "particle_id": np.int64(0),
                "tx": np.float32(x),
                "ty": np.float32(y),
                "tz": np.float32(z),
                "tpx": np.float32(0.0),
                "tpy": np.float32(0.0),
                "tpz": np.float32(0.0),
                "weight": np.float32(0.0),
            }
        )
        hit_id_counter += 1

    hits = pd.DataFrame(hit_rows)
    truth = pd.DataFrame(truth_rows)

    # Uniform placeholder weights that sum to 1.0 over the event, matching
    # the documented invariant (Step 2, Part C: "total sum of weights within
    # one event equals to one"). This is NOT the real order/pT-dependent
    # TrackML weighting scheme (Step 2, Part H / weights.py) — implementing
    # that correctly is explicitly deferred to a later milestone.
    n_total = len(truth)
    truth["weight"] = np.float32(1.0 / n_total) if n_total > 0 else np.float32(0.0)

    return hits, truth


def generate_synthetic_event(
    event_id: int = 0,
    n_particles: int = 40,
    n_layers: int = 6,
    noise_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate one synthetic event as a dict of DataFrames.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys: 'hits', 'particles', 'truth' — same shape as
        src.data_loader.load_event's return value, so downstream code
        cannot tell the difference structurally (only the VALUES differ
        from real physics).
    """
    rng = np.random.default_rng(seed)
    layer_radii = np.linspace(30, 300, n_layers)  # mm, arbitrary spacing

    particles = _generate_particles(n_particles, rng)
    n_noise = int(round(noise_fraction * n_particles * 4))  # rough scale echo, not a formula from the literature

    hits, truth = _generate_hits_and_truth(particles, n_layers, layer_radii, n_noise, rng)

    return {"hits": hits, "particles": particles, "truth": truth}


def write_synthetic_event(out_dir: str, event: dict[str, pd.DataFrame], event_id: int = 0) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    prefix = out_path / f"event{event_id:09d}"
    for part_name, df in event.items():
        df.to_csv(f"{prefix}-{part_name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/interim/synthetic")
    parser.add_argument("--event-id", type=int, default=0)
    parser.add_argument("--n-particles", type=int, default=40)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--noise-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    event = generate_synthetic_event(
        event_id=args.event_id,
        n_particles=args.n_particles,
        n_layers=args.n_layers,
        noise_fraction=args.noise_fraction,
        seed=args.seed,
    )
    write_synthetic_event(args.out_dir, event, event_id=args.event_id)
    print(f"[{SYNTHETIC_TAG} DATA] Wrote synthetic event {args.event_id} to {args.out_dir}")
    print(f"[{SYNTHETIC_TAG} DATA]   n_hits={len(event['hits'])}, "
          f"n_particles={len(event['particles'])}, "
          f"n_truth_rows={len(event['truth'])}")


if __name__ == "__main__":
    main()
