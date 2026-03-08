"""
Quality metrics for folded origami.

Computes bounding box, deployment ratio, fold count, and aggregated
metric dictionaries for the trainer reward functions.
"""

from __future__ import annotations

import numpy as np

from .paper import Paper


def compute_bounding_box(paper: Paper) -> np.ndarray:
    """Axis-aligned bounding-box dimensions (dx, dy, dz).

    Returns shape (3,) array.  Minimum z-thickness accounts for
    material thickness times estimated layer count.
    """
    if len(paper.vertices) == 0:
        return np.zeros(3)

    ptp = np.ptp(paper.vertices, axis=0)
    ptp = np.where(np.abs(ptp) < 1e-12, 0.0, ptp)

    # Minimum z from material thickness * layers
    t = paper.material.thickness_mm / 1000.0
    ptp[2] = max(ptp[2], t * paper.num_layers)

    return ptp


def compute_deployment_ratio(paper: Paper) -> float:
    """Ratio of folded XY footprint area to original sheet area.

    A fully flat unfolded sheet has ratio 1.0; a tightly folded sheet
    approaches 0.0.
    """
    if paper.original_area <= 0:
        return 1.0

    bb = compute_bounding_box(paper)
    folded_area = bb[0] * bb[1]

    ratio = folded_area / paper.original_area
    return float(np.clip(ratio, 0.0, 1.0))


def compute_fold_count(paper: Paper) -> int:
    """Number of mountain (M) and valley (V) edges."""
    return sum(1 for a in paper.assignments if a in ("M", "V"))


def compute_compactness(paper: Paper) -> float:
    """1 - deployment_ratio.  Higher is more compact."""
    return 1.0 - compute_deployment_ratio(paper)


def compute_volume(paper: Paper) -> float:
    """Bounding-box volume in cubic meters."""
    bb = compute_bounding_box(paper)
    return float(bb[0] * bb[1] * bb[2])


def compute_metrics(paper: Paper, original_paper: Paper | None = None) -> dict:
    """Compute all quality metrics and return as a dict.

    Parameters
    ----------
    paper : Paper
        The current (folded) paper state.
    original_paper : Paper or None
        The original (unfolded) paper, used for strain comparison.
        If None, strain is computed against the current paper's rest lengths.

    Returns
    -------
    dict with keys:
        bounding_box, deployment_ratio, fold_count, compactness,
        volume, max_strain, mean_strain, num_vertices, num_faces,
        num_layers.
    """
    from .physics import compute_strain  # local import to avoid circular

    bb = compute_bounding_box(paper)
    strain = compute_strain(paper)

    return {
        "bounding_box": {
            "x": float(bb[0]),
            "y": float(bb[1]),
            "z": float(bb[2]),
        },
        "deployment_ratio": compute_deployment_ratio(paper),
        "fold_count": compute_fold_count(paper),
        "compactness": compute_compactness(paper),
        "volume": compute_volume(paper),
        "max_strain": float(np.max(strain)) if len(strain) > 0 else 0.0,
        "mean_strain": float(np.mean(strain)) if len(strain) > 0 else 0.0,
        "num_vertices": len(paper.vertices),
        "num_faces": len(paper.faces),
        "num_layers": paper.num_layers,
    }
