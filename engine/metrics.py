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


def compute_all_metrics(paper, task: dict, validation: dict) -> dict:
    """Compute every metric and return a flat dict.

    Called after physics + validation. Combines validity, compactness,
    structural, efficiency, and deployability metrics.

    Parameters
    ----------
    paper : Paper
        Current paper state (after simulate()).
    task : dict
        Task definition with keys: width, height, target_ratio, target_box, must_deploy.
    validation : dict
        Output of validate_state(paper).
    """
    import numpy as np

    bb = paper.bounding_box  # (3,) array
    original_area = paper.original_area if paper.original_area > 0 else (paper.material.thickness_mm / 1000.0)
    t = paper.material.thickness_mm / 1000.0
    original_bbox_vol = original_area * t
    folded_bbox_vol = float(bb[0] * bb[1] * bb[2]) if bb[2] > 0 else float(bb[0] * bb[1] * t)

    # ── Folded area (XY footprint) ────────────────────────────────
    if len(paper.vertices) >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(paper.vertices[:, :2])
            folded_area = float(hull.volume)
        except Exception:
            ptp = np.ptp(paper.vertices[:, :2], axis=0)
            folded_area = float(ptp[0] * ptp[1])
    else:
        folded_area = original_area

    deployment_ratio = folded_area / original_area if original_area > 0 else 1.0
    compactness = 1.0 - deployment_ratio
    volume_compaction = folded_bbox_vol / original_bbox_vol if original_bbox_vol > 0 else 1.0
    material_volume = original_area * t
    packing_efficiency = material_volume / folded_bbox_vol if folded_bbox_vol > 0 else 0.0

    # ── Target box check ─────────────────────────────────────────
    target_box = task.get("target_box")
    fits_target_box = False
    if target_box and len(target_box) == 3:
        fits_target_box = bool(
            bb[0] <= target_box[0] + 1e-6 and
            bb[1] <= target_box[1] + 1e-6 and
            bb[2] <= target_box[2] + 1e-6
        )

    # ── Strain ───────────────────────────────────────────────────
    strain = paper.strain_per_vertex
    max_strain = float(np.max(strain)) if len(strain) > 0 else 0.0
    mean_strain = float(np.mean(strain)) if len(strain) > 0 else 0.0

    # ── Energy ───────────────────────────────────────────────────
    energy = paper.energy

    # ── Efficiency ───────────────────────────────────────────────
    fold_count = paper.fold_count

    # Crease complexity: entropy of M/V assignment distribution
    mv_assignments = [a for a in paper.assignments if a in ("M", "V")]
    if mv_assignments:
        total = len(mv_assignments)
        m_count = mv_assignments.count("M")
        v_count = mv_assignments.count("V")
        p_m = m_count / total if total > 0 else 0
        p_v = v_count / total if total > 0 else 0
        crease_complexity = 0.0
        if p_m > 0:
            crease_complexity -= p_m * np.log2(p_m)
        if p_v > 0:
            crease_complexity -= p_v * np.log2(p_v)
    else:
        crease_complexity = 0.0

    folding_efficiency = compactness / max(fold_count, 1)

    # ── Deployability ─────────────────────────────────────────────
    must_deploy = task.get("must_deploy", False)
    # Simple deployability heuristic: if valid and compactness > 0, assume deployable
    is_deployable = bool(validation.get("is_valid", False) and compactness > 0.01) if must_deploy else None
    # Deployment force estimate from total energy gradient (rough)
    deployment_force_estimate = float(energy.get("fold", 0.0)) / max(paper.original_area, 1e-6)

    return {
        # Validity (from validation dict)
        "is_valid": validation.get("is_valid", False),
        "kawasaki_violations": validation.get("kawasaki_violations", 0),
        "kawasaki_total_error": validation.get("kawasaki_total_error", 0.0),
        "maekawa_violations": validation.get("maekawa_violations", 0),
        "self_intersections": validation.get("self_intersections", 0),
        "strain_exceeded": validation.get("strain_exceeded", False),

        # Compactness
        "deployment_ratio": float(deployment_ratio),
        "compactness": float(compactness),
        "volume_compaction": float(volume_compaction),
        "packing_efficiency": float(packing_efficiency),
        "fits_target_box": fits_target_box,
        "bounding_box": bb.tolist(),

        # Structural
        "max_strain": max_strain,
        "mean_strain": mean_strain,
        "total_energy": float(energy.get("total", 0.0)),
        "energy_bar": float(energy.get("bar", 0.0)),
        "energy_facet": float(energy.get("facet", 0.0)),
        "energy_fold": float(energy.get("fold", 0.0)),

        # Efficiency
        "fold_count": fold_count,
        "folding_efficiency": float(folding_efficiency),
        "crease_complexity": float(crease_complexity),

        # Deployability
        "is_deployable": is_deployable,
        "deployment_force_estimate": float(deployment_force_estimate),

        # Shape similarity placeholders
        "chamfer_distance": None,
        "hausdorff_distance": None,
    }
