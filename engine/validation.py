"""
Geometric validation for origami crease patterns.

Implements Kawasaki's theorem, Maekawa's theorem, and triangle-triangle
self-intersection detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paper import Paper


# ────────────────────────────────────────────────────────────────────
# Result container
# ────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    kawasaki_valid: bool
    kawasaki_violation: float
    maekawa_valid: bool
    maekawa_violation: float
    intersection_free: bool
    self_intersection_count: int
    is_valid: bool  # all checks pass


# ────────────────────────────────────────────────────────────────────
# Kawasaki's theorem
# ────────────────────────────────────────────────────────────────────

def check_kawasaki(paper: Paper) -> tuple[bool, float]:
    """At each interior vertex, the alternating sum of sector angles equals pi.

    Specifically, for a vertex with 2n incident creases, the sum of
    odd-indexed sector angles equals the sum of even-indexed sector
    angles equals pi.

    Returns (is_valid, total_violation).  A violation of < 1e-6 is
    considered valid.
    """
    verts = paper.vertices
    edges = paper.edges
    n_verts = len(verts)

    # Build adjacency: vertex -> list of neighbor vertices (via edges)
    adj: dict[int, list[int]] = {}
    for e in edges:
        adj.setdefault(int(e[0]), []).append(int(e[1]))
        adj.setdefault(int(e[1]), []).append(int(e[0]))

    # Identify boundary vertices (incident to a 'B' edge)
    boundary_verts: set[int] = set()
    for ei, e in enumerate(edges):
        if paper.assignments[ei] == "B":
            boundary_verts.add(int(e[0]))
            boundary_verts.add(int(e[1]))

    total_violation = 0.0

    for vi in range(n_verts):
        if vi in boundary_verts:
            continue
        neighbors = adj.get(vi, [])
        if len(neighbors) < 2:
            continue

        # Sort neighbors by angle around vi (in the XY plane for flat-foldability)
        center = verts[vi][:2]
        angles = []
        for ni in neighbors:
            d = verts[ni][:2] - center
            angles.append((np.arctan2(d[1], d[0]), ni))
        angles.sort(key=lambda x: x[0])

        # Sector angles
        sector_angles = []
        for k in range(len(angles)):
            a1 = angles[k][0]
            a2 = angles[(k + 1) % len(angles)][0]
            diff = a2 - a1
            if diff <= 0:
                diff += 2.0 * np.pi
            sector_angles.append(diff)

        if len(sector_angles) < 2:
            continue

        # Kawasaki: alternating sums should both equal pi
        even_sum = sum(sector_angles[i] for i in range(0, len(sector_angles), 2))
        odd_sum = sum(sector_angles[i] for i in range(1, len(sector_angles), 2))

        violation = abs(even_sum - odd_sum)
        total_violation += violation

    is_valid = bool(total_violation < 1e-4)
    return is_valid, float(total_violation)


# ────────────────────────────────────────────────────────────────────
# Maekawa's theorem
# ────────────────────────────────────────────────────────────────────

def check_maekawa(paper: Paper) -> tuple[bool, float]:
    """At each interior vertex, |M - V| = 2.

    Returns (is_valid, total_violation) where violation is
    sum of |abs(M-V) - 2| over all interior vertices.
    """
    edges = paper.edges
    verts = paper.vertices
    n_verts = len(verts)

    # Boundary vertices
    boundary_verts: set[int] = set()
    for ei, e in enumerate(edges):
        if paper.assignments[ei] == "B":
            boundary_verts.add(int(e[0]))
            boundary_verts.add(int(e[1]))

    # Count M and V edges per vertex
    m_count = [0] * n_verts
    v_count = [0] * n_verts
    total_mv_per_vertex = [0] * n_verts

    for ei, e in enumerate(edges):
        a = paper.assignments[ei]
        if a == "M":
            m_count[int(e[0])] += 1
            m_count[int(e[1])] += 1
        elif a == "V":
            v_count[int(e[0])] += 1
            v_count[int(e[1])] += 1
        if a in ("M", "V"):
            total_mv_per_vertex[int(e[0])] += 1
            total_mv_per_vertex[int(e[1])] += 1

    total_violation = 0.0
    for vi in range(n_verts):
        if vi in boundary_verts:
            continue
        # Only check vertices that actually have creases
        if total_mv_per_vertex[vi] == 0:
            continue
        diff = abs(m_count[vi] - v_count[vi])
        violation = abs(diff - 2)
        total_violation += violation

    is_valid = total_violation < 0.5  # integer theorem, so < 0.5 means exact
    return is_valid, float(total_violation)


# ────────────────────────────────────────────────────────────────────
# Self-intersection detection (triangle-triangle)
# ────────────────────────────────────────────────────────────────────

def check_self_intersection(paper: Paper) -> tuple[bool, int]:
    """Check for triangle-triangle intersections among the paper's faces.

    Uses the separating-axis theorem (SAT) for triangle-triangle overlap
    in 3-D.  Faces that share an edge or vertex are skipped.

    Returns (is_valid, count_of_intersections).
    """
    verts = paper.vertices
    faces = paper.faces
    count = 0

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            # Skip faces that share vertices (adjacent faces)
            if set(faces[i]) & set(faces[j]):
                continue
            if _triangles_intersect(verts, faces[i], faces[j]):
                count += 1

    return count == 0, count


def _triangles_intersect(
    verts: np.ndarray,
    face1: list[int],
    face2: list[int],
) -> bool:
    """Test whether two triangular faces intersect in 3-D using
    the separating-axis theorem (Moller's method simplified).

    For non-triangular faces, only tests the first three vertices.
    Returns True if the triangles intersect.
    """
    if len(face1) < 3 or len(face2) < 3:
        return False

    t1 = verts[face1[:3]]
    t2 = verts[face2[:3]]

    # 13 potential separating axes:
    # - normals of each triangle (2)
    # - cross products of edge pairs (3x3 = 9)
    # - edges themselves don't need separate tests in 3D SAT

    e1_edges = [t1[1] - t1[0], t1[2] - t1[1], t1[0] - t1[2]]
    e2_edges = [t2[1] - t2[0], t2[2] - t2[1], t2[0] - t2[2]]

    n1 = np.cross(e1_edges[0], e1_edges[1])
    n2 = np.cross(e2_edges[0], e2_edges[1])

    axes = [n1, n2]
    for e1 in e1_edges:
        for e2 in e2_edges:
            ax = np.cross(e1, e2)
            if np.linalg.norm(ax) > 1e-12:
                axes.append(ax)

    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        axis = axis / norm

        proj1 = np.dot(t1, axis)
        proj2 = np.dot(t2, axis)

        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()

        # Check for separation (with small tolerance for shared-edge adjacency)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False  # separating axis found

    return True  # no separating axis → intersection


# ────────────────────────────────────────────────────────────────────
# Combined validation
# ────────────────────────────────────────────────────────────────────

def validate_paper(paper: Paper) -> ValidationResult:
    """Run all validation checks and return a combined result."""
    k_valid, k_violation = check_kawasaki(paper)
    m_valid, m_violation = check_maekawa(paper)
    si_valid, si_count = check_self_intersection(paper)

    return ValidationResult(
        kawasaki_valid=k_valid,
        kawasaki_violation=k_violation,
        maekawa_valid=m_valid,
        maekawa_violation=m_violation,
        intersection_free=si_valid,
        self_intersection_count=si_count,
        is_valid=k_valid and m_valid and si_valid,
    )


def validate_state(paper: Paper) -> dict:
    """Run all validation checks and return a flat dict.

    This is the interface used by OrigamiEnvironment. It calls the
    existing validation functions and returns a dict with all fields
    the environment and metrics system need.
    """
    result = validate_paper(paper)
    strain_exceeded = bool(
        len(paper.strain_per_vertex) > 0
        and float(paper.strain_per_vertex.max()) > paper.material.max_strain
    )
    return {
        "is_valid": result.is_valid and not strain_exceeded,
        "kawasaki_violations": int(not result.kawasaki_valid),
        "kawasaki_total_error": float(result.kawasaki_violation),
        "maekawa_violations": int(not result.maekawa_valid),
        "self_intersections": result.self_intersection_count,
        "strain_exceeded": strain_exceeded,
    }
