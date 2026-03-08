"""
Bar-and-hinge physics model.

Three energy components:
  E_total = E_bar + E_facet + E_fold

Stiffness parameters are derived from the material properties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paper import Paper


# ────────────────────────────────────────────────────────────────────
# Stiffness
# ────────────────────────────────────────────────────────────────────

@dataclass
class StiffnessParams:
    """Stiffness values derived from material properties."""
    k_axial: np.ndarray   # per-edge axial stiffness  (E,)
    k_facet: float         # facet (panel bending) stiffness
    k_fold: float          # fold (crease torsion) stiffness


def compute_stiffness(paper: Paper) -> StiffnessParams:
    """Derive stiffness parameters from the paper's material and geometry.

    k_axial = E * t * w / L0   (per edge, w ≈ average of adjacent edge lengths)
    k_facet = E * t^3 / (12 * (1 - nu^2))
    k_fold  = 0.1 * k_facet    (crease torsional stiffness, empirical fraction)
    """
    mat = paper.material
    E = mat.youngs_modulus_pa  # Pa
    t = mat.thickness_m        # m
    nu = mat.poissons_ratio

    rest = paper.rest_lengths
    # Guard against zero rest lengths
    safe_rest = np.where(rest > 1e-15, rest, 1e-15)

    # Approximate edge width as the average rest length (simple heuristic)
    w = np.mean(safe_rest) if len(safe_rest) > 0 else 1e-3

    k_axial = E * t * w / safe_rest  # (E,)

    k_facet = E * t ** 3 / (12.0 * (1.0 - nu ** 2))

    # Crease torsional stiffness — a fraction of facet stiffness
    k_fold = 0.1 * k_facet

    return StiffnessParams(k_axial=k_axial, k_facet=k_facet, k_fold=k_fold)


# ────────────────────────────────────────────────────────────────────
# Energy components
# ────────────────────────────────────────────────────────────────────

def compute_bar_energy(paper: Paper) -> float:
    """E_bar = sum (1/2) * k_axial * (L - L0)^2

    Measures stretching / compression of edges relative to rest lengths.
    """
    if len(paper.edges) == 0:
        return 0.0

    verts = paper.vertices
    edges = paper.edges
    current_lengths = np.array([
        np.linalg.norm(verts[e[1]] - verts[e[0]]) for e in edges
    ])

    stiff = compute_stiffness(paper)
    delta = current_lengths - paper.rest_lengths
    energy = 0.5 * np.sum(stiff.k_axial * delta ** 2)
    return float(energy)


def compute_facet_energy(paper: Paper) -> float:
    """E_facet = sum (1/2) * k_facet * l * (theta - pi)^2

    Measures bending of facet panels away from flat (pi).
    *l* is the edge length (hinge length) and *theta* is the dihedral angle
    across the edge between two adjacent faces.  For edges that are not
    shared by two faces we skip them.
    """
    if len(paper.edges) == 0 or len(paper.faces) < 2:
        return 0.0

    stiff = compute_stiffness(paper)
    verts = paper.vertices
    edges = paper.edges

    # Build edge → face adjacency
    edge_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(paper.faces):
        n = len(face)
        for k in range(n):
            va, vb = face[k], face[(k + 1) % n]
            for ei, e in enumerate(edges):
                if (e[0] == va and e[1] == vb) or (e[0] == vb and e[1] == va):
                    edge_faces.setdefault(ei, []).append(fi)
                    break

    energy = 0.0
    for ei, adj_faces in edge_faces.items():
        if len(adj_faces) < 2:
            continue
        # Only consider non-fold edges (flat or boundary interior)
        if paper.assignments[ei] in ("M", "V"):
            continue

        f1, f2 = adj_faces[0], adj_faces[1]
        theta = _dihedral_angle(verts, paper.faces[f1], paper.faces[f2], edges[ei])
        l = np.linalg.norm(verts[edges[ei][1]] - verts[edges[ei][0]])
        energy += 0.5 * stiff.k_facet * l * (theta - np.pi) ** 2

    return float(energy)


def compute_fold_energy(paper: Paper) -> float:
    """E_fold = sum (1/2) * k_fold * l * (rho - rho_target)^2

    Measures deviation of fold creases from their target angles.
    *rho* is the current dihedral angle across the fold edge and
    *rho_target* comes from ``fold_angles``.
    """
    if len(paper.edges) == 0:
        return 0.0

    stiff = compute_stiffness(paper)
    verts = paper.vertices
    edges = paper.edges

    # Build edge → face adjacency
    edge_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(paper.faces):
        n = len(face)
        for k in range(n):
            va, vb = face[k], face[(k + 1) % n]
            for ei, e in enumerate(edges):
                if (e[0] == va and e[1] == vb) or (e[0] == vb and e[1] == va):
                    edge_faces.setdefault(ei, []).append(fi)
                    break

    energy = 0.0
    for ei in range(len(edges)):
        if paper.assignments[ei] not in ("M", "V"):
            continue
        if ei not in edge_faces or len(edge_faces[ei]) < 2:
            continue

        f1, f2 = edge_faces[ei][0], edge_faces[ei][1]
        rho = _dihedral_angle(verts, paper.faces[f1], paper.faces[f2], edges[ei])
        rho_target = np.radians(paper.fold_angles[ei])  # fold_angles stored in degrees
        l = np.linalg.norm(verts[edges[ei][1]] - verts[edges[ei][0]])
        energy += 0.5 * stiff.k_fold * l * (rho - rho_target) ** 2

    return float(energy)


def compute_total_energy(paper: Paper) -> float:
    """E_total = E_bar + E_facet + E_fold."""
    return compute_bar_energy(paper) + compute_facet_energy(paper) + compute_fold_energy(paper)


# ────────────────────────────────────────────────────────────────────
# Strain
# ────────────────────────────────────────────────────────────────────

def compute_strain(paper: Paper) -> np.ndarray:
    """Per-vertex Cauchy strain: average fractional edge-length deviation.

    Returns shape (N,) array of non-negative strain values.
    """
    n_verts = len(paper.vertices)
    if n_verts == 0:
        return np.empty(0)

    verts = paper.vertices
    edges = paper.edges
    rest = paper.rest_lengths

    # Build vertex → edge adjacency
    vert_edges: dict[int, list[int]] = {}
    for ei, e in enumerate(edges):
        vert_edges.setdefault(int(e[0]), []).append(ei)
        vert_edges.setdefault(int(e[1]), []).append(ei)

    strain = np.zeros(n_verts, dtype=np.float64)
    for vi in range(n_verts):
        adj = vert_edges.get(vi, [])
        if not adj:
            continue
        devs = []
        for ei in adj:
            v1, v2 = edges[ei]
            L = np.linalg.norm(verts[v1] - verts[v2])
            L0 = rest[ei]
            if L0 > 1e-15:
                devs.append(abs(L - L0) / L0)
        if devs:
            strain[vi] = float(np.mean(devs))

    return strain


# ────────────────────────────────────────────────────────────────────
# Dihedral angle helper
# ────────────────────────────────────────────────────────────────────

def _dihedral_angle(
    verts: np.ndarray,
    face1: list[int],
    face2: list[int],
    edge: np.ndarray,
) -> float:
    """Compute the dihedral angle (in radians) between two faces sharing *edge*.

    Returns angle in [0, 2*pi).  Returns pi if normals cannot be computed.
    """
    n1 = _face_normal(verts, face1)
    n2 = _face_normal(verts, face2)

    if n1 is None or n2 is None:
        return np.pi

    cos_a = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_a)

    # Determine sign from edge direction
    edge_dir = verts[edge[1]] - verts[edge[0]]
    edge_dir = edge_dir / (np.linalg.norm(edge_dir) + 1e-30)
    cross = np.cross(n1, n2)
    if np.dot(cross, edge_dir) < 0:
        angle = 2.0 * np.pi - angle

    return float(angle)


def _face_normal(verts: np.ndarray, face: list[int]) -> np.ndarray | None:
    """Compute outward unit normal of a face, or None if degenerate."""
    if len(face) < 3:
        return None
    v0 = verts[face[0]]
    v1 = verts[face[1]]
    v2 = verts[face[2]]
    normal = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normal)
    if norm < 1e-15:
        return None
    return normal / norm
