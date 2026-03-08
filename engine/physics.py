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


# ────────────────────────────────────────────────────────────────────
# Topology precomputation
# ────────────────────────────────────────────────────────────────────

def build_beam_list(paper: Paper) -> list[tuple[int, int, float, float]]:
    """Build list of (node_a, node_b, rest_len, k_axial) for every edge.

    Uses normalized stiffness values (arch doc constants) scaled by material
    Young's modulus ratio — keeps the Verlet integrator stable at unit scale.
    """
    # Normalized stiffness constants (arch doc values)
    K_AXIAL_BASE = 70.0
    # Scale by material: paper (3 GPa) = 1.0 baseline
    mat = paper.material
    E_ratio = mat.youngs_modulus_gpa / 3.0
    k_axial = K_AXIAL_BASE * E_ratio

    beams = []
    for ei, (v1, v2) in enumerate(paper.edges):
        L0 = paper.rest_lengths[ei]
        beams.append((int(v1), int(v2), float(L0), float(k_axial)))
    return beams


def build_crease_list(paper: Paper) -> list[tuple[int, int, int, int, float, float, str]]:
    """Build list of (n1, n2, n3, n4, target_angle_rad, k, type) for each crease hinge.

    Each hinge is defined by 4 nodes: n1-n2 is the hinge edge, n3 and n4 are
    the wing-tip nodes of the two adjacent faces.
    type is 'fold' (M/V crease) or 'facet' (interior flat edge).
    """
    verts = paper.vertices

    # Build edge → face adjacency
    edge_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(paper.faces):
        n = len(face)
        for k in range(n):
            va, vb = face[k], face[(k + 1) % n]
            for ei, e in enumerate(paper.edges):
                if (e[0] == va and e[1] == vb) or (e[0] == vb and e[1] == va):
                    edge_faces.setdefault(ei, []).append(fi)
                    break

    creases = []
    for ei, adj in edge_faces.items():
        if len(adj) < 2:
            continue
        f1, f2 = adj[0], adj[1]
        face1, face2 = paper.faces[f1], paper.faces[f2]
        n1, n2 = int(paper.edges[ei][0]), int(paper.edges[ei][1])

        # Find wing-tip nodes (in each face, the vertex NOT on the shared edge)
        wing1 = [v for v in face1 if v != n1 and v != n2]
        wing2 = [v for v in face2 if v != n1 and v != n2]
        if not wing1 or not wing2:
            continue
        n3, n4 = int(wing1[0]), int(wing2[0])

        # Normalized stiffness constants (arch doc values), scaled by material
        E_ratio = paper.material.youngs_modulus_gpa / 3.0
        K_FACET = 0.2 * E_ratio
        K_FOLD = 0.7 * E_ratio

        asgn = paper.assignments[ei]
        if asgn in ("M", "V"):
            target = float(np.radians(paper.fold_angles[ei]))
            k = K_FOLD
            ctype = "fold"
        else:
            target = float(np.pi)
            k = K_FACET
            ctype = "facet"

        creases.append((n1, n2, n3, n4, target, k, ctype))
    return creases


def _torque_to_forces(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
    torque: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a dihedral torque into forces on the 4 hinge nodes.

    p1-p2 is the hinge edge. p3 and p4 are wing tips.
    Returns (f1, f2, f3, f4) as (3,) arrays.
    """
    e = p2 - p1
    e_len = np.linalg.norm(e)
    if e_len < 1e-12:
        zero = np.zeros(3)
        return zero, zero, zero, zero

    e_hat = e / e_len

    # Perpendicular components of wing vectors relative to hinge
    d3 = p3 - p1
    d4 = p4 - p1
    d3_perp = d3 - np.dot(d3, e_hat) * e_hat
    d4_perp = d4 - np.dot(d4, e_hat) * e_hat

    len3 = np.linalg.norm(d3_perp)
    len4 = np.linalg.norm(d4_perp)

    if len3 < 1e-12 or len4 < 1e-12:
        zero = np.zeros(3)
        return zero, zero, zero, zero

    # Force on wing tips proportional to torque / lever arm
    f3 = torque / (len3 * e_len) * np.cross(e_hat, d3_perp / len3)
    f4 = -torque / (len4 * e_len) * np.cross(e_hat, d4_perp / len4)

    # Reaction forces distributed to hinge nodes
    f1 = -(f3 + f4) * 0.5
    f2 = -(f3 + f4) * 0.5

    return f1, f2, f3, f4


# ────────────────────────────────────────────────────────────────────
# Verlet solver
# ────────────────────────────────────────────────────────────────────

def simulate(
    paper: Paper,
    fold_percent: float = 1.0,
    n_steps: int = 500,
    dt: float = 0.005,
    damping: float = 0.15,
) -> Paper:
    """Run bar-and-hinge Verlet integration to relax the mesh.

    Updates paper.vertices, paper.strain_per_vertex, and paper.energy in-place.
    Returns the mutated paper for chaining.

    Parameters
    ----------
    paper : Paper
        Paper state after a fold has been applied (vertices already rotated).
    fold_percent : float
        How far along the fold to drive (0=flat, 1=full target angle).
    n_steps : int
        Maximum integration steps.
    dt : float
        Time step. Keep small (0.005) for stability with stiff materials.
    damping : float
        Velocity damping coefficient (0=undamped, 1=fully damped).
    """
    if len(paper.vertices) == 0:
        return paper

    beams = build_beam_list(paper)
    creases = build_crease_list(paper)

    pos = paper.vertices.copy()        # (N, 3) current positions
    last_pos = pos.copy()              # (N, 3) previous positions (Verlet)

    max_force_cap = 1e6  # prevent runaway forces

    for _ in range(n_steps):
        forces = np.zeros_like(pos)

        # ── Beam (axial spring) forces ───────────────────────────────
        for (a, b, L0, k) in beams:
            delta = pos[b] - pos[a]
            L = np.linalg.norm(delta)
            if L < 1e-12:
                continue
            strain = (L - L0) / L0
            F_mag = k * strain
            F_vec = F_mag * (delta / L)
            # Clamp to prevent instability
            F_vec = np.clip(F_vec, -max_force_cap, max_force_cap)
            forces[a] += F_vec
            forces[b] -= F_vec

        # ── Crease (dihedral spring) forces ─────────────────────────
        for (n1, n2, n3, n4, target, k, ctype) in creases:
            actual_target = target * fold_percent if ctype == "fold" else target
            try:
                theta = _compute_dihedral_rad(pos[n1], pos[n2], pos[n3], pos[n4])
            except Exception:
                continue
            delta_theta = theta - actual_target
            edge_len = np.linalg.norm(pos[n2] - pos[n1])
            torque = k * edge_len * delta_theta
            torque = float(np.clip(torque, -max_force_cap, max_force_cap))

            f1, f2, f3, f4 = _torque_to_forces(
                pos[n1], pos[n2], pos[n3], pos[n4], torque
            )
            forces[n1] += np.clip(f1, -max_force_cap, max_force_cap)
            forces[n2] += np.clip(f2, -max_force_cap, max_force_cap)
            forces[n3] += np.clip(f3, -max_force_cap, max_force_cap)
            forces[n4] += np.clip(f4, -max_force_cap, max_force_cap)

        # ── Verlet integration ───────────────────────────────────────
        new_pos = pos + (1.0 - damping) * (pos - last_pos) + forces * (dt * dt)

        # NaN guard
        if np.any(np.isnan(new_pos)):
            break

        last_pos = pos
        pos = new_pos

        # ── Convergence check ────────────────────────────────────────
        kinetic = np.sum((pos - last_pos) ** 2)
        if kinetic < 1e-12:
            break

    # ── Write results back to paper ──────────────────────────────────
    paper.vertices = pos
    paper.strain_per_vertex = compute_strain(paper)
    paper.energy = {
        "total": compute_total_energy(paper),
        "bar": compute_bar_energy(paper),
        "facet": compute_facet_energy(paper),
        "fold": compute_fold_energy(paper),
    }

    return paper


def _compute_dihedral_rad(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> float:
    """Dihedral angle in radians between planes (p1,p2,p3) and (p1,p2,p4).

    p1-p2 is the hinge edge. p3 and p4 are the wing tips.
    Returns angle in [0, 2*pi).
    """
    e = p2 - p1
    e_norm = np.linalg.norm(e)
    if e_norm < 1e-12:
        return float(np.pi)
    e_hat = e / e_norm

    n1 = np.cross(p3 - p1, e)
    n2 = np.cross(e, p4 - p1)
    len1 = np.linalg.norm(n1)
    len2 = np.linalg.norm(n2)
    if len1 < 1e-12 or len2 < 1e-12:
        return float(np.pi)

    n1 = n1 / len1
    n2 = n2 / len2

    cos_a = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle = np.arccos(cos_a)

    cross = np.cross(n1, n2)
    if np.dot(cross, e_hat) < 0:
        angle = 2.0 * np.pi - angle

    return float(angle)
