"""
Origami mass-spring dynamic relaxation simulator.

Based on: Ghassaei et al., "Fast, Interactive Origami Simulation using GPU
Computation", 7OSME 2018.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

# ── Physics constants ────────────────────────────────────────────────────────

AXIAL_STIFFNESS  = 20.0   # K = AXIAL_STIFFNESS / rest_length
CREASE_STIFFNESS = 0.7    # K = CREASE_STIFFNESS * edge_length  (M/V creases)
PANEL_STIFFNESS  = 0.7    # K = PANEL_STIFFNESS  * edge_length  (F / panel edges)
PERCENT_DAMPING  = 0.45   # global viscous damping fraction
DT               = 0.002  # timestep (seconds)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _triangulate_faces(faces_vertices: list[list[int]]) -> np.ndarray:
    """Fan-triangulate polygonal faces (triangles and quads supported)."""
    tris = []
    for face in faces_vertices:
        if len(face) == 3:
            tris.append(face)
        elif len(face) == 4:
            a, b, c, d = face
            tris.append([a, b, c])
            tris.append([a, c, d])
        else:
            # General fan triangulation for n-gons
            for k in range(1, len(face) - 1):
                tris.append([face[0], face[k], face[k + 1]])
    return np.array(tris, dtype=np.int32)


def _point_on_segment(p: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                      tol: float = 1e-6) -> bool:
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-10:
        return False
    seg_dir = seg / seg_len
    t = np.dot(p - p0, seg_dir)
    perp = (p - p0) - t * seg_dir
    return -tol <= t <= seg_len + tol and np.linalg.norm(perp) < tol


# ── Mesh subdivision ──────────────────────────────────────────────────────────

def _subdivide(pos2d: np.ndarray, triangles: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Split each triangle into 4 by inserting edge midpoints."""
    midpoint_cache: dict[tuple[int, int], int] = {}
    new_pos = list(pos2d)
    new_tris = []

    def get_mid(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in midpoint_cache:
            mid = (np.array(new_pos[i]) + np.array(new_pos[j])) / 2.0
            midpoint_cache[key] = len(new_pos)
            new_pos.append(mid)
        return midpoint_cache[key]

    for tri in triangles:
        a, b, c = tri
        ab = get_mid(a, b)
        bc = get_mid(b, c)
        ca = get_mid(c, a)
        new_tris.extend([
            [a,  ab, ca],
            [ab, b,  bc],
            [ca, bc, c ],
            [ab, bc, ca],
        ])

    return np.array(new_pos, dtype=np.float64), np.array(new_tris, dtype=np.int32)


# ── Main simulator ────────────────────────────────────────────────────────────

class OrigamiSimulator:
    """
    Mass-spring dynamic relaxation simulator for origami.

    Parameters
    ----------
    fold_data : dict
        Parsed FOLD JSON with keys: vertices_coords, edges_vertices,
        edges_assignment.
    subdivisions : int
        Number of midpoint subdivision passes (default 2 → 4× mesh density).
    """

    def __init__(self, fold_data: dict, subdivisions: int = 2) -> None:
        self._fold_percent = 0.0
        self._build(fold_data, subdivisions)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_fold_percent(self, percent: float) -> None:
        """Update all crease spring target angles (0.0 = flat, 1.0 = fully folded)."""
        self._fold_percent = float(percent)
        self._crease_target = self._fold_percent * self._crease_full_theta

    def step(self, n_steps: int = 50) -> None:
        """Advance the simulation by n_steps Euler integration steps."""
        for _ in range(n_steps):
            self._euler_step()

    def reset(self) -> None:
        """Reset to flat state (z=0, vel=0), preserving current fold percent."""
        self.pos = self._flat_pos.copy()
        self.vel[:] = 0.0

    @property
    def crease_indices(self) -> list[tuple[int, int, str]]:
        """Return list of (a, b, assignment) for all crease springs."""
        return list(zip(
            self._crease_a.tolist(),
            self._crease_b.tolist(),
            self._crease_assign,
        ))

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self, fold_data: dict, subdivisions: int) -> None:
        coords = fold_data['vertices_coords']
        orig_edges = fold_data['edges_vertices']
        orig_assign = fold_data['edges_assignment']

        # Original 2-D positions
        pts2d = np.array([[x, y] for x, y in coords], dtype=np.float64)

        # Build triangles from faces_vertices when available (preferred: ensures
        # crease edges appear as actual mesh edges after subdivision).
        # Quads [a,b,c,d] are split into [a,b,c] + [a,c,d].
        # Fall back to Delaunay only if faces_vertices is absent.
        if 'faces_vertices' in fold_data:
            triangles = _triangulate_faces(fold_data['faces_vertices'])
        else:
            tri = Delaunay(pts2d)
            triangles = tri.simplices.astype(np.int32)

        # Build original crease segments for later classification
        # Only M and V assignments are actual fold creases; B is boundary.
        orig_creases: list[tuple[np.ndarray, np.ndarray, str]] = []
        for (u, v), asgn in zip(orig_edges, orig_assign):
            if asgn in ('M', 'V'):
                orig_creases.append((pts2d[u], pts2d[v], asgn))

        # Midpoint subdivision passes
        pos2d = pts2d.copy()
        for _ in range(subdivisions):
            pos2d, triangles = _subdivide(pos2d, triangles)

        n = len(pos2d)

        # 3-D positions (flat, z=0)
        pos3d = np.zeros((n, 3), dtype=np.float64)
        pos3d[:, :2] = pos2d

        self.pos        = pos3d
        self._flat_pos  = pos3d.copy()
        self.vel        = np.zeros((n, 3), dtype=np.float64)
        self.triangles  = triangles

        self._build_beams(triangles)
        self._build_masses(triangles)
        self._build_creases(triangles, pos2d, orig_creases)

    def _build_beams(self, triangles: np.ndarray) -> None:
        """Collect all unique triangle edges as structural (axial) springs."""
        edge_set: set[tuple[int, int]] = set()
        for tri in triangles:
            a, b, c = tri
            for i, j in [(a, b), (b, c), (c, a)]:
                edge_set.add((min(i, j), max(i, j)))

        edges = np.array(sorted(edge_set), dtype=np.int32)
        i_arr = edges[:, 0]
        j_arr = edges[:, 1]

        rest = np.linalg.norm(self.pos[i_arr] - self.pos[j_arr], axis=1)
        K    = AXIAL_STIFFNESS / np.maximum(rest, 1e-12)

        self._beam_i    = i_arr
        self._beam_j    = j_arr
        self._beam_rest = rest
        self._beam_K    = K

    def _build_masses(self, triangles: np.ndarray) -> None:
        """Mass per node = sum of (adjacent triangle area / 3)."""
        n = len(self.pos)
        mass = np.zeros(n, dtype=np.float64)
        for tri in triangles:
            a, b, c = tri
            pa, pb, pc = self.pos[a], self.pos[b], self.pos[c]
            area = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa))
            mass[a] += area / 3.0
            mass[b] += area / 3.0
            mass[c] += area / 3.0
        # Guard against zero-mass nodes (degenerate triangles)
        mass = np.maximum(mass, 1e-12)
        self.mass = mass

    def _build_creases(self, triangles: np.ndarray, pos2d: np.ndarray,
                       orig_creases: list[tuple[np.ndarray, np.ndarray, str]]
                       ) -> None:
        """
        Identify interior edges (shared by exactly 2 triangles) and classify
        them as M/V fold creases or F panel springs.
        """
        # Map each canonical edge → list of triangle indices containing it
        edge_to_tris: dict[tuple[int, int], list[int]] = {}
        tri_edge_map: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

        for t_idx, tri in enumerate(triangles):
            a, b, c = tri
            for (ei, ej), opposite in [
                ((min(a, b), max(a, b)), c),
                ((min(b, c), max(b, c)), a),
                ((min(c, a), max(c, a)), b),
            ]:
                edge_to_tris.setdefault((ei, ej), []).append(t_idx)
                tri_edge_map.setdefault((ei, ej), []).append((ei, ej, opposite))

        crease_a: list[int] = []
        crease_b: list[int] = []
        crease_c: list[int] = []
        crease_d: list[int] = []
        crease_assign: list[str] = []
        crease_full_theta: list[float] = []
        crease_K: list[float] = []

        for edge_key, t_indices in edge_to_tris.items():
            if len(t_indices) != 2:
                continue  # boundary edge

            ei, ej = edge_key
            # Collect opposite nodes for each of the two triangles
            # Find the opposite node for tri 0 and tri 1
            opp_nodes = [None, None]
            for t_pos, t_idx in enumerate(t_indices):
                tri = triangles[t_idx]
                for node in tri:
                    if node != ei and node != ej:
                        opp_nodes[t_pos] = node
                        break

            c_node = opp_nodes[0]
            d_node = opp_nodes[1]
            if c_node is None or d_node is None:
                continue

            # Classify: check if both endpoints lie on the same original crease segment
            pi = pos2d[ei]
            pj = pos2d[ej]
            asgn = 'F'
            for p0, p1, crease_type in orig_creases:
                if _point_on_segment(pi, p0, p1) and _point_on_segment(pj, p0, p1):
                    asgn = crease_type
                    break

            if asgn == 'M':
                full_theta = +np.pi
                K = CREASE_STIFFNESS * np.linalg.norm(pos2d[ej] - pos2d[ei])
            elif asgn == 'V':
                full_theta = -np.pi
                K = CREASE_STIFFNESS * np.linalg.norm(pos2d[ej] - pos2d[ei])
            else:  # 'F' panel
                full_theta = 0.0
                K = PANEL_STIFFNESS * np.linalg.norm(pos2d[ej] - pos2d[ei])

            crease_a.append(ei)
            crease_b.append(ej)
            crease_c.append(c_node)
            crease_d.append(d_node)
            crease_assign.append(asgn)
            crease_full_theta.append(full_theta)
            crease_K.append(K)

        self._crease_a          = np.array(crease_a, dtype=np.int32)
        self._crease_b          = np.array(crease_b, dtype=np.int32)
        self._crease_c          = np.array(crease_c, dtype=np.int32)
        self._crease_d          = np.array(crease_d, dtype=np.int32)
        self._crease_assign     = crease_assign
        self._crease_full_theta = np.array(crease_full_theta, dtype=np.float64)
        self._crease_K          = np.array(crease_K, dtype=np.float64)
        self._crease_target     = np.zeros(len(crease_a), dtype=np.float64)

    # ── Physics ───────────────────────────────────────────────────────────────

    def _beam_forces(self) -> np.ndarray:
        """Vectorized axial spring forces for all beams."""
        n = len(self.pos)
        forces = np.zeros((n, 3), dtype=np.float64)

        pi = self.pos[self._beam_i]
        pj = self.pos[self._beam_j]
        diff = pj - pi
        lengths = np.linalg.norm(diff, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        unit = diff / lengths

        stretch = lengths[:, 0] - self._beam_rest
        F_mag = self._beam_K * stretch          # scalar force magnitude

        # Damping along the edge
        vi = self.vel[self._beam_i]
        vj = self.vel[self._beam_j]
        rel_vel = np.sum((vj - vi) * unit, axis=1)
        damp_mag = PERCENT_DAMPING * rel_vel
        F_total = (F_mag + damp_mag)[:, None] * unit

        np.add.at(forces, self._beam_i,  F_total)
        np.add.at(forces, self._beam_j, -F_total)
        return forces

    def _crease_forces(self) -> np.ndarray:
        """Torsional spring forces for all crease/panel edges (Python loop)."""
        n = len(self.pos)
        forces = np.zeros((n, 3), dtype=np.float64)

        pos = self.pos
        for idx in range(len(self._crease_a)):
            a = self._crease_a[idx]
            b = self._crease_b[idx]
            c = self._crease_c[idx]
            d = self._crease_d[idx]
            K = self._crease_K[idx]
            target = self._crease_target[idx]

            pa, pb, pc, pd = pos[a], pos[b], pos[c], pos[d]

            edge_vec = pb - pa
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-12:
                continue
            edge_dir = edge_vec / edge_len

            # Face normals
            n1_raw = np.cross(pb - pa, pc - pa)
            n2_raw = np.cross(pa - pb, pd - pb)
            n1_len = np.linalg.norm(n1_raw)
            n2_len = np.linalg.norm(n2_raw)
            if n1_len < 1e-12 or n2_len < 1e-12:
                continue
            n1 = n1_raw / n1_len
            n2 = n2_raw / n2_len

            # Dihedral angle via atan2
            cross_n = np.cross(n1, n2)
            sin_theta = np.dot(cross_n, edge_dir)
            cos_theta = np.dot(n1, n2)
            theta = np.arctan2(sin_theta, cos_theta)

            delta  = theta - target
            torque = -K * delta

            # Moment arms (perpendicular distance from c, d to crease line)
            vc = pc - pa
            vd = pd - pa
            vc_perp = vc - np.dot(vc, edge_dir) * edge_dir
            vd_perp = vd - np.dot(vd, edge_dir) * edge_dir
            h_c = np.linalg.norm(vc_perp)
            h_d = np.linalg.norm(vd_perp)
            if h_c < 1e-12 or h_d < 1e-12:
                continue

            # Forces on opposite nodes
            F_c =  (torque / h_c) * n1
            F_d = -(torque / h_d) * n2

            # Reaction on crease nodes (moment balance)
            proj_c = np.dot(pc - pa, edge_dir)
            proj_d = np.dot(pd - pa, edge_dir)
            coef_c_a = 1.0 - proj_c / edge_len
            coef_c_b =       proj_c / edge_len
            coef_d_a = 1.0 - proj_d / edge_len
            coef_d_b =       proj_d / edge_len

            forces[c] += F_c
            forces[d] += F_d
            forces[a] -= coef_c_a * F_c + coef_d_a * F_d
            forces[b] -= coef_c_b * F_c + coef_d_b * F_d

        return forces

    def _euler_step(self) -> None:
        forces = self._beam_forces() + self._crease_forces()
        accel  = forces / self.mass[:, None]
        vel_new = self.vel + accel * DT
        vel_new *= (1.0 - PERCENT_DAMPING * DT)
        self.pos += vel_new * DT
        self.vel  = vel_new
