"""
Paper — the core geometric data structure for origami simulation.

Stores vertices, edges, faces, fold assignments, fold angles, layer ordering,
and material.  Supports FOLD-format serialization and the face-splitting
operation needed by the fold engine.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .materials import Material, get_material


# ────────────────────────────────────────────────────────────────────
# Helper: 2-D line-segment intersection
# ────────────────────────────────────────────────────────────────────

def _seg_seg_intersect_2d(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray | None:
    """Return the intersection point of segments (p1-p2) and (p3-p4) in 2-D,
    or None if they do not intersect.  Points that lie on the segment
    endpoints are considered intersections (within tolerance *eps*).

    All inputs are shape (2,).
    """
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(denom) < eps:
        return None  # parallel / collinear

    dp = p3 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    u = (dp[0] * d1[1] - dp[1] * d1[0]) / denom

    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return p1 + np.clip(t, 0.0, 1.0) * d1
    return None


# ────────────────────────────────────────────────────────────────────
# Paper dataclass
# ────────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """Origami sheet state.

    Attributes
    ----------
    vertices : np.ndarray, shape (N, 3)
        Vertex positions in 3-D.
    edges : np.ndarray, shape (E, 2), dtype int
        Each row is (v_start, v_end).
    faces : list[list[int]]
        Each face is an ordered list of vertex indices (CCW winding).
    assignments : list[str]
        Per-edge assignment: 'M' (mountain), 'V' (valley), 'B' (boundary),
        'F' (flat / unfolded), 'U' (unassigned).
    fold_angles : np.ndarray, shape (E,)
        Current fold angle (degrees) per edge.
    face_orders : list[tuple[int, int, int]]
        Layer ordering triples (f_i, f_j, +1/-1) meaning f_i is above/below f_j.
    material : Material
        The sheet material.
    rest_lengths : np.ndarray, shape (E,)
        Original (unfolded) edge lengths — used for strain computation.
    original_area : float
        Area of the sheet before any folds.
    """

    vertices: np.ndarray
    edges: np.ndarray
    faces: list[list[int]]
    assignments: list[str]
    fold_angles: np.ndarray
    face_orders: list[tuple[int, int, int]] = field(default_factory=list)
    material: Material = field(default_factory=lambda: get_material("paper"))
    rest_lengths: np.ndarray = field(default_factory=lambda: np.empty(0))
    original_area: float = 0.0
    rest_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    strain_per_vertex: np.ndarray = field(default_factory=lambda: np.empty(0))
    energy: dict = field(default_factory=lambda: {"total": 0.0, "bar": 0.0, "facet": 0.0, "fold": 0.0})
    fold_count: int = 0

    # ── constructors ────────────────────────────────────────────────

    @staticmethod
    def create_flat_sheet(
        width: float = 1.0,
        height: float = 1.0,
        material: Material | None = None,
    ) -> "Paper":
        """Create a flat rectangular sheet with 4 vertices, 5 edges
        (including one diagonal), and 2 triangular faces."""
        mat = material if material is not None else get_material("paper")

        verts = np.array([
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1],  # bottom
            [1, 2],  # right
            [2, 3],  # top
            [3, 0],  # left
            [0, 2],  # diagonal
        ], dtype=np.int64)

        faces: list[list[int]] = [[0, 1, 2], [0, 2, 3]]
        assignments = ["B", "B", "B", "B", "F"]
        fold_angles = np.zeros(len(edges), dtype=np.float64)
        rest_lengths = np.array(
            [np.linalg.norm(verts[e[1]] - verts[e[0]]) for e in edges],
            dtype=np.float64,
        )

        paper = Paper(
            vertices=verts,
            edges=edges,
            faces=faces,
            assignments=assignments,
            fold_angles=fold_angles,
            material=mat,
            rest_lengths=rest_lengths,
            original_area=width * height,
        )
        paper.rest_positions = verts.copy()
        return paper

    # ── dict / prompt serialization (matches mock_env.PaperState.to_dict) ──

    def to_dict(self) -> dict:
        """Return a simplified dict suitable for LLM prompts.

        The format matches ``mock_env.PaperState.to_dict()`` so that the
        trainer reward functions work with either engine.
        """
        bb = self.bounding_box
        return {
            "width": float(bb[0]),
            "height": float(bb[1]),
            "material": {
                "name": self.material.name,
                "thickness_mm": self.material.thickness_mm,
                "youngs_modulus_gpa": self.material.youngs_modulus_gpa,
            },
            "vertices": self.vertices.tolist(),
            "edges": self.edges.tolist(),
            "assignments": list(self.assignments),
            "fold_angles": self.fold_angles.tolist(),
            "num_layers_at_center": self.num_layers,
            "bounding_box": {
                "x": float(bb[0]),
                "y": float(bb[1]),
                "z": float(bb[2]),
            },
        }

    def to_observation_dict(self) -> dict:
        bb = self.bounding_box
        return {
            "vertices_coords": self.vertices.tolist(),
            "edges_vertices": self.edges.tolist(),
            "faces_vertices": self.faces,
            "edges_assignment": list(self.assignments),
            "edges_foldAngle": self.fold_angles.tolist(),
            "num_vertices": len(self.vertices),
            "num_edges": len(self.edges),
            "num_faces": len(self.faces),
            "bounding_box": bb.tolist(),
            "num_layers": self.num_layers,
            "material": {
                "name": self.material.name,
                "thickness_mm": self.material.thickness_mm,
                "youngs_modulus_gpa": self.material.youngs_modulus_gpa,
                "max_strain": self.material.max_strain,
                "poisson_ratio": self.material.poissons_ratio,
            },
            "strain_per_vertex": self.strain_per_vertex.tolist(),
            "energy": dict(self.energy),
            "fold_count": self.fold_count,
            "width": float(self.original_area ** 0.5) if self.original_area > 0 else 1.0,
            "height": float(self.original_area ** 0.5) if self.original_area > 0 else 1.0,
        }

    # ── FOLD format serialization ───────────────────────────────────

    def to_fold_json(self) -> str:
        """Serialize to FOLD JSON format (v1.1 subset)."""
        fold = {
            "file_spec": 1.1,
            "file_creator": "optigami",
            "file_classes": ["singleModel"],
            "frame_classes": ["foldedForm"],
            "vertices_coords": self.vertices.tolist(),
            "edges_vertices": self.edges.tolist(),
            "edges_assignment": self.assignments,
            "edges_foldAngle": self.fold_angles.tolist(),
            "faces_vertices": self.faces,
            "faceOrders": [list(fo) for fo in self.face_orders],
        }
        return json.dumps(fold, indent=2)

    @staticmethod
    def from_fold_json(data: str | dict, material: Material | None = None) -> "Paper":
        """Deserialize from FOLD JSON format."""
        if isinstance(data, str):
            data = json.loads(data)

        verts = np.array(data["vertices_coords"], dtype=np.float64)
        edges = np.array(data["edges_vertices"], dtype=np.int64)
        faces = data.get("faces_vertices", [])
        assignments = data.get("edges_assignment", ["U"] * len(edges))
        fold_angles = np.array(
            data.get("edges_foldAngle", [0.0] * len(edges)),
            dtype=np.float64,
        )
        face_orders = [tuple(fo) for fo in data.get("faceOrders", [])]

        rest_lengths = np.array(
            [np.linalg.norm(verts[e[1]] - verts[e[0]]) for e in edges],
            dtype=np.float64,
        )

        mat = material if material is not None else get_material("paper")

        # Approximate original area from convex hull of initial XY footprint
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(verts[:, :2])
            area = hull.volume  # 2-D ConvexHull.volume is area
        except Exception:
            # Fallback: bounding-box area from XY coordinates
            if len(verts) >= 2:
                ptp = np.ptp(verts[:, :2], axis=0)
                area = float(ptp[0] * ptp[1])
            else:
                area = 0.0

        return Paper(
            vertices=verts,
            edges=edges,
            faces=faces,
            assignments=assignments,
            fold_angles=fold_angles,
            face_orders=face_orders,
            material=mat,
            rest_lengths=rest_lengths,
            original_area=area,
        )

    # ── computed properties ─────────────────────────────────────────

    @property
    def bounding_box(self) -> np.ndarray:
        """Axis-aligned bounding-box dimensions (dx, dy, dz)."""
        if len(self.vertices) == 0:
            return np.zeros(3)
        ptp = np.ptp(self.vertices, axis=0)
        ptp = np.where(np.abs(ptp) < 1e-12, 0.0, ptp)
        # Ensure minimum z height from material thickness * layers
        t = self.material.thickness_mm / 1000.0
        ptp[2] = max(ptp[2], t * self.num_layers)
        return ptp

    @property
    def num_layers(self) -> int:
        """Estimate layer count from face-order triples.

        Falls back to 1 + number of M/V edges as a simple heuristic when
        face_orders is empty.
        """
        if self.face_orders:
            face_ids = set()
            for fo in self.face_orders:
                face_ids.add(fo[0])
                face_ids.add(fo[1])
            return max(len(face_ids), 1)
        # Heuristic: each fold adds one layer
        mv_count = sum(1 for a in self.assignments if a in ("M", "V"))
        return 1 + mv_count

    # ── topology helpers ────────────────────────────────────────────

    def _find_or_add_vertex(self, point_3d: np.ndarray, tol: float = 1e-8) -> int:
        """Return index of an existing vertex close to *point_3d*, or add a
        new vertex and return its index."""
        for i, v in enumerate(self.vertices):
            if np.linalg.norm(v - point_3d) < tol:
                return i
        idx = len(self.vertices)
        self.vertices = np.vstack([self.vertices, point_3d.reshape(1, 3)])
        return idx

    def _find_or_add_edge(self, v1: int, v2: int) -> int:
        """Return index of edge (v1,v2) or (v2,v1), or add a new edge and
        return its index.  New edges get assignment 'F' and fold-angle 0."""
        for i, e in enumerate(self.edges):
            if (e[0] == v1 and e[1] == v2) or (e[0] == v2 and e[1] == v1):
                return i
        idx = len(self.edges)
        self.edges = np.vstack([self.edges, np.array([[v1, v2]], dtype=np.int64)])
        self.assignments.append("F")
        self.fold_angles = np.append(self.fold_angles, 0.0)
        # Rest length for the new edge
        rl = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
        self.rest_lengths = np.append(self.rest_lengths, rl)
        return idx

    # ── face splitting ──────────────────────────────────────────────

    def split_faces_along_line(
        self,
        start_2d: np.ndarray | list,
        end_2d: np.ndarray | list,
    ) -> list[int]:
        """Split every face that the 2-D line (start_2d -> end_2d) crosses.

        The line is infinite for intersection purposes (we test each face
        edge-segment against the full fold-line extent clipped to the paper).

        Returns a list of edge indices that lie *on* the fold line (i.e. the
        newly created edges along the fold path).

        This mutates ``self`` in-place (vertices, edges, faces, assignments,
        fold_angles, rest_lengths are updated).
        """
        start_2d = np.asarray(start_2d, dtype=np.float64)
        end_2d = np.asarray(end_2d, dtype=np.float64)

        fold_edge_indices: list[int] = []
        new_faces: list[list[int]] = []

        faces_to_process = list(range(len(self.faces)))

        for fi in faces_to_process:
            face = self.faces[fi]
            n = len(face)

            # Gather intersection points along the face boundary
            hits: list[tuple[int, np.ndarray]] = []  # (local_edge_index, point_2d)

            for k in range(n):
                v_a = face[k]
                v_b = face[(k + 1) % n]
                pa = self.vertices[v_a][:2]
                pb = self.vertices[v_b][:2]

                pt = _seg_seg_intersect_2d(start_2d, end_2d, pa, pb)
                if pt is not None:
                    hits.append((k, pt))

            # Deduplicate hits that are at the same location (e.g. hitting a vertex)
            if len(hits) >= 2:
                unique_hits: list[tuple[int, np.ndarray]] = [hits[0]]
                for h in hits[1:]:
                    is_dup = False
                    for uh in unique_hits:
                        if np.linalg.norm(h[1] - uh[1]) < 1e-8:
                            is_dup = True
                            break
                    if not is_dup:
                        unique_hits.append(h)
                hits = unique_hits

            if len(hits) < 2:
                # Line does not fully cross this face — keep face as-is
                new_faces.append(face)
                continue

            # We only handle the first two intersection points (one chord across face)
            hit_a_edge_idx, hit_a_pt = hits[0]
            hit_b_edge_idx, hit_b_pt = hits[1]

            # Create / find 3-D vertices at intersection points (z=0 for flat, interpolated otherwise)
            def _interp_z(pt2d: np.ndarray, edge_local: int) -> np.ndarray:
                """Interpolate z from the edge endpoints."""
                v_a = face[edge_local]
                v_b = face[(edge_local + 1) % n]
                pa = self.vertices[v_a]
                pb = self.vertices[v_b]
                seg = pb[:2] - pa[:2]
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-12:
                    return np.array([pt2d[0], pt2d[1], pa[2]])
                t = np.linalg.norm(pt2d - pa[:2]) / seg_len
                t = np.clip(t, 0.0, 1.0)
                z = pa[2] + t * (pb[2] - pa[2])
                return np.array([pt2d[0], pt2d[1], z])

            pt_a_3d = _interp_z(hit_a_pt, hit_a_edge_idx)
            pt_b_3d = _interp_z(hit_b_pt, hit_b_edge_idx)

            idx_a = self._find_or_add_vertex(pt_a_3d)
            idx_b = self._find_or_add_vertex(pt_b_3d)

            if idx_a == idx_b:
                new_faces.append(face)
                continue

            # Add the fold-line edge between the two intersection points
            fold_eidx = self._find_or_add_edge(idx_a, idx_b)
            fold_edge_indices.append(fold_eidx)

            # ── Split the face into two sub-faces ──
            # Walk around the face vertices, inserting idx_a and idx_b at the
            # appropriate positions, then split into two loops.
            ordered_verts = list(face)

            # Insert intersection vertices into the vertex ring if not already present
            def _insert_after(ring: list[int], after_local: int, vid: int) -> list[int]:
                """Insert *vid* after position *after_local* if it is not already
                adjacent in the ring at that position."""
                pos = after_local + 1
                if ring[after_local % len(ring)] == vid:
                    return ring
                if ring[pos % len(ring)] == vid:
                    return ring
                return ring[:pos] + [vid] + ring[pos:]

            # Determine insertion order — always insert the one with the
            # larger local-edge index first so that the earlier index stays valid.
            if hit_a_edge_idx <= hit_b_edge_idx:
                ordered_verts = _insert_after(ordered_verts, hit_b_edge_idx, idx_b)
                # Recompute hit_a_edge_idx offset if idx_b was inserted before it
                # (it shouldn't be, since hit_b >= hit_a, but guard anyway)
                a_pos = hit_a_edge_idx
                ordered_verts = _insert_after(ordered_verts, a_pos, idx_a)
            else:
                ordered_verts = _insert_after(ordered_verts, hit_a_edge_idx, idx_a)
                ordered_verts = _insert_after(ordered_verts, hit_b_edge_idx, idx_b)

            # Now split the ring at idx_a and idx_b
            try:
                pos_a = ordered_verts.index(idx_a)
                pos_b = ordered_verts.index(idx_b)
            except ValueError:
                new_faces.append(face)
                continue

            if pos_a > pos_b:
                pos_a, pos_b = pos_b, pos_a

            loop1 = ordered_verts[pos_a: pos_b + 1]
            loop2 = ordered_verts[pos_b:] + ordered_verts[: pos_a + 1]

            # Only keep faces with >= 3 unique vertices
            for loop in (loop1, loop2):
                unique = list(dict.fromkeys(loop))  # preserve order, dedupe
                if len(unique) >= 3:
                    new_faces.append(unique)
                    # Ensure all edges of this new face exist
                    for k in range(len(unique)):
                        self._find_or_add_edge(unique[k], unique[(k + 1) % len(unique)])

        self.faces = new_faces
        return fold_edge_indices

    # ── vertex side test ────────────────────────────────────────────

    def get_vertices_on_side(
        self,
        line_start: np.ndarray | list,
        line_end: np.ndarray | list,
        side: str = "positive",
    ) -> list[int]:
        """Return indices of vertices on one side of a 2-D line.

        *side* can be ``"positive"`` or ``"negative"``.  The positive side is
        defined by the left-hand normal of (line_end - line_start).
        """
        ls = np.asarray(line_start, dtype=np.float64)[:2]
        le = np.asarray(line_end, dtype=np.float64)[:2]
        d = le - ls
        normal = np.array([-d[1], d[0]])

        indices: list[int] = []
        for i, v in enumerate(self.vertices):
            dot = np.dot(v[:2] - ls, normal)
            if side == "positive" and dot > 1e-9:
                indices.append(i)
            elif side == "negative" and dot < -1e-9:
                indices.append(i)
        return indices

    # ── deep copy ───────────────────────────────────────────────────

    def copy(self) -> "Paper":
        """Return an independent deep copy."""
        return Paper(
            vertices=self.vertices.copy(),
            edges=self.edges.copy(),
            faces=copy.deepcopy(self.faces),
            assignments=list(self.assignments),
            fold_angles=self.fold_angles.copy(),
            face_orders=list(self.face_orders),
            material=Material(
                name=self.material.name,
                thickness_mm=self.material.thickness_mm,
                youngs_modulus_gpa=self.material.youngs_modulus_gpa,
                max_strain=self.material.max_strain,
                poissons_ratio=self.material.poissons_ratio,
            ),
            rest_lengths=self.rest_lengths.copy(),
            original_area=self.original_area,
            rest_positions=self.rest_positions.copy(),
            strain_per_vertex=self.strain_per_vertex.copy(),
            energy=dict(self.energy),
            fold_count=self.fold_count,
        )
