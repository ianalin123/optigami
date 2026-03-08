"""
Mock origami environment for trainer development.

Returns fake PaperState responses so we can iterate on the GRPO loop
without waiting for the real physics engine. The mock applies geometric
transforms (vertex rotations around fold lines) but skips energy/strain
computation — those return plausible dummy values.
"""

import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Material:
    name: str = "paper"
    thickness_mm: float = 0.1
    youngs_modulus_gpa: float = 2.0
    max_strain: float = 0.03  # 3%


@dataclass
class PaperState:
    vertices: np.ndarray           # (N, 3)
    edges: np.ndarray              # (E, 2)
    faces: list[list[int]]
    assignments: list[str]         # M/V/B per edge
    fold_angles: np.ndarray        # (E,) degrees

    rest_lengths: np.ndarray       # (E,)
    strain: np.ndarray             # (N,)
    energy: float = 0.0

    face_orders: list[tuple] = field(default_factory=list)
    num_layers: int = 1

    material: Material = field(default_factory=Material)

    bounding_box: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.0]))
    deployment_ratio: float = 1.0
    is_valid: bool = True
    kawasaki_violation: float = 0.0
    maekawa_violation: float = 0.0
    self_intersections: int = 0

    def to_dict(self) -> dict:
        return {
            "width": float(self.bounding_box[0]),
            "height": float(self.bounding_box[1]),
            "material": {
                "name": self.material.name,
                "thickness_mm": self.material.thickness_mm,
                "youngs_modulus_gpa": self.material.youngs_modulus_gpa,
            },
            "vertices": self.vertices.tolist(),
            "edges": self.edges.tolist(),
            "assignments": self.assignments,
            "fold_angles": self.fold_angles.tolist(),
            "num_layers_at_center": self.num_layers,
            "bounding_box": {
                "x": float(self.bounding_box[0]),
                "y": float(self.bounding_box[1]),
                "z": float(self.bounding_box[2]),
            },
        }


def create_flat_sheet(width: float = 1.0, height: float = 1.0,
                      material: Material | None = None) -> PaperState:
    """Create a flat rectangular sheet with 4 vertices, 5 edges (incl diagonal), 2 faces."""
    verts = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0],
    ], dtype=float)

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # boundary
        [0, 2],  # diagonal
    ], dtype=int)

    faces = [[0, 1, 2], [0, 2, 3]]
    assignments = ["B", "B", "B", "B", "F"]  # boundary + flat diagonal
    fold_angles = np.zeros(len(edges))
    rest_lengths = np.array([np.linalg.norm(verts[e[1]] - verts[e[0]]) for e in edges])
    strain = np.zeros(len(verts))

    mat = material or Material()
    return PaperState(
        vertices=verts, edges=edges, faces=faces,
        assignments=assignments, fold_angles=fold_angles,
        rest_lengths=rest_lengths, strain=strain,
        material=mat,
        bounding_box=np.array([width, height, 0.0]),
    )


def _rotate_points(points: np.ndarray, axis_point: np.ndarray,
                   axis_dir: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate points around an arbitrary axis using Rodrigues' formula."""
    k = axis_dir / np.linalg.norm(axis_dir)
    translated = points - axis_point
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated = (translated * cos_a +
               np.cross(k, translated) * sin_a +
               k * (np.dot(translated, k).reshape(-1, 1)) * (1 - cos_a))
    return rotated + axis_point


def apply_fold_mock(state: PaperState, fold: dict) -> tuple[PaperState, str | None]:
    """
    Apply a single fold operation to the paper state (mock version).

    fold = {
        "type": "valley" | "mountain",
        "line": {"start": [x, y], "end": [x, y]},
        "angle": 0-180
    }

    Returns (new_state, error_string_or_None).
    """
    fold_type = fold.get("type", "valley")
    line = fold.get("line", {})
    angle_deg = fold.get("angle", 180)

    start = np.array(line.get("start", [0, 0]), dtype=float)
    end = np.array(line.get("end", [0, 0]), dtype=float)

    if np.allclose(start, end):
        return state, "Fold line has zero length"

    if fold_type not in ("valley", "mountain"):
        return state, f"Unknown fold type: {fold_type}"

    if not (0 < angle_deg <= 180):
        return state, f"Angle must be in (0, 180], got {angle_deg}"

    # Fold direction: valley folds up (+z), mountain folds down (-z)
    sign = 1.0 if fold_type == "valley" else -1.0
    angle_rad = sign * math.radians(angle_deg)

    # Determine which vertices are on the "folding" side of the line
    line_dir_2d = end - start
    normal_2d = np.array([-line_dir_2d[1], line_dir_2d[0]])  # perpendicular

    new_verts = state.vertices.copy()
    for i, v in enumerate(new_verts):
        point_2d = v[:2] - start
        side = np.dot(point_2d, normal_2d)
        if side > 1e-9:  # on the positive side → rotate
            axis_point = np.array([start[0], start[1], 0.0])
            axis_dir = np.array([line_dir_2d[0], line_dir_2d[1], 0.0])
            new_verts[i] = _rotate_points(
                v.reshape(1, -1), axis_point, axis_dir, angle_rad
            ).flatten()

    # Update bounding box (clamp near-zero values from floating point)
    bb = np.ptp(new_verts, axis=0)  # max - min per axis
    bb = np.where(np.abs(bb) < 1e-10, 0.0, bb)
    # Add minimum thickness per layer (material thickness)
    thickness = state.material.thickness_mm / 1000.0  # convert mm to m
    num_layers = state.num_layers + 1
    bb[2] = max(bb[2], thickness * num_layers)

    # Mock strain: small random value per vertex
    new_strain = np.random.uniform(0, 0.01, len(new_verts))

    # Mock energy
    new_energy = state.energy + 0.1 * angle_deg / 180.0

    # Update assignments — add new edge as M or V
    new_assignments = state.assignments.copy()

    # Deployment ratio estimate: each full fold (180°) halves the area in one direction.
    # Partial folds reduce proportionally. This is a mock approximation —
    # the real engine will compute from actual face overlaps.
    fold_factor = angle_deg / 180.0  # 1.0 for full fold, 0.5 for 90°, etc.
    deploy_ratio = state.deployment_ratio * (1.0 - 0.5 * fold_factor)

    new_state = PaperState(
        vertices=new_verts,
        edges=state.edges.copy(),
        faces=state.faces.copy(),
        assignments=new_assignments,
        fold_angles=state.fold_angles.copy(),
        rest_lengths=state.rest_lengths.copy(),
        strain=new_strain,
        energy=new_energy,
        material=state.material,
        bounding_box=bb,
        deployment_ratio=deploy_ratio,
        num_layers=state.num_layers + 1,
        is_valid=True,
        kawasaki_violation=0.0,
        maekawa_violation=0.0,
        self_intersections=0,
    )
    return new_state, None


def execute_fold_strategy(strategy_fn, paper_state: PaperState,
                          max_folds: int = 20) -> tuple[PaperState, list[dict], str | None]:
    """
    Execute a fold_strategy function against the mock environment.

    Returns (final_state, applied_folds, error_or_None).
    """
    state_dict = paper_state.to_dict()
    try:
        folds = strategy_fn(state_dict)
    except Exception as e:
        return paper_state, [], f"Strategy function raised: {e}"

    if not isinstance(folds, list):
        return paper_state, [], "Strategy must return a list of fold dicts"

    applied = []
    current = paper_state
    for i, fold in enumerate(folds):
        if i >= max_folds:
            break
        if not isinstance(fold, dict):
            return current, applied, f"Fold {i} is not a dict"

        current, error = apply_fold_mock(current, fold)
        if error:
            return current, applied, f"Fold {i} failed: {error}"
        applied.append(fold)

    return current, applied, None
