"""
Validates all .fold target files against origami theorems.
Run directly: python -m env.targets.validator
"""
import json
import os
import sys
from pathlib import Path

from ..graph import CreaseGraph
from ..verifier import check_kawasaki_at_vertex, check_maekawa_at_vertex, check_blb_at_vertex


def build_graph_from_fold(fold_data: dict) -> CreaseGraph:
    """
    Reconstruct a CreaseGraph from a FOLD JSON dict.
    Used to validate target files.
    """
    graph = CreaseGraph()

    verts = fold_data['vertices_coords']
    edges = fold_data['edges_vertices']
    assignments = fold_data['edges_assignment']

    # Map file vertex indices to graph vertex IDs
    vert_map = {}
    for i, (x, y) in enumerate(verts):
        vid = graph.add_vertex(float(x), float(y))
        vert_map[i] = vid

    # Add edges (boundary edges from init may already exist, add_edge handles dedup)
    for i, (v1_idx, v2_idx) in enumerate(edges):
        v1_id = vert_map[v1_idx]
        v2_id = vert_map[v2_idx]
        assignment = assignments[i]
        graph.add_edge(v1_id, v2_id, assignment)

    return graph


def validate_target(fold_path: str) -> dict:
    """
    Validate a single .fold target file.
    Returns {'file': str, 'valid': bool, 'issues': list[str], 'interior_vertices': int}
    """
    with open(fold_path) as f:
        fold_data = json.load(f)

    issues = []

    # Basic structure checks
    required = ['vertices_coords', 'edges_vertices', 'edges_assignment', 'edges_foldAngle']
    for field in required:
        if field not in fold_data:
            issues.append(f"Missing field: {field}")

    if issues:
        return {'file': os.path.basename(fold_path), 'valid': False, 'issues': issues, 'interior_vertices': -1}

    n_edges = len(fold_data['edges_vertices'])
    if len(fold_data['edges_assignment']) != n_edges:
        issues.append("edges_assignment length mismatch")
    if len(fold_data['edges_foldAngle']) != n_edges:
        issues.append("edges_foldAngle length mismatch")

    # Build graph and check theorems
    graph = build_graph_from_fold(fold_data)
    interior = graph.interior_vertices()

    for v_id in interior:
        ok, alt_sum = check_kawasaki_at_vertex(v_id, graph)
        if not ok:
            issues.append(f"Kawasaki violated at vertex {v_id} (alt_sum={alt_sum:.6f})")

        if not check_maekawa_at_vertex(v_id, graph):
            issues.append(f"Maekawa violated at vertex {v_id}")

        blb_violations = check_blb_at_vertex(v_id, graph)
        if blb_violations:
            issues.append(f"BLB violated at vertex {v_id}: {blb_violations}")

    return {
        'file': os.path.basename(fold_path),
        'valid': len(issues) == 0,
        'issues': issues,
        'interior_vertices': len(interior),
    }


def validate_all(targets_dir: str = None) -> bool:
    """Validate all .fold files in the targets directory. Returns True if all pass."""
    if targets_dir is None:
        targets_dir = Path(__file__).parent

    all_pass = True
    fold_files = sorted(Path(targets_dir).glob('*.fold'))

    if not fold_files:
        print("No .fold files found")
        return False

    for fold_path in fold_files:
        result = validate_target(str(fold_path))
        status = "OK" if result['valid'] else "FAIL"
        n_interior = result['interior_vertices']
        print(f"  [{status}] {result['file']} — {n_interior} interior vertices")
        if result['issues']:
            for issue in result['issues']:
                print(f"         ! {issue}")
        if not result['valid']:
            all_pass = False

    return all_pass


if __name__ == '__main__':
    print("Validating targets...")
    ok = validate_all()
    sys.exit(0 if ok else 1)
