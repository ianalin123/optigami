import numpy as np
from .graph import CreaseGraph
from .paper_state import PaperState


def _compute_sector_angles(vertex_id: int, graph: CreaseGraph) -> list[float]:
    """Compute consecutive sector angles (CCW) at a vertex from its cyclic edges."""
    cyclic_edges = graph.get_cyclic_edges(vertex_id)
    n = len(cyclic_edges)
    vx, vy = graph.vertices[vertex_id]

    angles = []
    for eid in cyclic_edges:
        ev1, ev2, _ = graph.edges[eid]
        other_id = ev2 if ev1 == vertex_id else ev1
        ox, oy = graph.vertices[other_id]
        angles.append(np.arctan2(oy - vy, ox - vx))

    sectors = []
    for i in range(n):
        diff = angles[(i + 1) % n] - angles[i]
        if diff < 0:
            diff += 2 * np.pi
        if diff > 2 * np.pi:
            diff -= 2 * np.pi
        sectors.append(diff)

    return sectors


def check_kawasaki_at_vertex(vertex_id: int, graph: CreaseGraph) -> tuple[bool, float]:
    """
    Checks Kawasaki-Justin theorem at a single vertex.

    Kawasaki: at an interior vertex with 2n creases, the alternating sum
    of consecutive sector angles = 0.
    Equivalently: sum(odd-indexed sectors) == sum(even-indexed sectors) == π.

    Returns (satisfied: bool, |alternating_sum|: float).
    Returns (True, 0.0) for vertices with degree < 4 (not an interior fold vertex yet).
    Returns (False, inf) for odd-degree vertices (impossible for flat folds).
    """
    cyclic_edges = graph.get_cyclic_edges(vertex_id)
    n = len(cyclic_edges)

    if n % 2 != 0:
        return (False, float('inf'))

    if n < 4:
        return (True, 0.0)

    sectors = _compute_sector_angles(vertex_id, graph)
    alt_sum = sum(s * ((-1) ** i) for i, s in enumerate(sectors))
    return (abs(alt_sum) < 1e-9, abs(alt_sum))


def check_maekawa_at_vertex(vertex_id: int, graph: CreaseGraph) -> bool:
    """
    Checks Maekawa-Justin theorem at a single vertex.

    Maekawa: |M - V| == 2 where M, V are counts of mountain/valley fold edges
    at the vertex. BOUNDARY edges ('B') are NOT counted.

    Returns True if satisfied or if vertex has fewer than 4 fold edges (not yet active).
    """
    edge_ids = graph.vertex_edges[vertex_id]
    fold_edges = [
        eid for eid in edge_ids
        if graph.edges[eid][2] in ('M', 'V')
    ]

    if len(fold_edges) < 4:
        return True

    m_count = sum(1 for eid in fold_edges if graph.edges[eid][2] == 'M')
    v_count = sum(1 for eid in fold_edges if graph.edges[eid][2] == 'V')
    return abs(m_count - v_count) == 2


def check_blb_at_vertex(vertex_id: int, graph: CreaseGraph) -> list[tuple[int, int]]:
    """
    Checks Big-Little-Big lemma at a single vertex.

    BLB: if sector angle i is a strict local minimum (smaller than both neighbors),
    the fold edges bounding that sector must have OPPOSITE MV assignments.

    Returns list of (edge_a_id, edge_b_id) pairs where BLB is violated.
    Empty list = no violations.
    """
    cyclic_edges = graph.get_cyclic_edges(vertex_id)
    n = len(cyclic_edges)

    if n < 4:
        return []

    sectors = _compute_sector_angles(vertex_id, graph)
    violations = []

    for i in range(n):
        prev_sector = sectors[(i - 1) % n]
        next_sector = sectors[(i + 1) % n]

        if sectors[i] < prev_sector and sectors[i] < next_sector:
            edge_a = cyclic_edges[i]
            edge_b = cyclic_edges[(i + 1) % n]

            assign_a = graph.edges[edge_a][2]
            assign_b = graph.edges[edge_b][2]

            if assign_a in ('M', 'V') and assign_b in ('M', 'V'):
                if assign_a == assign_b:
                    violations.append((edge_a, edge_b))

    return violations


def _angle_diff(a1: float, a2: float) -> float:
    """Minimum angle difference between two directed lines (considering 180° symmetry)."""
    diff = abs(a1 - a2) % np.pi
    return min(diff, np.pi - diff)


def geometric_crease_coverage(
    state: PaperState,
    target_edges: list[dict],
    tol_pos: float = 0.05,
    tol_angle_deg: float = 5.0,
) -> tuple[float, float]:
    """
    Computes how well the current crease pattern matches the target.

    Args:
        target_edges: list of {'v1': (x1,y1), 'v2': (x2,y2), 'assignment': 'M'|'V'}

    Returns:
        (coverage, economy)
        coverage: fraction of target creases matched [0, 1]
        economy: penalty for excess creases [0, 1], 1.0 = no excess
    """
    current_edges = state.crease_edges()
    tol_angle_rad = np.deg2rad(tol_angle_deg)

    matched = 0
    for target in target_edges:
        tx1, ty1 = target['v1']
        tx2, ty2 = target['v2']
        t_mid = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
        t_angle = np.arctan2(ty2 - ty1, tx2 - tx1)

        for current in current_edges:
            cx1, cy1 = current['v1']
            cx2, cy2 = current['v2']
            c_mid = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
            c_angle = np.arctan2(cy2 - cy1, cx2 - cx1)

            mid_dist = np.hypot(c_mid[0] - t_mid[0], c_mid[1] - t_mid[1])
            angle_distance = _angle_diff(c_angle, t_angle)

            if mid_dist <= tol_pos and angle_distance <= tol_angle_rad:
                matched += 1
                break

    coverage = matched / max(len(target_edges), 1)
    n_excess = max(0, len(current_edges) - len(target_edges))
    economy = max(0.0, 1.0 - n_excess / max(len(target_edges), 1))
    return (coverage, economy)


def check_all_vertices(graph: CreaseGraph) -> dict:
    """
    Run all vertex-level checks on every interior vertex.

    Returns dict with:
        'kawasaki': float  # fraction of interior vertices passing Kawasaki [0,1]
        'maekawa': float   # fraction passing Maekawa [0,1]
        'blb': float       # fraction with no BLB violations [0,1]
        'n_interior': int  # number of interior vertices checked
        'per_vertex': list[dict]  # per-vertex details
    """
    interior = graph.interior_vertices()

    if not interior:
        return {
            'kawasaki': 1.0,
            'maekawa': 1.0,
            'blb': 1.0,
            'n_interior': 0,
            'per_vertex': [],
        }

    per_vertex = []
    kaw_pass = 0
    mae_pass = 0
    blb_pass = 0

    for vid in interior:
        kaw_ok, kaw_val = check_kawasaki_at_vertex(vid, graph)
        mae_ok = check_maekawa_at_vertex(vid, graph)
        blb_violations = check_blb_at_vertex(vid, graph)
        blb_ok = len(blb_violations) == 0

        kaw_pass += int(kaw_ok)
        mae_pass += int(mae_ok)
        blb_pass += int(blb_ok)

        per_vertex.append({
            'vertex_id': vid,
            'kawasaki_ok': kaw_ok,
            'kawasaki_error': kaw_val,
            'maekawa_ok': mae_ok,
            'blb_violations': blb_violations,
        })

    n = len(interior)
    return {
        'kawasaki': kaw_pass / n,
        'maekawa': mae_pass / n,
        'blb': blb_pass / n,
        'n_interior': n,
        'per_vertex': per_vertex,
    }
