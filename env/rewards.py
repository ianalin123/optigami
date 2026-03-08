import json
from .verifier import check_all_vertices, geometric_crease_coverage
from .paper_state import PaperState


def load_target(target_path: str) -> dict:
    """Load a .fold target file and return it as a dict."""
    with open(target_path) as f:
        return json.load(f)


def target_crease_edges(target: dict) -> list[dict]:
    """
    Extract crease edges from a FOLD target dict as list of
    {'v1': (x1,y1), 'v2': (x2,y2), 'assignment': 'M'|'V'} dicts.
    """
    verts = target['vertices_coords']
    result = []
    for i, (v1_idx, v2_idx) in enumerate(target['edges_vertices']):
        assignment = target['edges_assignment'][i]
        if assignment in ('M', 'V'):
            result.append({
                'v1': tuple(verts[v1_idx]),
                'v2': tuple(verts[v2_idx]),
                'assignment': assignment,
            })
    return result


def compute_reward(
    state: PaperState,
    action_result: dict,
    target: dict,
) -> dict:
    """
    Compute the full reward dict for a fold action.

    Args:
        state: current PaperState AFTER the action was applied
        action_result: {'valid': bool, 'anchored': bool, 'new_vertices': list, 'errors': list}
        target: FOLD target dict

    Returns dict with keys:
        format, anchored, kawasaki, maekawa, blb, progress, economy, completion, efficiency, total
    """
    r = {}

    # Gate 1: format — did the action parse and apply?
    r['format'] = 1.0 if action_result.get('valid', False) else 0.0
    if not r['format']:
        r['total'] = -0.1
        return r

    # Gate 2: anchoring — were endpoints valid anchor points?
    r['anchored'] = 1.0 if action_result.get('anchored', False) else 0.3

    # Vertex-level validity checks (all interior vertices)
    vertex_scores = check_all_vertices(state.graph)
    r['kawasaki'] = vertex_scores['kawasaki']
    r['maekawa'] = vertex_scores['maekawa']
    r['blb'] = vertex_scores['blb']

    # Geometric progress
    t_edges = target_crease_edges(target)
    coverage, economy = geometric_crease_coverage(state, t_edges)
    r['progress'] = coverage
    r['economy'] = economy

    # Completion bonus: high coverage + all vertex conditions satisfied
    all_valid = (r['kawasaki'] == 1.0 and r['maekawa'] == 1.0 and r['blb'] == 1.0)
    r['completion'] = 10.0 if (r['progress'] > 0.9 and all_valid) else 0.0

    # Step cost
    r['efficiency'] = -0.01

    # Weighted total
    r['total'] = (
        0.05 * r['anchored'] +
        0.08 * r['kawasaki'] +
        0.07 * r['maekawa'] +
        0.05 * r['blb'] +
        0.45 * r['progress'] +
        0.10 * r['economy'] +
        r['completion'] +
        r['efficiency']
    )
    return r


def compute_terminal_reward(state: PaperState, target: dict) -> dict:
    """Compute reward for the final state after a complete fold sequence."""
    fake_result = {'valid': True, 'anchored': True, 'new_vertices': [], 'errors': []}
    return compute_reward(state, fake_result, target)
