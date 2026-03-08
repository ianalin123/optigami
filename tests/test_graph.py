import numpy as np
import pytest
from env.graph import CreaseGraph, VERTEX_TOL


def test_init_boundary():
    g = CreaseGraph()
    assert len(g.vertices) == 4
    assert len(g.edges) == 4
    for eid, (v1, v2, assignment) in g.edges.items():
        assert assignment == 'B'
    assert g.interior_vertices() == []


def test_add_vertex_dedup():
    g = CreaseGraph()
    id1 = g.add_vertex(0.5, 0.5)
    id2 = g.add_vertex(0.5, 0.5)
    assert id1 == id2


def test_add_vertex_dedup_near():
    g = CreaseGraph()
    id1 = g.add_vertex(0.5, 0.5)
    id2 = g.add_vertex(0.5 + VERTEX_TOL * 0.5, 0.5)
    assert id1 == id2


def test_cyclic_order():
    g = CreaseGraph()
    center_id = g.add_vertex(0.5, 0.5)

    right_id = g.add_vertex(0.8, 0.5)   # 0 degrees
    top_id = g.add_vertex(0.5, 0.8)     # 90 degrees
    left_id = g.add_vertex(0.2, 0.5)    # 180 degrees
    bottom_id = g.add_vertex(0.5, 0.2)  # 270 degrees / -90 degrees

    e_right = g.add_edge(center_id, right_id, 'M')
    e_top = g.add_edge(center_id, top_id, 'M')
    e_left = g.add_edge(center_id, left_id, 'M')
    e_bottom = g.add_edge(center_id, bottom_id, 'M')

    cyclic = g.get_cyclic_edges(center_id)
    # Sorted by angle ascending: right(0), top(90), left(180), bottom(-90 → 270)
    # arctan2 for bottom gives -pi/2 which sorts before 0 in ascending order
    # So actual ascending order: bottom(-pi/2), right(0), top(pi/2), left(pi)
    assert len(cyclic) == 4

    def edge_angle(eid):
        ev1, ev2, _ = g.edges[eid]
        other_id = ev2 if ev1 == center_id else ev1
        ox, oy = g.vertices[other_id]
        cx, cy = g.vertices[center_id]
        return float(np.arctan2(oy - cy, ox - cx))

    angles = [edge_angle(eid) for eid in cyclic]
    assert angles == sorted(angles), "Edges should be sorted by ascending angle"

    assert e_right in cyclic
    assert e_top in cyclic
    assert e_left in cyclic
    assert e_bottom in cyclic

    # Verify specific order: bottom < right < top < left in angle space
    pos = {eid: i for i, eid in enumerate(cyclic)}
    assert pos[e_bottom] < pos[e_right] < pos[e_top] < pos[e_left]


def test_interior_vertices_empty():
    g = CreaseGraph()
    assert g.interior_vertices() == []


def test_interior_vertices_with_crease_intersection():
    g = CreaseGraph()
    center_id = g.add_vertex(0.5, 0.5)
    assert center_id in g.interior_vertices()


def test_split_edge():
    g = CreaseGraph()
    # Find the bottom boundary edge (0,0)-(1,0) which is edge 0: v0-v1
    original_edge_id = None
    for eid, (v1, v2, assignment) in g.edges.items():
        x1, y1 = g.vertices[v1]
        x2, y2 = g.vertices[v2]
        if {(x1, y1), (x2, y2)} == {(0.0, 0.0), (1.0, 0.0)}:
            original_edge_id = eid
            original_v1 = v1
            original_v2 = v2
            break

    assert original_edge_id is not None

    mid_id = g.add_vertex(0.5, 0.0)
    eid1, eid2 = g.split_edge(original_edge_id, mid_id)

    assert original_edge_id not in g.edges

    assert eid1 in g.edges
    assert eid2 in g.edges

    _, _, a1 = g.edges[eid1]
    _, _, a2 = g.edges[eid2]
    assert a1 == 'B'
    assert a2 == 'B'

    def edge_vertex_set(eid):
        v1, v2, _ = g.edges[eid]
        return {v1, v2}

    assert mid_id in edge_vertex_set(eid1)
    assert mid_id in edge_vertex_set(eid2)
    assert original_v1 in edge_vertex_set(eid1) or original_v1 in edge_vertex_set(eid2)
    assert original_v2 in edge_vertex_set(eid1) or original_v2 in edge_vertex_set(eid2)
