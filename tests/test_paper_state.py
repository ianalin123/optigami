import pytest
from env.paper_state import PaperState, UNIT_SQUARE_CORNERS
from env.graph import VERTEX_TOL


def test_single_crease_no_interior_vertices():
    paper = PaperState()
    result = paper.add_crease([0.0, 0.5], [1.0, 0.5], 'V')
    assert result['valid'] is True
    interior = paper.graph.interior_vertices()
    assert interior == [], f"Expected no interior vertices, got {interior}"


def test_anchor_points_initial():
    paper = PaperState()
    anchors = paper.anchor_points()
    for corner in UNIT_SQUARE_CORNERS:
        assert any(
            abs(ax - corner[0]) < VERTEX_TOL and abs(ay - corner[1]) < VERTEX_TOL
            for ax, ay in anchors
        ), f"Corner {corner} not found in anchor_points"


def test_anchor_points_grow():
    paper = PaperState()
    result = paper.add_crease([0.0, 0.5], [1.0, 0.5], 'V')
    assert result['valid'] is True

    anchors = paper.anchor_points()

    def has_point(px, py):
        return any(abs(ax - px) < VERTEX_TOL and abs(ay - py) < VERTEX_TOL for ax, ay in anchors)

    assert has_point(0.0, 0.5), "(0, 0.5) should be in anchor_points after crease"
    assert has_point(1.0, 0.5), "(1, 0.5) should be in anchor_points after crease"


def test_invalid_assignment():
    paper = PaperState()
    result = paper.add_crease([0.0, 0.5], [1.0, 0.5], 'X')
    assert result['valid'] is False
    assert 'invalid_assignment' in result['errors']


def test_fold_history():
    paper = PaperState()
    paper.add_crease([0.0, 0.5], [1.0, 0.5], 'M')
    assert len(paper.fold_history) == 1


def test_unanchored_returns_false_anchored():
    paper = PaperState()
    result = paper.add_crease([0.3, 0.3], [0.7, 0.7], 'M')
    assert result['anchored'] is False


def test_crease_edges_returned():
    paper = PaperState()
    paper.add_crease([0.0, 0.5], [1.0, 0.5], 'M')
    edges = paper.crease_edges()
    assert len(edges) >= 1
    for e in edges:
        assert e['assignment'] in ('M', 'V')
        assert 'v1' in e
        assert 'v2' in e


def test_two_intersecting_creases():
    paper = PaperState()
    r1 = paper.add_crease([0.0, 0.5], [1.0, 0.5], 'M')
    r2 = paper.add_crease([0.5, 0.0], [0.5, 1.0], 'V')
    assert r1['valid'] is True
    assert r2['valid'] is True
    interior = paper.graph.interior_vertices()
    assert len(interior) >= 1
    coords = [paper.graph.vertices[vid] for vid in interior]
    assert any(abs(x - 0.5) < VERTEX_TOL and abs(y - 0.5) < VERTEX_TOL for x, y in coords)
