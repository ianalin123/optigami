import pytest
import numpy as np
from env.graph import CreaseGraph
from env.paper_state import PaperState
from env.verifier import (
    check_kawasaki_at_vertex,
    check_maekawa_at_vertex,
    check_blb_at_vertex,
    geometric_crease_coverage,
    check_all_vertices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cross_graph(center_coords=(0.5, 0.5), assignment='M') -> tuple[CreaseGraph, int]:
    """
    Degree-4 vertex at center with 4 spokes pointing N/S/E/W.
    All spokes have the given assignment.
    """
    g = CreaseGraph()
    cx, cy = center_coords
    vid = g.add_vertex(cx, cy)

    neighbors = [
        (0.0, cy),   # left  (180°)
        (1.0, cy),   # right (0°)
        (cx, 0.0),   # down  (-90°)
        (cx, 1.0),   # up    (90°)
    ]
    for nx, ny in neighbors:
        nid = g.add_vertex(nx, ny)
        g.add_edge(vid, nid, assignment)

    return g, vid


# ---------------------------------------------------------------------------
# Kawasaki tests
# ---------------------------------------------------------------------------

def test_kawasaki_no_interior_vertices():
    paper = PaperState()
    paper.add_crease([0, 0.5], [1, 0.5], 'V')
    assert paper.graph.interior_vertices() == []
    result = check_all_vertices(paper.graph)
    assert result['kawasaki'] == 1.0
    assert result['n_interior'] == 0


def test_kawasaki_valid_degree4_vertex():
    """Equal 90° sectors → alternating sum = 0 → Kawasaki satisfied."""
    g, vid = make_cross_graph()
    ok, err = check_kawasaki_at_vertex(vid, g)
    assert ok == True
    assert err == pytest.approx(0.0, abs=1e-9)


def test_kawasaki_invalid_vertex():
    """
    Manually construct a degree-4 vertex whose sectors are 60°,120°,80°,100°.
    Alternating sum = 60 - 120 + 80 - 100 = -80° ≠ 0 → should fail.
    """
    g = CreaseGraph()
    cx, cy = 0.5, 0.5
    vid = g.add_vertex(cx, cy)

    # Place neighbours at specific angles so sectors are exactly as desired.
    # Sectors are measured CCW between consecutive rays.
    # We choose ray angles (from center) in ascending arctan2 order:
    #   a0 = 0°
    #   a1 = 60°    (sector0 = 60°)
    #   a2 = 180°   (sector1 = 120°)
    #   a3 = 260°   = -100°  (sector2 = 80°)
    #   sector3 (wraparound to a0) = 360° - 260° = 100°
    # alt_sum = 60 - 120 + 80 - 100 = -80° → |alt_sum| ≈ 1.396 rad
    r = 0.3
    angles_deg = [0.0, 60.0, 180.0, 260.0]
    for deg in angles_deg:
        rad = np.deg2rad(deg)
        nx = cx + r * np.cos(rad)
        ny = cy + r * np.sin(rad)
        nid = g.add_vertex(nx, ny)
        g.add_edge(vid, nid, 'M')

    ok, err = check_kawasaki_at_vertex(vid, g)
    assert ok == False
    expected_err = abs(np.deg2rad(60 - 120 + 80 - 100))
    assert err == pytest.approx(expected_err, abs=1e-6)


# ---------------------------------------------------------------------------
# Maekawa tests
# ---------------------------------------------------------------------------

def test_maekawa_excludes_boundary():
    """
    Boundary edges at a vertex should NOT count toward M/V tally.
    A corner vertex has only boundary edges; Maekawa should return True
    (fewer than 4 fold edges → vacuously satisfied).
    """
    g = CreaseGraph()
    corner_id = 0  # vertex (0,0)
    assert check_maekawa_at_vertex(corner_id, g) is True


def test_maekawa_valid():
    """3 M + 1 V → |3-1| = 2 → True."""
    g = CreaseGraph()
    cx, cy = 0.5, 0.5
    vid = g.add_vertex(cx, cy)

    r = 0.3
    angles_deg = [0.0, 90.0, 180.0, 270.0]
    assignments = ['M', 'M', 'M', 'V']
    for deg, asgn in zip(angles_deg, assignments):
        rad = np.deg2rad(deg)
        nid = g.add_vertex(cx + r * np.cos(rad), cy + r * np.sin(rad))
        g.add_edge(vid, nid, asgn)

    assert check_maekawa_at_vertex(vid, g) is True


def test_maekawa_invalid():
    """2 M + 2 V → |2-2| = 0 → False."""
    g = CreaseGraph()
    cx, cy = 0.5, 0.5
    vid = g.add_vertex(cx, cy)

    r = 0.3
    angles_deg = [0.0, 90.0, 180.0, 270.0]
    assignments = ['M', 'M', 'V', 'V']
    for deg, asgn in zip(angles_deg, assignments):
        rad = np.deg2rad(deg)
        nid = g.add_vertex(cx + r * np.cos(rad), cy + r * np.sin(rad))
        g.add_edge(vid, nid, asgn)

    assert check_maekawa_at_vertex(vid, g) is False


# ---------------------------------------------------------------------------
# BLB tests
# ---------------------------------------------------------------------------

def test_blb_no_violations_equal_sectors():
    """Equal 90° sectors → no strict local minimum → BLB returns []."""
    g, vid = make_cross_graph()
    violations = check_blb_at_vertex(vid, g)
    assert violations == []


def test_blb_violation_detected():
    """
    Create a vertex with a strict local-minimum sector whose bounding edges
    share the same MV assignment → BLB violation.

    Use angles 0°, 10°, 180°, 270° so sector[0]=10° is the strict local min
    relative to sector[3] (90°) and sector[1] (170°). The two bounding edges
    are at 0° and 10°; assign both 'M' → violation.
    """
    g = CreaseGraph()
    cx, cy = 0.5, 0.5
    vid = g.add_vertex(cx, cy)

    r = 0.3
    # angles ascending (arctan2 order): 0°, 10°, 180°, 270° (= -90°)
    # sorted arctan2: -90°, 0°, 10°, 180°
    # sectors: 90°, 10°, 170°, 90°  (sum=360°)
    # sector at index 1 (between 0° and 10°) = 10° is strict local min (90 > 10 < 170)
    angles_deg = [0.0, 10.0, 180.0, 270.0]
    edge_ids = []
    for deg in angles_deg:
        rad = np.deg2rad(deg)
        nid = g.add_vertex(cx + r * np.cos(rad), cy + r * np.sin(rad))
        eid = g.add_edge(vid, nid, 'M')
        edge_ids.append(eid)

    violations = check_blb_at_vertex(vid, g)
    assert len(violations) > 0


def test_blb_no_violation_when_opposite_assignments():
    """
    Same geometry as above but with opposite assignments on the two edges
    bounding the small sector → no BLB violation.
    """
    g = CreaseGraph()
    cx, cy = 0.5, 0.5
    vid = g.add_vertex(cx, cy)

    r = 0.3
    angles_deg = [0.0, 10.0, 180.0, 270.0]
    # sorted arctan2: -90°(270°), 0°, 10°, 180°
    # small sector is between 0° and 10° (index 1 and 2 in sorted order)
    # assign them opposite assignments
    assignments_by_angle = {
        0.0: 'M',
        10.0: 'V',
        180.0: 'M',
        270.0: 'V',
    }
    for deg in angles_deg:
        rad = np.deg2rad(deg)
        nid = g.add_vertex(cx + r * np.cos(rad), cy + r * np.sin(rad))
        g.add_edge(vid, nid, assignments_by_angle[deg])

    violations = check_blb_at_vertex(vid, g)
    assert violations == []


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

def test_coverage_exact_match():
    """Add exact crease matching target → coverage = 1.0, economy = 1.0."""
    paper = PaperState()
    paper.add_crease([0.0, 0.5], [1.0, 0.5], 'M')

    target = [{'v1': (0.0, 0.5), 'v2': (1.0, 0.5), 'assignment': 'M'}]
    coverage, economy = geometric_crease_coverage(paper, target)
    assert coverage == pytest.approx(1.0)
    assert economy == pytest.approx(1.0)


def test_coverage_no_match():
    """No creases added → coverage = 0.0."""
    paper = PaperState()
    target = [{'v1': (0.0, 0.5), 'v2': (1.0, 0.5), 'assignment': 'M'}]
    coverage, economy = geometric_crease_coverage(paper, target)
    assert coverage == pytest.approx(0.0)


def test_coverage_excess_penalty():
    """
    Target has 1 crease. Add 3 non-intersecting creases, one matching target.
    coverage = 1.0, economy = 1 - 2/1 → clamped to 0.0 (economy < 1.0).
    Uses non-intersecting extras to avoid PaperState edge splitting the target crease.
    """
    paper = PaperState()
    paper.add_crease([0.0, 0.5], [1.0, 0.5], 'M')   # matches target (midpoint 0.5,0.5)
    paper.add_crease([0.0, 0.3], [0.5, 0.3], 'V')   # extra, no intersection
    paper.add_crease([0.0, 0.7], [0.5, 0.7], 'V')   # extra, no intersection

    target = [{'v1': (0.0, 0.5), 'v2': (1.0, 0.5), 'assignment': 'M'}]
    coverage, economy = geometric_crease_coverage(paper, target)
    assert coverage == pytest.approx(1.0)
    assert economy < 1.0


# ---------------------------------------------------------------------------
# check_all_vertices vacuous test
# ---------------------------------------------------------------------------

def test_check_all_vertices_vacuous():
    """Single horizontal crease → no interior vertices → all scores = 1.0."""
    paper = PaperState()
    paper.add_crease([0.0, 0.5], [1.0, 0.5], 'V')
    result = check_all_vertices(paper.graph)
    assert result['kawasaki'] == 1.0
    assert result['maekawa'] == 1.0
    assert result['blb'] == 1.0
    assert result['n_interior'] == 0
    assert result['per_vertex'] == []
