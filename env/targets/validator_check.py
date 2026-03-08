import json, sys, os

targets_dir = "/Users/ianalin/Desktop/optigami/env/targets"
for fname in os.listdir(targets_dir):
    if not fname.endswith(".fold"):
        continue
    with open(os.path.join(targets_dir, fname)) as f:
        d = json.load(f)
    n_v = len(d["vertices_coords"])
    n_e = len(d["edges_vertices"])
    assert len(d["edges_assignment"]) == n_e, f"{fname}: assignment length mismatch"
    assert len(d["edges_foldAngle"]) == n_e, f"{fname}: foldAngle length mismatch"
    for e in d["edges_vertices"]:
        assert e[0] < n_v and e[1] < n_v, f"{fname}: edge references invalid vertex"
    for face in d["faces_vertices"]:
        for vi in face:
            assert vi < n_v, f"{fname}: face references invalid vertex"
    creases = [i for i,a in enumerate(d["edges_assignment"]) if a in ('M','V')]
    print(f"{fname}: {n_v} vertices, {n_e} edges, {len(creases)} creases, level={d.get('level','?')} OK")
