# Optigami — Implementation Plan

> Derived from handoff doc critique, origami math/physics research, and plan review.
> Last updated: 2026-03-07

---

## Resolved Architectural Decisions

### 1. Code-as-policy for training, step-level for demo

GRPO samples N completions for a fixed prompt, evaluates each independently, computes group advantages. That maps cleanly to **code-as-policy**: the model outputs a complete fold sequence as a JSON list, the environment executes it sequentially, terminal reward is computed once.

Step-level breaks GRPO's assumption: at step k, the prompt is conditioned on prior steps which differ across rollouts, so you're no longer comparing N completions to the same situation.

**Resolution:** Training is code-as-policy (full sequence → single reward). Demo is step-by-step (one fold at a time with live feedback). Same environment, different prompt wrapper. Same model at inference — you just prompt it one fold at a time for the demo.

### 2. 2D crease pattern is Phase 1, engineering metrics are Phase 2

**Phase 1 (hackathon MVP):** Build the crease pattern graph, check local foldability, use geometric coverage as progress proxy. Self-contained, can show reward improvement.

**Phase 2 (if time permits):** Apply fold angles to compute the 3D folded state, compute deployment ratio and bounding box. These become the primary reward, with crease coverage as scaffolding. This is where the "model discovers Miura-ori" story lives.

If the deadline forces a cut, Phase 1 ships and Phase 2 is explicitly called out as the next step.

### 3. Scope to local flat-foldability (NP-hardness acknowledged)

Global flat-foldability (layer ordering) is NP-complete (Bern-Hayes 1996). We target **local flat-foldability** at each vertex, which is polynomial. This is a feature, not a limitation — the pitch: "our rewards check the conditions every origami designer verifies. Global layer ordering is provably NP-complete."

### 4. Symmetry masking is a noted risk

For Level 1-2 targets the anchor set is small (≤8 points), manageable. For Level 3+, intersection vertices accumulate to 15-20+ points, giving O(300+) candidate fold lines. The unit square has dihedral-4 symmetry (4 rotations + 4 reflections). For Level 3+, if training shows no convergence after 500 steps, add explicit symmetry-based action pruning.

---

## File Structure

```
optigami/
  env/
    __init__.py
    graph.py            # CreaseGraph: vertices, edges, cyclic ordering
    paper_state.py      # PaperState using CreaseGraph, add_crease
    verifier.py         # Kawasaki, Maekawa, BLB, coverage, deployment ratio
    rewards.py          # compute_reward (Phase 1 + Phase 2 extension)
    environment.py      # OpenEnv wrapper, code-as-policy and step modes
    prompts.py          # LLM observation formatting
    fold_engine.py      # Phase 2: apply fold angles, compute 3D bounding box
    targets/
      validator.py      # crimp-check all .fold files before training
      half_horizontal.fold
      half_vertical.fold
      diagonal.fold
      cross_fold.fold
      x_fold.fold
      pinwheel_base.fold
      preliminary_base.fold
      fish_base.fold
  train.py
  requirements.txt
  src/                  # React demo visualizer (existing)
  plans/
    implementation_plan.md
```

---

## Phase 1: CreaseGraph (`env/graph.py`)

Everything builds on this. Get it right first.

**Data:**
- `vertices`: `dict[vertex_id → (x, y)]`
- `edges`: `dict[edge_id → (v1, v2, assignment)]` where assignment ∈ `{M, V, B}`
- `vertex_edges`: `dict[vertex_id → [edge_ids]]`

**Key operations:**
- `add_vertex(x, y, tol=1e-9)` — deduplicated by proximity
- `add_edge(v1, v2, assignment)` — no duplicates
- `get_cyclic_edges(vertex_id)` — incident edge IDs sorted by angle of the other endpoint around the vertex (the cyclic order Kawasaki requires)
- `interior_vertices()` — vertices not on the unit square boundary
- `split_edge(edge_id, new_vertex_id)` — splits an edge at a vertex, used when a new crease intersects an existing one

**`add_crease(p1, p2, assignment)` in `PaperState`:**
1. Validate both endpoints are in the anchor set (within tolerance)
2. Find all intersections with existing edges
3. Add intersection vertices and split existing edges at them
4. Add the new crease edge(s) (possibly split by intersections)
5. Return `{valid, anchored, new_vertices, errors}`

**Anchor point set** (grows as creases are added):
- Boundary corners: `(0,0), (1,0), (1,1), (0,1)`
- Boundary midpoints of any existing boundary edge
- All crease-crease intersection vertices
- Midpoints of existing crease edges

---

## Phase 2: Verifiers (`env/verifier.py`)

### Even-degree fast-fail

```python
def has_even_degree(vertex_id, graph) -> bool:
    return len(graph.get_cyclic_edges(vertex_id)) % 2 == 0
```

Runs before Kawasaki. Odd-degree interior vertices are impossible — short-circuit immediately.

### Kawasaki-Justin

Sector angles must be computed in **cyclic angular order** around each vertex — not by magnitude, not arbitrarily. The handoff's sorted-angle approach was wrong; cyclic order is recovered by sorting incident edge directions by `arctan2`.

```python
def check_kawasaki_at_vertex(vertex_id, graph) -> tuple[bool, float]:
    cyclic_edges = graph.get_cyclic_edges(vertex_id)  # sorted by angle
    n = len(cyclic_edges)
    if n % 2 != 0:
        return False, float('inf')
    if n < 4:
        return True, 0.0  # boundary vertex, not an interior fold vertex

    v = graph.vertices[vertex_id]
    angles = []
    for eid in cyclic_edges:
        v1, v2, _ = graph.edges[eid]
        other = v2 if v1 == vertex_id else v1
        other_pos = graph.vertices[other]
        angles.append(np.arctan2(other_pos[1] - v[1], other_pos[0] - v[0]))
    # angles is already in cyclic order (cyclic_edges sorted by angle)

    sectors = []
    for i in range(n):
        diff = angles[(i+1) % n] - angles[i]
        if diff < 0:
            diff += 2 * np.pi
        sectors.append(diff)

    alt_sum = sum(s * ((-1)**i) for i, s in enumerate(sectors))
    return abs(alt_sum) < 1e-9, abs(alt_sum)
```

### Maekawa-Justin

Boundary edges (`B`) must not be counted — only fold edges (`M`, `V`). The handoff counted boundary edges, which breaks Maekawa for any crease touching the paper edge.

```python
def check_maekawa_at_vertex(vertex_id, graph) -> bool:
    fold_edges = [eid for eid in graph.vertex_edges[vertex_id]
                  if graph.edges[eid][2] in ('M', 'V')]
    if len(fold_edges) < 4:
        return True  # not an interior fold vertex yet
    M = sum(1 for eid in fold_edges if graph.edges[eid][2] == 'M')
    V = len(fold_edges) - M
    return abs(M - V) == 2
```

### Big-Little-Big (BLB)

At any interior vertex, if a sector angle is a strict local minimum, the two crease lines bounding that sector must have **opposite MV parity**. This is the key pruning rule between Maekawa and layer-ordering — a pattern can satisfy Maekawa while violating BLB, meaning no valid layer ordering exists.

```python
def check_blb_at_vertex(vertex_id, graph) -> list[tuple]:
    """Returns list of (edge_a, edge_b) pairs where BLB is violated."""
    cyclic_edges = graph.get_cyclic_edges(vertex_id)
    n = len(cyclic_edges)
    if n < 4:
        return []
    sectors = _compute_sectors(vertex_id, cyclic_edges, graph)
    violations = []
    for i in range(n):
        prev_s = sectors[(i-1) % n]
        next_s = sectors[(i+1) % n]
        if sectors[i] < prev_s and sectors[i] < next_s:  # strict local min
            left_eid = cyclic_edges[i]
            right_eid = cyclic_edges[(i+1) % n]
            a_left = graph.edges[left_eid][2]
            a_right = graph.edges[right_eid][2]
            if a_left in ('M', 'V') and a_right in ('M', 'V') and a_left == a_right:
                violations.append((left_eid, right_eid))
    return violations
```

### Geometric Coverage (with excess penalty)

One-sided coverage alone rewards placing target creases but doesn't penalize surplus creases. Both are returned separately so the reward function can weight them independently.

```python
def geometric_coverage(state, target_edges, tol_pos=0.05, tol_angle=5.0) -> tuple[float, float]:
    """
    Returns (coverage, economy).
    coverage: fraction of target creases matched by current creases [0, 1]
    economy:  penalty for excess creases [0, 1], 1.0 = no excess
    """
    matched = 0
    for t_edge in target_edges:
        for c_edge in state.crease_edges():
            if _edges_match(t_edge, c_edge, tol_pos, tol_angle):
                matched += 1
                break
    n_target = max(len(target_edges), 1)
    n_current = len(state.crease_edges())
    coverage = matched / n_target
    economy = max(0.0, 1.0 - max(0, n_current - n_target) / n_target)
    return coverage, economy
```

---

## Phase 3: Reward Function (`env/rewards.py`)

### Phase 1 reward

Single consistent definition. `progress` carries 45% — it's the only signal with real geometric content at every step. Validity signals split 20% total. Economy penalizes excess creases.

```python
def compute_reward_phase1(state, action_result, target) -> dict:
    r = {}

    r['format'] = 1.0 if action_result['valid'] else 0.0
    if not r['format']:
        return {**r, 'total': -0.1}

    r['anchored'] = 1.0 if action_result['anchored'] else 0.3

    interior = state.graph.interior_vertices()
    n = max(len(interior), 1)

    kaw = [check_kawasaki_at_vertex(v, state.graph) for v in interior]
    mae = [check_maekawa_at_vertex(v, state.graph) for v in interior]
    blb = [check_blb_at_vertex(v, state.graph) for v in interior]

    r['kawasaki'] = sum(ok for ok, _ in kaw) / n
    r['maekawa']  = sum(mae) / n
    r['blb']      = 1.0 - sum(len(v) > 0 for v in blb) / n

    coverage, economy = geometric_coverage(state, target['edges'])
    r['progress'] = coverage
    r['economy']  = economy

    all_valid = (r['kawasaki'] == 1.0 and r['maekawa'] == 1.0 and r['blb'] == 1.0)
    r['completion'] = 10.0 if (r['progress'] > 0.9 and all_valid) else 0.0
    r['efficiency'] = -0.01

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
```

### Phase 2 reward extension

When `fold_engine.py` is available, replace `progress` and `economy` with engineering metrics. No pre-specified target pattern required — the model optimizes objectives directly and can discover that Miura-ori is optimal.

```python
def compute_reward_phase2(state, action_result, folded_state) -> dict:
    # ... same gates as phase 1 ...

    r['deployment_ratio'] = compute_deployment_ratio(folded_state)
    # = unfolded_area / folded_bounding_box_area

    r['bbox_compactness'] = 1.0 - (folded_bbox_area / unfolded_area)
    # higher = more compact fold

    r['total'] = (
        0.05 * r['anchored'] +
        0.08 * r['kawasaki'] +
        0.07 * r['maekawa'] +
        0.05 * r['blb'] +
        0.30 * r['deployment_ratio'] +
        0.20 * r['bbox_compactness'] +
        0.05 * r['economy'] +
        r['completion'] +
        r['efficiency']
    )
    return r
```

---

## Phase 4: Prompts (`env/prompts.py`)

### Code-as-policy prompt (training mode)

```
You are an origami designer. Generate a complete fold sequence for a unit square [0,1]x[0,1].

TARGET CREASE PATTERN:
  Valley fold: (0.0, 0.5) -> (1.0, 0.5)
  Mountain fold: (0.5, 0.0) -> (0.5, 1.0)

RULES (your sequence must satisfy at every interior vertex):
  - Kawasaki: alternating sector angles sum equally (each half = 180 degrees)
  - Maekawa: |mountain_count - valley_count| = 2
  - Big-Little-Big: folds bounding the smallest sector must have opposite types

ANCHOR POINTS (valid fold endpoints):
  Corners:   (0,0)  (1,0)  (1,1)  (0,1)
  Midpoints: (0.5,0)  (1,0.5)  (0.5,1)  (0,0.5)
  Note: the square has 4-fold dihedral symmetry — symmetric fold sequences are equivalent.

Output a JSON list of fold operations in order. Both endpoints must be anchor points.

<folds>
[
  {"instruction": "...", "from": [x1, y1], "to": [x2, y2], "assignment": "M"|"V"},
  ...
]
</folds>
```

### Step-level prompt (demo mode)

Same information, but shows only the current step's observation with prior fold history and last-step reward appended. Same model, different prompt wrapper.

```
... [same header] ...

CURRENT STATE (step 2 of 8):
  Creases placed:
    1. Mountain fold: (0.5, 0.0) -> (0.5, 1.0)

AVAILABLE ANCHOR POINTS:
  Corners:       (0.0,0.0)  (1.0,0.0)  (1.0,1.0)  (0.0,1.0)
  Edge midpoints:(0.5,0.0)  (1.0,0.5)  (0.5,1.0)  (0.0,0.5)
  Intersections: (0.5,0.5)

LAST REWARD: format=1.0  kawasaki=1.0  maekawa=1.0  blb=1.0  progress=0.32  total=0.33

Add the next crease. Output JSON only:
{"instruction": "...", "from": [x1, y1], "to": [x2, y2], "assignment": "M"|"V"}
```

---

## Phase 5: Target Files + Validator (`env/targets/`)

Targets are hand-authored `.fold` JSON. Before any target enters training, `validator.py` runs:

1. Parse FOLD JSON, reconstruct the CreaseGraph
2. For each interior vertex: even-degree → Kawasaki → Maekawa → BLB
3. Enumerate at least one valid MV assignment via the crimp algorithm
4. Fail loudly with vertex + violation details if any check fails

**Target set:**

| File | Creases | Level | Interior vertices |
|------|---------|-------|-------------------|
| `half_horizontal.fold` | 1 | 1 | 0 |
| `half_vertical.fold` | 1 | 1 | 0 |
| `diagonal.fold` | 1 | 1 | 0 |
| `cross_fold.fold` | 2 | 2 | 1 (degree 4) |
| `x_fold.fold` | 2 | 2 | 1 (degree 4) |
| `pinwheel_base.fold` | 4 | 2 | 4 |
| `preliminary_base.fold` | 4 | 3 | 4 |
| `fish_base.fold` | 6 | 3 | 6 |

Level 1 targets have zero interior vertices — Kawasaki/Maekawa are vacuously satisfied, the only reward signal is `progress`. The model learns to place geometrically correct folds before worrying about vertex constraints.

---

## Phase 6: OpenEnv Wrapper (`env/environment.py`)

Both modes supported. The `info` dict explicitly labels what is and isn't checked.

```python
class OrigamiEnvironment(Environment):

    async def step(self, action):
        if isinstance(action, list):
            return self._execute_sequence(action)  # code-as-policy
        else:
            return self._execute_single(action)    # step mode

    def _execute_sequence(self, folds):
        for fold in folds:
            result = self.paper.add_crease(
                fold['from'], fold['to'], fold['assignment']
            )
            if not result['valid']:
                break  # partial credit: reward up to failure point
        reward = compute_reward_phase1(self.paper, result, self.target)
        return self._get_observation(), reward, True, self._info()

    def _info(self):
        interior = self.paper.graph.interior_vertices()
        return {
            'local_foldability': all(
                check_kawasaki_at_vertex(v, self.paper.graph)[0] and
                check_maekawa_at_vertex(v, self.paper.graph)
                for v in interior
            ),
            'blb_satisfied': all(
                len(check_blb_at_vertex(v, self.paper.graph)) == 0
                for v in interior
            ),
            'global_foldability': 'not_checked',  # NP-complete (Bern-Hayes 1996)
            'n_interior_vertices': len(interior),
        }
```

---

## Phase 7: Training Script (`train.py`)

Code-as-policy GRPO. Each completion is a complete fold sequence. N=8 completions per prompt evaluated in parallel, each with its own fresh `PaperState`. Terminal reward only.

```python
def origami_reward_fn(completions, prompts, targets):
    rewards = []
    for completion, target in zip(completions, targets):
        try:
            folds = parse_fold_list(completion)  # extract JSON from <folds> tags
            paper = PaperState()
            for fold in folds:
                paper.add_crease(fold['from'], fold['to'], fold['assignment'])
            r = compute_reward_phase1(paper, {'valid': True, 'anchored': True}, target)
            rewards.append(r['total'])
        except Exception:
            rewards.append(-0.1)
    return rewards
```

Log all reward components separately (kawasaki, maekawa, blb, progress, economy) — the decomposed curves are the demo artifact showing the model learning to satisfy geometric constraints.

---

## Phase 8: Fold Engine / Phase 2 (`env/fold_engine.py`)

For flat-folded patterns (all creases at 180°), the folded bounding box is computable from crease pattern + simplified layer assignment. For Level 1-3 targets the layer assignment is tractable (polynomial for single-vertex, and our simple patterns have at most a few interior vertices).

Apply fold angles via reflection transforms, project to get 2D bounding box of the folded state, compute:

```
deployment_ratio = 1.0 / (folded_bbox_area / unfolded_area)
```

Higher = more compact = better engineering. With this signal the model can discover optimal fold patterns (Miura-ori, accordion folds) without a pre-specified target.

---

## Build Order

```
[ ] 1.  requirements.txt (shapely, numpy, pytest)
[ ] 2.  env/graph.py — CreaseGraph with cyclic ordering, split_edge
[ ] 3.  Unit test: two crossing creases -> 1 interior vertex of degree 4, correct cyclic order
[ ] 4.  env/paper_state.py — PaperState.add_crease with intersection handling
[ ] 5.  env/verifier.py — even-degree, Kawasaki, Maekawa, BLB, geometric_coverage
[ ] 6.  Unit test: degree-4 vertex with known valid/invalid angles -> Kawasaki pass/fail
[ ] 7.  Unit test: single crease -> zero interior vertices -> verifiers return defaults (True)
[ ] 8.  Unit test: excess crease penalty activates correctly
[ ] 9.  targets/validator.py — crimp-check routine
[ ] 10. env/targets/*.fold — 4 Level 1 + 4 Level 2 targets, all passing validator
[ ] 11. env/rewards.py — Phase 1 compute_reward
[ ] 12. env/prompts.py — code-as-policy prompt + step-level prompt
[ ] 13. env/environment.py — both sequence and step modes + info dict
[ ] 14. Integration test: known valid sequence on half_horizontal, reward >= 0.9
[ ] 15. Integration test: invalid MV assignment on cross_fold, BLB fires
[ ] 16. train.py — GRPO with code-as-policy reward fn
[ ] 17. First training run on Level 1 targets, log all reward components to W&B
[ ] 18. env/fold_engine.py — Phase 2: fold angles -> 3D state -> deployment ratio
[ ] 19. Visualizer (React): render crease graph from FOLD JSON, animate fold history
```

Steps 2-3 and 5-8 are highest risk. Get the graph data structure and cyclic Kawasaki check correct before building anything on top of them. Steps 14-15 are the checkpoint before touching the training script.

---

## Key Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cyclic sector angle computation incorrect | High | Explicit unit tests with known valid/invalid patterns |
| Level 3+ action space too large to learn | Medium | Dihedral symmetry hints in prompt; hard masking if no convergence after 500 steps |
| GRPO reward signal too sparse (no interior vertices on Level 1) | Medium | Level 1 reward is purely `progress`; works without vertex constraints |
| fold_engine Phase 2 infeasible in hackathon time | Medium | Phase 1 ships independently; Phase 2 is an extension |
| Layer ordering required for deployment ratio on complex patterns | Low | Level 1-3 patterns are tractable; flag NP-hardness in info dict |
