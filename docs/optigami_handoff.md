# OrigamiRL — OpenEnv Hackathon Handoff Document

## TL;DR

Build the **first multi-turn RL environment where an LLM learns to generate origami folding instructions**, verified by a computational origami simulator. Target the OpenEnv Hackathon (March 7-8, 2026, SF — $100K+ in prizes). Use OpenEnv spec + Unsloth GRPO for training. Dense verifiable rewards from origami geometry theorems (Kawasaki, Maekawa). No learned reward model needed.

---

## Hackathon Context

- **Event:** OpenEnv Hackathon SF, hosted by Cerebral Valley + Shack15 + Meta/PyTorch
- **Date:** March 7-8, 2026 (happening NOW)
- **Prize:** $100K+ cash
- **Teams:** Up to 4 people
- **Format:** Build RL environments, post-train a base model

### Judging Criteria

| Category | Weight | What Matters |
|----------|--------|-------------|
| Environment Innovation | 40% | Novel, creative, challenging. Does it meaningfully test agent behavior? |
| Storytelling | 30% | Clear problem explanation, engaging demo, easy to follow |
| Training Script Showing Improvement | 20% | Observable reward curves, before/after behavior |
| Reward and Training Pipeline Setup | 10% | Coherent reward logic, meaningful improvement in inference |

### Key Sponsors to Impress

- **Meta/PyTorch** — OpenEnv creators, want environments using their spec
- **Unsloth AI** — GRPO training infra, ART (Agent Reinforcement Trainer). USE THEIR TOOLS.
- **OpenPipe** — ART trainer (frontend/backend split for GRPO). Also use.
- **Patronus AI** — Building "generative simulators" (auto-scaling RL environments). They care about curriculum difficulty scaling and verifiable rewards.
- **Snorkel AI** — "2026 is the year of environments." They care about data quality and environment diversity.
- **Hugging Face** — OpenEnv Hub, want environments deployed there
- **Scale AI / Mercor** — Agent evaluation, structured task environments

---

## The Pitch (for judges)

> "Spatial reasoning is the next frontier for LLM training — NeurIPS 2025 papers like OrigamiSpace showed that even GPT-5 fails at multi-step origami reasoning. But those are benchmarks, not training environments. We built OrigamiRL: the first multi-turn RL environment where an LLM agent learns to fold paper by outputting instructions, receiving geometric feedback, and improving through GRPO. Our reward function is fully verifiable — fold validity is checked against computational origami axioms, not an LLM judge. We built it on OpenEnv + Unsloth with a natural curriculum from single folds to full cranes."

---

## Prior Work (What Exists, Where the Gaps Are)

### 1. OrigamiSpace (NeurIPS 2025 Spotlight)

- **Paper:** https://arxiv.org/abs/2511.18450
- **What it is:** Benchmark with 350 origami data instances (CP diagrams, folding processes, folded shapes). 4 evaluation tasks: Pattern Prediction, Multi-step Spatial Reasoning, Spatial Relationship Prediction, End-to-End CP Code Generation.
- **Their compiler:** Outputs detailed flattened diagrams with crease locations and stacking relationships, supports interactive simulation with MLLMs, provides comprehensive error feedback. Checks: syntax validity, geometric foldability, no self-intersections, Kawasaki's theorem, Maekawa's theorem.
- **Their reward metrics for code gen:** Hausdorff distance (shape similarity), dihedral angle distribution, bounding box aspect ratios, constraint satisfaction.
- **Difficulty levels:** Easy (3-9 steps), Medium (10-19 steps), Hard (20-30 steps)
- **Gap:** Single-turn only (LLM generates complete CP code in one shot). They mention RL exploration but it's not the focus. No multi-turn sequential folding.

### 2. GamiBench (Dec 2025)

- **Paper:** https://arxiv.org/abs/2512.22207
- **What it is:** 186 regular + 186 impossible 2D crease patterns with 3D folded shapes from 6 viewpoints. 3 VQA tasks.
- **Gap:** Evaluation-only, no training. Tests single-step spatial understanding.

### 3. SpatialThinker (NeurIPS 2025)

- **Paper:** https://arxiv.org/abs/2511.07403
- **What it is:** 3D-aware MLLM trained with RL using dense spatial rewards. Constructs scene graphs. Multi-objective reward with lexicographic gating.
- **Key architecture to steal:** Dense reward design with lexicographic ordering — format → count → accuracy → spatial. Nearly doubled RL training gains vs sparse rewards. Only needed 7K training samples with GRPO.
- **Gap:** Static scene understanding (objects on a table), not sequential physical transformations.

### 4. rigid-origami Gym (IJCAI 2023)

- **Repo:** https://github.com/belalugaX/rigid-origami
- **Paper:** "Automating Rigid Origami Design" (https://arxiv.org/abs/2211.13219)
- **What it is:** Gym environment where agent constructs crease pattern graphs on a board. Sparse rewards. Foldability validated by triangle intersection tests + kinematic rigidity model. Game terminates on non-foldable states.
- **Gap:** Classical RL agents (discrete grid actions), NOT LLMs generating text. Rigid-origami tessellations only, not traditional origami. No natural language.

### 5. The Unique Gap We Fill

Nobody has built a model that reasons about **sequential 2D-to-3D geometric transformations with physical constraints** through **natural language instructions** in a **multi-turn RL training loop**. Origami is uniquely hard because it requires tracking how a flat sheet's topology changes through a sequence of folds — mental rotation, spatial visualization, and perspective-taking all at once.

---

## Environment Design

### Architecture Overview

```
+---------------------------------------------------+
|                   OpenEnv Server                   |
|  +-----------+  +----------+  +--------------+    |
|  |   State   |  |  Action  |  |   Reward     |    |
|  | (FOLD JSON|  | (LLM     |  | (Dense,      |    |
|  |  + target)|  |  output) |  |  verifiable) |    |
|  +-----------+  +----------+  +--------------+    |
|         |              |              |            |
|         v              v              v            |
|  +-----------------------------------------------+|
|  |         Paper Geometry Engine (Python)         ||
|  |  - Polygon state (Shapely)                    ||
|  |  - Fold operations (reflection across line)   ||
|  |  - Kawasaki/Maekawa constraint checks         ||
|  |  - Layer tracking                             ||
|  |  - FOLD format import/export                  ||
|  +-----------------------------------------------+|
|         |                                          |
|         v                                          |
|  +-----------------------------------------------+|
|  |         Three.js Visualizer (Demo only)        ||
|  |  - 3D fold animation                          ||
|  |  - Strain heatmap                             ||
|  |  - Instruction stream                         ||
|  +-----------------------------------------------+|
+---------------------------------------------------+
         |                    ^
         v                    |
+---------------------------------------------------+
|              Unsloth ART / GRPO Trainer            |
|  - Qwen2.5-VL-7B or Qwen3-4B base model          |
|  - LoRA/QLoRA for efficient training              |
|  - Multi-turn rollouts                            |
+---------------------------------------------------+
```

### OpenEnv Spec Compliance

Must implement these APIs:

```python
class OrigamiEnv:
    async def reset() -> Observation     # New episode: flat paper + target
    async def step(action) -> (Observation, reward, done, info)
    async def state() -> State           # Current paper geometry
    async def close()                    # Cleanup
```

OpenEnv repo: https://github.com/meta-pytorch/OpenEnv
Install: `pip install -e .` then `openenv init origami_env`

### State Space

```python
@dataclass
class OrigamiState:
    # Current paper geometry
    vertices: List[Tuple[float, float]]       # 2D vertex positions
    edges: List[Tuple[int, int]]              # Edge connectivity
    edges_assignment: List[str]               # 'M', 'V', 'B', 'F' (mountain/valley/boundary/flat)
    edges_foldAngle: List[float]              # -180 to 180 degrees
    faces: List[List[int]]                    # Face vertex indices
    layer_order: List[List[int]]              # Face stacking order

    # Episode context
    target_crease_pattern: dict               # Target FOLD JSON
    target_shape_image: Optional[np.ndarray]  # Target folded shape (for multimodal)
    instruction_history: List[str]            # Previous instructions
    step_count: int
    max_steps: int
```

This maps directly to the **FOLD format** (JSON-based, used by all origami software):

```json
{
  "vertices_coords": [[0,0], [1,0], [1,1], [0,1]],
  "edges_vertices": [[0,1], [1,2], [2,3], [3,0]],
  "edges_assignment": ["B", "B", "B", "B"],
  "edges_foldAngle": [0, 0, 0, 0],
  "faces_vertices": [[0, 1, 2, 3]]
}
```

FOLD spec: https://github.com/edemaine/fold
FOLD JS library: https://edemaine.github.io/fold/

### Action Space

The LLM outputs a JSON action:

```json
{
  "instruction": "Fold the top edge down to meet the bottom edge",
  "fold_line": [[0, 0.5], [1, 0.5]],
  "fold_angle": -180,
  "assignment": "V"
}
```

The `instruction` field is natural language (what we're training the model to produce well). The geometric fields are the verifiable representation. During training, the model outputs both; for the final demo, the NL instruction is the star.

Alternative simpler action (for early iterations):

```json
{
  "instruction": "Valley fold along the horizontal center line",
  "fold_type": "valley",
  "fold_axis": "horizontal",
  "fold_position": 0.5
}
```

### Reward Function — Dense, Multi-Objective, Lexicographically Gated

Inspired by SpatialThinker's design. Rewards are computed in order; later rewards only apply if earlier gates pass.

```python
def compute_reward(state, action, new_state, target) -> dict:
    rewards = {}

    # LEVEL 1: Format (gate for everything else)
    # Does the output parse into a valid fold operation?
    rewards['format'] = 1.0 if parseable(action) else 0.0
    if rewards['format'] == 0:
        return rewards  # Stop here

    # LEVEL 2: Local Geometric Validity
    # Kawasaki's theorem: sector angles at each interior vertex sum to 2pi
    kawasaki_valid = check_kawasaki(new_state)
    # Maekawa's theorem: |M - V| = 2 at each interior vertex
    maekawa_valid = check_maekawa(new_state)
    # No self-intersection
    no_intersection = check_no_self_intersection(new_state)
    rewards['validity'] = (kawasaki_valid + maekawa_valid + no_intersection) / 3.0
    if rewards['validity'] < 0.5:
        return rewards  # Stop here

    # LEVEL 3: Physical Feasibility
    # Can this fold actually be performed given layer stack?
    layer_consistent = check_layer_ordering(new_state)
    fold_achievable = check_fold_angle_feasible(new_state)
    rewards['feasibility'] = (layer_consistent + fold_achievable) / 2.0

    # LEVEL 4: Progress Toward Target (Dense)
    # Crease pattern graph similarity
    cp_similarity = crease_pattern_similarity(new_state, target)
    # Fold angle distribution match
    angle_similarity = fold_angle_distribution_match(new_state, target)
    # Bounding box aspect ratio match
    bbox_similarity = bounding_box_similarity(new_state, target)
    rewards['progress'] = 0.4 * cp_similarity + 0.4 * angle_similarity + 0.2 * bbox_similarity

    # LEVEL 5: Completion Bonus
    if shape_matches_target(new_state, target, tolerance=0.05):
        rewards['completion'] = 10.0

    # LEVEL 6: Efficiency
    rewards['efficiency'] = -0.01  # Small step penalty to encourage fewer folds

    # Total
    rewards['total'] = (
        0.1 * rewards['format'] +
        0.2 * rewards['validity'] +
        0.1 * rewards['feasibility'] +
        0.5 * rewards['progress'] +
        rewards.get('completion', 0) +
        rewards['efficiency']
    )
    return rewards
```

### Key Origami Theorems for Verification

These are the verifiable constraints — the "unit tests" of origami:

1. **Kawasaki's Theorem:** At any interior vertex of a flat-foldable crease pattern, the alternating sum of sector angles equals zero (equivalently, they sum to 2pi on each side). NECESSARY condition for flat-foldability.

2. **Maekawa's Theorem:** At any interior vertex, the number of mountain folds minus valley folds equals +/-2. |M - V| = 2.

3. **No self-intersection:** Faces cannot penetrate each other during folding.

4. **Euler's formula for planar graphs:** V - E + F = 2 (sanity check on graph structure).

5. **Huzita-Hatori axioms:** The 7 axioms defining all possible single-fold operations (point-to-point, point-to-line, line-to-line, etc.). These define the VALID action space.

### Curriculum Design

| Level | Folds | Examples | Complexity |
|-------|-------|----------|-----------|
| 1 | 1 | Valley fold in half, mountain fold corner | Single fold validity |
| 2 | 2-3 | Paper airplane nose, triangle fold | Sequential dependency |
| 3 | 4-6 | Simple boat, fortune teller | Multi-step with symmetry |
| 4 | 7-12 | Paper airplane (full), jumping frog | Longer horizon planning |
| 5 | 13-20 | Crane, lily | Complex spatial tracking |

For the hackathon, focus on Levels 1-3. Even showing reward improvement on Level 1-2 is a strong result.

---

## Core Implementation: Python Geometry Engine

This is the MOST IMPORTANT piece. Pure Python, no JS dependencies.

```python
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import split
from typing import List, Tuple, Dict
import json

class PaperState:
    """Represents the current state of the origami paper."""

    def __init__(self, size: float = 1.0):
        # Start with a unit square
        self.regions = [Polygon([(0,0), (size,0), (size,size), (0,size)])]
        self.fold_history = []
        self.crease_lines = []
        self.crease_assignments = []  # 'M' or 'V'
        self.crease_angles = []
        self.layer_order = [0]  # Stack order of regions

    def apply_fold(self, fold_line: LineString, angle: float, assignment: str) -> dict:
        """
        Apply a fold operation. Returns dict with validity info.
        fold_line: Shapely LineString defining the fold axis
        angle: fold angle in degrees (-180 to 180)
        assignment: 'M' (mountain) or 'V' (valley)
        """
        result = {'valid': True, 'errors': []}

        # 1. Split regions by fold line
        new_regions = []
        for region in self.regions:
            if fold_line.intersects(region):
                parts = split(region, fold_line)
                new_regions.extend(parts.geoms)
            else:
                new_regions.append(region)

        # 2. Determine which side folds (based on assignment)
        folding_side = []
        staying_side = []
        for region in new_regions:
            centroid = region.centroid
            side = self._point_side(centroid, fold_line)
            if side > 0:
                folding_side.append(region)
            else:
                staying_side.append(region)

        # 3. Reflect folding regions across fold line
        reflected = [self._reflect_polygon(r, fold_line) for r in folding_side]

        # 4. Update state
        self.regions = staying_side + reflected
        self.crease_lines.append(fold_line)
        self.crease_assignments.append(assignment)
        self.crease_angles.append(angle)
        self.fold_history.append({
            'line': list(fold_line.coords),
            'angle': angle,
            'assignment': assignment
        })

        # 5. Update layer order
        self._update_layer_order(staying_side, reflected)

        return result

    def _reflect_polygon(self, poly: Polygon, line: LineString) -> Polygon:
        """Reflect a polygon across a line."""
        coords = list(poly.exterior.coords)
        reflected_coords = [self._reflect_point(p, line) for p in coords]
        return Polygon(reflected_coords)

    def _reflect_point(self, point: tuple, line: LineString) -> tuple:
        """Reflect a point across a line."""
        p = np.array(point[:2])
        l1 = np.array(line.coords[0])
        l2 = np.array(line.coords[1])
        d = l2 - l1
        d = d / np.linalg.norm(d)
        # Reflection formula: p' = p - 2(p-l1).n * n where n is normal to line
        n = np.array([-d[1], d[0]])
        v = p - l1
        return tuple(p - 2 * np.dot(v, n) * n)

    def _point_side(self, point, line: LineString) -> float:
        """Returns positive if point is on left side of line, negative if right."""
        p = np.array([point.x, point.y])
        l1 = np.array(line.coords[0])
        l2 = np.array(line.coords[1])
        return float(np.cross(l2 - l1, p - l1))

    def _update_layer_order(self, staying, reflected):
        """Update the layer stacking order after a fold."""
        self.layer_order = list(range(len(staying))) + \
                          list(range(len(staying), len(staying) + len(reflected)))

    def to_fold_json(self) -> dict:
        """Export current state as FOLD format JSON."""
        vertices = set()
        for line in self.crease_lines:
            for coord in line.coords:
                vertices.add(tuple(round(c, 10) for c in coord))
        # Add boundary vertices
        for region in self.regions:
            for coord in region.exterior.coords:
                vertices.add(tuple(round(c, 10) for c in coord[:2]))

        vertices = sorted(list(vertices))
        vertex_map = {v: i for i, v in enumerate(vertices)}

        edge_set = set()
        edges_list = []
        assignments_list = []
        angles_list = []

        # Add crease edges
        for i, line in enumerate(self.crease_lines):
            c = [tuple(round(x, 10) for x in coord) for coord in line.coords]
            edge = tuple(sorted([vertex_map[c[0]], vertex_map[c[1]]]))
            if edge not in edge_set:
                edge_set.add(edge)
                edges_list.append(list(edge))
                assignments_list.append(self.crease_assignments[i])
                angles_list.append(self.crease_angles[i])

        return {
            'vertices_coords': [list(v) for v in vertices],
            'edges_vertices': edges_list,
            'edges_assignment': assignments_list,
            'edges_foldAngle': angles_list,
        }


class OrigamiVerifier:
    """Verifiable reward functions based on origami theorems."""

    @staticmethod
    def check_kawasaki(state: PaperState) -> bool:
        """Kawasaki's theorem: alternating sum of angles at each interior vertex = 0."""
        fold_json = state.to_fold_json()
        vertices = fold_json['vertices_coords']
        edges = fold_json['edges_vertices']

        for v_idx in range(len(vertices)):
            v = vertices[v_idx]
            incident_edges = [e for e in edges if v_idx in e]
            if len(incident_edges) < 4:
                continue  # Need degree-4+ for Kawasaki

            # Calculate sector angles
            angles = []
            for e in incident_edges:
                other = e[1] if e[0] == v_idx else e[0]
                other_v = vertices[other]
                angle = np.arctan2(other_v[1] - v[1], other_v[0] - v[0])
                angles.append(angle)

            angles.sort()
            sector_angles = []
            for i in range(len(angles) - 1):
                sector_angles.append(angles[i+1] - angles[i])
            sector_angles.append(2*np.pi - (angles[-1] - angles[0]))

            # Kawasaki: alternating sum should be ~0
            if len(sector_angles) >= 4:
                alt_sum = sum(sector_angles[::2]) - sum(sector_angles[1::2])
                if abs(alt_sum) > 0.01:
                    return False
        return True

    @staticmethod
    def check_maekawa(state: PaperState) -> bool:
        """Maekawa's theorem: |M - V| = 2 at each interior vertex."""
        fold_json = state.to_fold_json()
        vertices = fold_json['vertices_coords']
        edges = fold_json['edges_vertices']
        assignments = fold_json['edges_assignment']

        for v_idx in range(len(vertices)):
            incident = [(i, e) for i, e in enumerate(edges) if v_idx in e]
            m_count = sum(1 for i, _ in incident if i < len(assignments) and assignments[i] == 'M')
            v_count = sum(1 for i, _ in incident if i < len(assignments) and assignments[i] == 'V')

            if m_count + v_count >= 4:  # Interior vertex with folds
                if abs(m_count - v_count) != 2:
                    return False
        return True

    @staticmethod
    def crease_pattern_similarity(state: PaperState, target_fold_json: dict) -> float:
        """Compare current crease pattern to target. Returns 0-1 similarity."""
        current = state.to_fold_json()

        n_current = len(current.get('edges_vertices', []))
        n_target = len(target_fold_json.get('edges_vertices', []))

        if n_target == 0:
            return 1.0 if n_current == 0 else 0.0

        edge_count_sim = 1.0 - abs(n_current - n_target) / max(n_target, 1)
        edge_count_sim = max(0, edge_count_sim)

        current_assignments = current.get('edges_assignment', [])
        target_assignments = target_fold_json.get('edges_assignment', [])

        c_m = current_assignments.count('M')
        c_v = current_assignments.count('V')
        t_m = target_assignments.count('M')
        t_v = target_assignments.count('V')

        total = max(t_m + t_v, 1)
        assign_sim = 1.0 - (abs(c_m - t_m) + abs(c_v - t_v)) / (2 * total)
        assign_sim = max(0, assign_sim)

        return 0.5 * edge_count_sim + 0.5 * assign_sim
```

---

## OpenEnv Environment Wrapper

```python
# origami_env/server.py
from openenv.core import Environment
from paper_engine import PaperState, OrigamiVerifier
from shapely.geometry import LineString
import json

class OrigamiEnvironment(Environment):

    def __init__(self, targets_dir="targets/", max_steps=20):
        self.targets_dir = targets_dir
        self.max_steps = max_steps
        self.paper = None
        self.target = None
        self.step_count = 0

    async def reset(self, target_id=None):
        self.paper = PaperState(size=1.0)
        self.target = self._load_target(target_id)
        self.step_count = 0
        return self._get_observation()

    async def step(self, action):
        self.step_count += 1

        # Parse action
        try:
            fold_line = LineString(action['fold_line'])
            angle = action['fold_angle']
            assignment = action['assignment']
        except (KeyError, Exception):
            reward = {'format': 0, 'total': -0.1}
            return self._get_observation(), reward, False, {'error': 'parse_failed'}

        # Apply fold
        result = self.paper.apply_fold(fold_line, angle, assignment)

        # Compute rewards
        reward = self._compute_reward(result)

        # Check termination
        done = (
            self.step_count >= self.max_steps or
            reward.get('completion', 0) > 0
        )

        return self._get_observation(), reward, done, {}

    async def state(self):
        return {
            'paper': self.paper.to_fold_json(),
            'target': self.target,
            'step': self.step_count,
            'fold_history': self.paper.fold_history
        }

    def _compute_reward(self, fold_result):
        rewards = {}
        rewards['format'] = 1.0

        kawasaki = OrigamiVerifier.check_kawasaki(self.paper)
        maekawa = OrigamiVerifier.check_maekawa(self.paper)
        rewards['validity'] = (float(kawasaki) + float(maekawa)) / 2.0

        rewards['progress'] = OrigamiVerifier.crease_pattern_similarity(
            self.paper, self.target
        )

        if rewards['progress'] > 0.95:
            rewards['completion'] = 10.0

        rewards['efficiency'] = -0.01

        rewards['total'] = (
            0.1 * rewards['format'] +
            0.2 * rewards['validity'] +
            0.6 * rewards['progress'] +
            rewards.get('completion', 0) +
            rewards['efficiency']
        )
        return rewards

    def _get_observation(self):
        return {
            'paper_state': self.paper.to_fold_json(),
            'target': self.target,
            'step': self.step_count,
            'instruction_history': [str(f['line']) for f in self.paper.fold_history]
        }

    def _load_target(self, target_id):
        if target_id:
            with open(f"{self.targets_dir}/{target_id}.fold") as f:
                return json.load(f)
        # Default: simple valley fold in half
        return {
            'vertices_coords': [[0,0], [1,0], [1,1], [0,1], [0,0.5], [1,0.5]],
            'edges_vertices': [[0,1], [1,2], [2,3], [3,0], [4,5]],
            'edges_assignment': ['B', 'B', 'B', 'B', 'V'],
            'edges_foldAngle': [0, 0, 0, 0, -180],
        }
```

---

## Training Script (Unsloth GRPO)

```python
# train.py
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# Reward function
def origami_reward(completions, prompts):
    """Compute rewards for a batch of completions."""
    rewards = []
    for completion in completions:
        try:
            action = parse_fold_action(completion)
            paper = PaperState()
            result = paper.apply_fold(action['fold_line'], action['angle'], action['assignment'])
            r = compute_reward(paper, target)
            rewards.append(r['total'])
        except Exception:
            rewards.append(-0.1)
    return rewards

# GRPO Config
config = GRPOConfig(
    output_dir="origami-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_completion_length=512,
    num_generations=8,
    temperature=1.0,
    logging_steps=1,
)

dataset = load_origami_prompts()

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    reward_funcs=[origami_reward],
    tokenizer=tokenizer,
)

trainer.train()
```

---

## Visualization (Demo Only — Not in Training Loop)

### Options

1. **Origami Simulator** — https://github.com/amandaghassaei/OrigamiSimulator — Three.js, accepts FOLD files, shows folding animation with strain visualization
2. **PackCAD** — https://packcad.com/ — Web-based, SVG crease patterns, rigid folding simulation
3. **Custom Three.js** — Simpler but more control

### Demo UI Layout

```
+----------------------+----------------------+
|   Instruction Stream |   3D Fold Viewer     |
|                      |                      |
| Step 1: Valley fold  |   [Three.js canvas]  |
| along center [OK]    |                      |
|                      |   Paper animating    |
| Step 2: Fold top     |   fold by fold       |
| corners to center    |                      |
|                      |                      |
+----------------------+----------------------+
|   Reward Dashboard                          |
|   Format:   ========== 1.0                  |
|   Validity: ========.. 0.8                  |
|   Progress: ======.... 0.6                  |
|   Total:    =======... 0.72                 |
|                                              |
|   [Reward curve over training steps]         |
+----------------------------------------------+
```

---

## Key Libraries and Resources

| Tool | Purpose | Link |
|------|---------|------|
| OpenEnv | Environment framework | https://github.com/meta-pytorch/OpenEnv |
| Unsloth | GRPO training | https://github.com/unslothai/unsloth |
| OpenPipe ART | Multi-turn RL trainer | https://github.com/OpenPipe/ART |
| FOLD format | Origami data structure | https://github.com/edemaine/fold |
| Rabbit Ear | JS origami library | https://github.com/rabbit-ear/rabbit-ear |
| Origami Simulator | 3D visualization | https://github.com/amandaghassaei/OrigamiSimulator |
| PackCAD | Folding simulation | https://packcad.com/ |
| Shapely | Python geometry | pip install shapely |
| rigid-origami gym | Reference gym env | https://github.com/belalugaX/rigid-origami |

### Papers to Cite

- OrigamiSpace: https://arxiv.org/abs/2511.18450
- GamiBench: https://arxiv.org/abs/2512.22207
- SpatialThinker: https://arxiv.org/abs/2511.07403
- Automating Rigid Origami Design: https://arxiv.org/abs/2211.13219
- FOLD format spec: https://github.com/edemaine/fold/blob/main/doc/spec.md

---

## Priority Build Order

1. **Python geometry engine** — PaperState class with fold operations and FOLD export
2. **Verifier functions** — Kawasaki, Maekawa, similarity metrics
3. **OpenEnv wrapper** — step/reset/state API
4. **Simple targets** — Hand-create 5-10 Level 1-2 targets as .fold files
5. **Training script** — Wire up Unsloth GRPO with reward function
6. **Run training** — Even on small model, get reward curves
7. **Three.js visualizer** — For demo only, not in training loop
8. **Before/after demo** — Show base model vs trained model outputs
9. **Polish presentation narrative**

---

## Narrative for Judges

**The story arc:**

1. "LLMs are great at text but terrible at spatial reasoning"
2. "Origami is the perfect testbed — it's sequential, physical, and verifiable"
3. "NeurIPS 2025 showed even GPT-5 fails at origami benchmarks, but nobody built a TRAINING environment"
4. "We built OrigamiRL — the first multi-turn RL environment for origami instruction generation"
5. "Our rewards come from math theorems, not vibes — Kawasaki's theorem is our unit test"
6. "Watch the model go from generating paper-tearing nonsense to valid fold sequences"
7. "This generalizes to any domain where LLMs need to output structured physical instructions"
