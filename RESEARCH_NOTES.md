# Optigami Research Notes

Comprehensive notes on all sources, tools, and architecture for the Optigami project.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Paper: OrigamiSpace (2511.18450)](#2-paper-origamispace-251118450)
3. [Paper: SpatialThinker (2511.07403)](#3-paper-spatialthinker-251107403)
4. [Paper: Automating Rigid Origami Design (2211.13219)](#4-paper-automating-rigid-origami-design-221113219)
5. [Tool: FOLD Format (edemaine/fold)](#5-tool-fold-format)
6. [Tool: Origami Simulator](#6-tool-origami-simulator)
7. [Tool: GamiBench](#7-tool-gamibench)
8. [Tool: SpatialThinker Codebase](#8-tool-spatialthinker-codebase)
9. [Tool: Trackio](#9-tool-trackio)
10. [Tool: Unsloth + GRPO Training](#10-tool-unsloth--grpo-training)
11. [Unsloth ART / GRPO Trainer Plan](#11-unsloth-art--grpo-trainer-plan)
12. [Current Project State](#12-current-project-state)

---

## 1. Project Architecture Overview

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

**Three major components:**
1. **OpenEnv Server** - RL environment serving state/action/reward for origami folding
2. **Paper Geometry Engine** - Python-based origami math (Shapely polygons, fold reflections, constraint checking)
3. **Unsloth ART / GRPO Trainer** - RL fine-tuning of vision-language models for origami reasoning

**Current focus:** Unsloth ART / GRPO Trainer

---

## 2. Paper: OrigamiSpace (2511.18450)

**Title:** ORIGAMISPACE: Benchmarking Multimodal LLMs in Multi-Step Spatial Reasoning with Mathematical Constraints
**Authors:** Rui Xu, Dakuan Lu, Zicheng Zhao, Xiaoyu Tan, Xintao Wang, Siyu Yuan, Jiangjie Chen, Yinghui Xu
**Date:** November 23, 2025
**Venue:** arXiv (cs.AI)

### Dataset
- **350 primary instances** + 471 auxiliary (without folding processes)
- Each instance: CP diagram, compiled flat pattern, folding process (multi-step images), final 3D shape
- Complexity: Easy (3-9 steps), Medium (10-19), Hard (20-30), avg 8.2 steps
- **1,620 total questions** across 4 tasks

### Four Evaluation Tasks

| Task | Questions | Description |
|------|-----------|-------------|
| Pattern Prediction | 350 | CP diagram -> predict final 3D shape (multiple choice) |
| Multi-step Spatial Reasoning | 250 | Shuffled fold images -> correct chronological sequence |
| Spatial Relationship Prediction | 900 | 3 subtypes: pose localization, layering analysis, geometric change |
| End-to-End CP Code Generation | 120 | Flat layout + folded shape -> generate CP code |

### Compiler Architecture (Critical for OpenEnv)
Four-category error feedback system:
1. **CSE (CP Code Syntax Error):** Validates vertices, edges, faces, crease types; checks Euler's formula V-E+F=2
2. **GIF (Geometrically Impossible Fold):** Maekawa's theorem |M-V|=2, Kawasaki's theorem sum(alpha_i)=2pi, Big-Little-Big angle constraint
3. **PSI (Paper Self-Intersection):** Cyclic layering, collision detection (discrete + CCD), octrees/BVHs
4. **AFS (Ambiguous Folding State):** Multiple valid M/V assignments, non-unique stacking

### CP Code Evaluation (4 dimensions, 0.25 weight each)
1. **Topological Structure Similarity (TSS):** Vertex/edge/face count comparison, s_v = e^(-0.5|V_gen - V_ref| / min(V_gen, V_ref))
2. **Geometric Similarity (GS):** Hausdorff distance, s_p = e^(-5 * d_H), dihedral angle distribution, aspect ratio
3. **Constraint Satisfaction (CS):** Taco-Taco, Taco-Tortilla, transitivity, Maekawa/Kawasaki
4. **Final Folded State (FFS):** Shape similarity, layering comparison, stacking order

### Learning Approaches
- **In-Context Learning:** Single-pass, detailed instructions + examples
- **Environmental Learning:** Iterative model<->compiler loop, max 10 rounds, performance saturates after 8-10
- **Reinforcement Learning (TRICO/PPO-based):**
  - Training data: 471 instances from environmental learning
  - Model: Qwen2.5-VL-32B
  - **Rewards:** Intermediate (success bonus + quality progress), step penalty, final evaluation score
  - Result: RL-trained 32B exceeded 72B baseline

### Key Results
- Best closed-source: GPT-4o (42.71% pattern), Gemini2.5-pro (53.45% multi-step)
- Best open-source: Qwen2.5-VL-72B (36.29% pattern, 39.10% multi-step)
- Expert human: 98.45% pattern, 100% multi-step
- **Constraint satisfaction is the primary bottleneck** (~30% for top models)
- Human-model gap: 20-45 percentage points

### Relevance to Optigami
- **Direct blueprint for our OpenEnv server**: the compiler architecture with 4 error types is exactly what we need
- The CP code evaluation framework (TSS/GS/CS/FFS) can be our reward function
- Environmental learning approach maps to multi-turn rollouts in GRPO
- Confirms Qwen2.5-VL as viable base model (they used 32B, we target 7B)

---

## 3. Paper: SpatialThinker (2511.07403)

**Title:** SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards
**Authors:** Hunar Batra, Haoqin Tu, Hardy Chen, Yuanze Lin, Cihang Xie, Ronald Clark
**Date:** November 10, 2025
**Venue:** NeurIPS 2025 Workshops (SpaVLE, EWM, ARLET, SEA)

### Core Innovation
Dense spatial rewards + GRPO for training Qwen2.5-VL on spatial reasoning tasks. Key insight: **sparse rewards lead to reward hacking; dense multi-objective rewards with lexicographic gating prevent this.**

### GRPO Training Configuration
- **Rollouts:** 8 samples per query, temperature 1.0
- **Batch size:** rollout=512, global=128
- **Training:** 75 steps (~5 episodes)
- **Hardware:** 4x NVIDIA H100 80GB
- **Time:** ~13h (3B), ~15h (7B)
- **Advantage:** A(i) = (r(i) - mu) / (sigma + epsilon), epsilon=1e-6
- **Loss:** PPO-style with clip(epsilon_l=0.2, epsilon_h=0.3), KL penalty beta=0.01

### Dense Spatial Reward Design (CRITICAL - template for our rewards)

**4-component reward with lexicographic gating:**

```
R_total = I[R_format=1] * (w_format*R_f + w_count*R_c + w_accuracy*R_a + I[R_accuracy=1]*w_spatial*R_s)
```

| Component | Weight | Description |
|-----------|--------|-------------|
| Format (R_f) | 0.1 | JSON-parseable scene graph with required fields |
| Count (R_c) | 0.2 | Penalizes deviation in object/relation counts (lambda_obj=0.7, lambda_rel=0.3) |
| Accuracy (R_a) | 0.5 | Binary exact string match |
| Spatial (R_s) | 0.2 | Hungarian matching with CIoU, activated ONLY when answer correct |

**Lexicographic gating is essential:** format compliance gates all rewards; spatial rewards only activate on correct answers. Without gating, severe reward hacking occurs (74.9% -> 23.7% with naive spatial rewards).

### STVQA-7K Dataset
- 7,587 spatial VQA pairs from Visual Genome scene graphs
- Generated by Claude Sonnet, validated by GPT-4o pass@2
- 9 spatial categories, 34 additional spatial predicates beyond standard VG150
- 90/10 train/val split

### Key Results
- SpatialThinker-7B surpasses GPT-4o on 3DSRBench by +12.1%
- Dense reward RL: +7.2% avg across 12 benchmarks (1.8x the +4.0% from sparse GRPO)
- Outperforms models trained on millions of samples (trained on only 7K)

### Relevance to Optigami
- **Direct template for our GRPO training pipeline**
- Dense reward design with lexicographic gating prevents reward hacking
- Proves Qwen2.5-VL-7B is excellent base for spatial reasoning RL
- veRL/EasyR1 framework for training infrastructure
- Shows 7K samples sufficient for strong results

---

## 4. Paper: Automating Rigid Origami Design (2211.13219)

**Title:** Automating Rigid Origami Design
**Authors:** Jeremia Geiger, Karolis Martinkus, Oliver Richter, Roger Wattenhofer
**Date:** November 2022 (revised April 2023)
**Venue:** IJCAI 2023 AI, Arts & Creativity Special Track

### Core Contribution
- Formulates rigid origami design as discrete optimization: the **"rigid origami game"**
- Based on "three units method" principle
- Framework supports diverse objectives via abstract reward functions
- Generates optimized, application-specific crease patterns

### Methodology
- Multiple search methods within optimization framework
- Flexible objective definition for application-specific requirements
- Can approximate target shapes and produce functional designs

### Relevance to Optigami
- Validates the "origami as game/environment" paradigm we're building
- Their reward formulation approach (function-based, abstract) aligns with our OpenEnv design
- Discrete optimization over crease patterns = the action space for our RL agent

---

## 5. Tool: FOLD Format

**Repo:** https://github.com/edemaine/fold
**Authors:** Erik Demaine (MIT), Jason Ku (MIT), Robert Lang
**License:** MIT

### What It Is
FOLD (Flexible Origami List Datastructure) - JSON-based file format (.fold) for representing origami models. The **standard interchange format** for computational origami.

### Data Structure
```json
{
  "vertices_coords": [[x,y], ...],      // 2D or 3D coordinates
  "edges_vertices": [[v1,v2], ...],      // Edge endpoints
  "edges_assignment": ["M","V",...],     // Mountain/Valley/Boundary/Flat/Unassigned
  "faces_vertices": [[v1,v2,v3], ...],   // Face vertex lists
  "faceOrders": [[f1,f2,order], ...],    // Stacking/layering order
  "frame_*": ...                         // Multiple frames (folding states)
}
```

### JavaScript API
```javascript
// Browser
<script src="https://edemaine.github.io/fold/dist/fold.js"></script>

// Node.js
npm install --save fold

// Usage: FOLD.moduleName.functionName
FOLD.filter.collapseNearbyVertices(foldObject)
```

### CLI Tools
- `fold-convert`: ORIPA .opx -> .fold conversion
- `fold-convert --flat-fold`: Compute flat-folded state

### Supported Software Ecosystem
OrigamiSimulator, Freeform Origami (Tachi), Rabbit Ear (Kraft), ORIPA, Crease Pattern Editor, Rhino Grasshopper

### Relevance to Optigami
- **Core data format for OpenEnv state representation**
- JSON = easy Python/JS interop
- Stacking order (faceOrders) = layer tracking
- edges_assignment = mountain/valley fold type
- Import/export between geometry engine and visualizer

---

## 6. Tool: Origami Simulator

**Repo:** https://github.com/amandaghassaei/OrigamiSimulator
**URL:** origamisimulator.org
**Author:** Amanda Ghassaei
**License:** MIT
**Stack:** JavaScript (68.4%), Three.js, GPU fragment shaders

### Capabilities
- Real-time GPU-accelerated folding simulation
- Folds ALL creases simultaneously (not sequential)
- Realistic bending simulation between creases
- Strain visualization (internal stress during folding)
- Fold Percent slider: 0% (flat) to 100% (fully folded) to -100% (inverted)

### File Formats
- **Input:** SVG, FOLD
- **Export:** FOLD, STL, OBJ

### Physics Engine
- **Stiffness-based finite element approach:** Triangulated faces are rigid panels connected by rotational hinges along fold lines
- Each fold edge has a **target angle** (+/-pi for mountain/valley), driven by angular spring forces
- Solver computes nodal displacements at each timestep to reach equilibrium
- **Fold stiffness:** Controls how strongly hinges drive toward target angle
- **Face stiffness:** Controls rigidity of triangulated faces (resistance to bending/deformation)
- **Damping:** Controls oscillation decay rate
- **Strain metric:** Per-triangle deviation of edge lengths from rest lengths (flat state)
- Self-intersection is NOT prevented (folds through itself if geometry demands it)
- Based on Schenk & Guest structural engineering approach
- Tomohiro Tachi's freeform origami variations
- Ruling-aware triangulation for curved creases
- GPU fragment shaders for parallel computation

### Programmatic Usage
- Core simulation can be driven **headlessly** (without UI) by importing solver module
- Feed FOLD JSON data -> step simulation programmatically
- FOLD is JSON, so easy to generate crease patterns from Python and pass to simulator
- Can embed in other web pages as a component

### Dependencies
- Three.js (3D rendering)
- FOLD API (internal data structure)
- Earcut + cdt2d (polygon triangulation)
- numeric.js (linear algebra)
- CCapture (GIF/WebM export)

### Relevance to Optigami
- **Direct integration for Three.js Visualizer component**
- Strain heatmap capability already built in
- FOLD format native support
- Can be used for visual verification of generated fold patterns
- Export to STL/OBJ for 3D shape comparison in rewards

---

## 7. Tool: GamiBench

**Repo:** https://github.com/stvngo/GamiBench
**Dataset:** https://huggingface.co/datasets/stvngo/GamiBench
**Paper:** arXiv 2512.22207
**License:** MIT

### Benchmark Design
- 186 valid + 186 impossible crease patterns
- 6 viewpoints per pattern (top, bottom, front, back, right, left)
- **777 total samples** in HuggingFace dataset (45.4 MB)
- 186 label classes (named origami patterns)

### Task Types
1. Standard tasks (2D CP -> 3D prediction)
2. Alternative-view tasks
3. Impossible tasks (validity checking)

### Dataset Schema
```python
{
  "image": PIL.Image,     # Origami pattern/fold image
  "label": int,           # 0-185 class label
  "split": str            # Split identifier
}
```

### Loading
```python
from datasets import load_dataset
dataset = load_dataset("stvngo/GamiBench")
```

### Model Support
- OpenAI (GPT-4, GPT-4o-mini)
- Anthropic (Claude 4.5 Sonnet)
- Google (Gemini)
- xAI (Grok)
- OpenRouter models

### Code Structure
```
models/          # Model wrappers & factory
evaluators/      # BaseEvaluator: evaluate(), evaluate_single()
benchmarks/      # Benchmark implementations
configs/         # YAML/JSON configuration
utils/           # Shared helpers
pipeline.py      # Orchestration
run.py           # Entry point
```

### Relevance to Optigami
- **Evaluation benchmark for our trained model**
- 186 origami patterns = potential training/eval data
- Impossible patterns useful for constraint satisfaction testing
- Multi-view evaluation tests true 3D understanding
- Config-driven, reproducible evaluation pipeline

---

## 8. Tool: SpatialThinker Codebase

**Repo:** https://github.com/hunarbatra/SpatialThinker
**Paper:** arXiv 2511.07403

### Architecture
- Built on Qwen2.5-VL (3B and 7B variants)
- Uses veRL/EasyR1 for RL training
- vLLM 0.8.0 for inference during rollouts

### Code Structure
```
scripts/         # Training bash scripts per model size
evaluation/      # 18+ benchmark evaluation suite
data_gen/        # Data synthesis pipeline
verl/            # RL training framework (GRPO)
```

### Data Generation Pipeline
1. Generate raw QA pairs (12K-56K options)
2. Balance/filter with 50% spatial relations focus
3. Validate via GPT-4o (~75% pass rate)
4. Upload to HuggingFace

### Requirements
- Python 3.9+
- Transformers >= 4.49.0
- Flash-Attn >= 2.4.3
- vLLM >= 0.7.3

### Relevance to Optigami
- **Reference implementation for our GRPO training setup**
- veRL/EasyR1 framework = our training infrastructure
- Dense reward design directly applicable
- Data generation pipeline can be adapted for origami QA pairs

---

## 9. Tool: Trackio

**Repo:** https://github.com/gradio-app/trackio
**Author:** Hugging Face / Gradio team
**License:** MIT

### What It Is
Lightweight, local-first experiment tracking (Weights & Biases alternative). API-compatible with wandb.

### Key Features
- `import trackio as wandb` - drop-in W&B replacement
- Non-blocking `log()` with background queue (0.5s drain interval)
- SQLite local storage at `~/.cache/huggingface/trackio`
- Optional HuggingFace Spaces deployment for dashboards
- Slack/Discord webhook alerts (INFO/WARN/ERROR)
- 2,000 logs/8s single run; 32,000 logs/14s with 32 threads

### Usage
```python
import trackio

trackio.init(project="optigami-grpo", config={"lr": 1e-6, "model": "Qwen2.5-VL-7B"})
trackio.log({"step": step, "reward": reward, "loss": loss})
trackio.alert(title="Training spike", text="...", level=trackio.AlertLevel.WARN)
trackio.finish()

# Dashboard
trackio.show(project="optigami-grpo")
trackio.sync(project="optigami-grpo", space_id="openenv-community/optigami-training")
```

### Relevance to Optigami
- **Training metrics dashboard for GRPO training runs**
- Can deploy live dashboard to HF Spaces
- Track reward components, loss, constraint satisfaction rates
- Alert on training anomalies (reward hacking, loss spikes)

---

## 10. Tool: Unsloth + GRPO Training

**Repo:** https://github.com/unslothai/unsloth
**Docs:** https://unsloth.ai/docs

### GRPO Algorithm in Unsloth
1. Generate N responses per prompt (8+ recommended)
2. Score each with custom reward functions
3. Z-score normalize rewards across group -> advantages
4. PPO-style policy update (no value model or reward model needed)

### Memory Efficiency
- **90% less VRAM** vs standard GRPO
- 20K context, 8 generations, Llama 8B: 54.3GB (vs 510.8GB standard)
- QLoRA 4-bit: model params (GB) ~ VRAM needed
- Shared GPU memory with vLLM inference engine

### Vision Model Support
- Qwen2.5-VL-7B directly supported
- Qwen3-VL-8B, Gemma 3 (4B) also available
- `FastVisionModel.get_peft_model()` with granular layer control:
  - `finetune_vision_layers`, `finetune_language_layers`
  - `finetune_attention_modules`, `finetune_mlp_modules`

### LoRA Configuration
```python
model = FastVisionModel.get_peft_model(
    model,
    r=16,                          # LoRA rank
    lora_alpha=16,                 # alpha == r recommended
    lora_dropout=0,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
```

### GRPOConfig Options
```python
GRPOConfig(
    loss_type='grpo',        # or 'gspo', 'dr_grpo'
    epsilon=0.2,
    epsilon_high=0.28,
    delta=1.5,
    # ... standard training args
)
```

### vLLM Integration
- Shared memory between Unsloth and vLLM saves 3-5GB
- A100 40GB: ~4000 tokens/sec, T4 16GB: ~300 tokens/sec
- `fast_inference=True` enables vLLM backend

### Training Requirements
- Minimum 300 steps before meaningful progress
- 500+ data rows recommended (works with 10+)
- Models >= 1.5B parameters for reasoning tokens
- Steps = rows x epochs; increase generations (8->16) for more data

### Vision Data Format
```python
[
    {"role": "user", "content": [
        {"type": "text", "text": "instruction"},
        {"type": "image", "image": pil_image}
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "response"}
    ]}
]
```

### GRPO vs PPO vs DPO Comparison

| Aspect | PPO | DPO | GRPO |
|--------|-----|-----|------|
| Critic/Value model | Required (same size as policy) | Not needed | **Not needed** |
| Reference model | Required | Required | Required (old policy) |
| Training data | Online rollouts | Offline preference pairs | **Online rollouts + group scoring** |
| Reward signal | Scalar per token/step | Implicit from preferences | **Verifiable/explicit** |
| VRAM overhead | ~2x (policy + critic) | ~2x (policy + ref) | **~1.5x (no critic)** |

### GRPO Advantage Estimation
```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```
By sampling G completions and normalizing rewards within the group, GRPO creates its own baseline without a value network - halving VRAM vs PPO.

### Complete Unsloth GRPO Code Example
```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # Patch TRL with Unsloth optimizations

from trl import GRPOConfig, GRPOTrainer

# Load model with QLoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=None,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,                    # Higher rank for reasoning tasks
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,           # alpha == r recommended
    lora_dropout=0,          # Unsloth recommends 0
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized GC
    random_state=3407,
)

# Reward functions (TRL accepts a list, scores are summed)
def correctness_reward(completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        answer_match = re.search(r'</think>\s*(.*?)$', completion, re.DOTALL)
        if answer_match and answer_match.group(1).strip() == gt.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    return [0.5 if ("<think>" in c and "</think>" in c) else 0.0 for c in completions]

# GRPO Config
config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,              # Group size G
    max_completion_length=2048,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    beta=0.04,                      # KL penalty coefficient
    max_grad_norm=0.1,
    logging_steps=1,
    save_steps=250,
    bf16=True,
    loss_type='grpo',               # or 'gspo', 'dr_grpo'
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    reward_funcs=[correctness_reward, format_reward],
    tokenizer=tokenizer,
)
trainer.train()

# Save LoRA adapter
model.save_pretrained("./grpo_lora_adapter")
# Optional: merge and push
# model.save_pretrained_merged("./grpo_merged", tokenizer)
# model.push_to_hub_merged("username/model-name", tokenizer)
```

### Vision GRPO with Qwen2.5-VL
```python
from unsloth import FastVisionModel, PatchFastRL
PatchFastRL("GRPO", FastVisionModel)

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# For VLMs: typically freeze vision encoder, train language layers
model = FastVisionModel.get_peft_model(
    model,
    r=16,                          # Lower rank often sufficient for VLMs
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    finetune_vision_layers=False,  # Keep vision encoder frozen
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
```

### Unsloth ART (Agentic Reasoning Training)

ART extends GRPO for multi-turn agentic tasks:

1. **Multi-turn rollouts:** Model interacts with environment over multiple turns (actions + observations)
2. **Environment integration:** Custom env provides observations and final rewards
3. **Verifiable rewards:** Emphasizes automatically verifiable outcomes

**Multi-turn pattern:**
```
Turn 1: User prompt -> Model <think> + action -> Environment observation
Turn 2: Observation  -> Model <think> + action -> Environment observation
Turn 3: Observation  -> Model final answer     -> Reward computed
```

**Implementation options for multi-turn:**
1. **Single-generation (simpler):** Model outputs full plan/sequence in one generation; reward function evaluates the whole sequence
2. **Custom rollout loop (advanced):** Alternate model generation and env response, collect full trajectory, compute GRPO gradients on combined trajectory

### Key Hyperparameters Reference

| Parameter | Range | Notes |
|-----------|-------|-------|
| `num_generations` (G) | 4-16 | 8 common. More = better advantages, more VRAM |
| `beta` (KL penalty) | 0.01-0.1 | 0.04 default. Higher = stay closer to reference |
| `learning_rate` | 1e-6 to 1e-5 | Lower than SFT. 5e-6 starting point |
| `max_completion_length` | 512-4096 | Task-dependent |
| `r` (LoRA rank) | 16-128 | 64 for reasoning, 16 for VLM |
| `gradient_accumulation_steps` | 4-16 | Effective batch = per_device * accum * GPUs |
| `max_grad_norm` | 0.1-1.0 | 0.1 for stability |
| `warmup_ratio` | 0.05-0.1 | Important for RL stability |
| `epsilon` (clip) | 0.2 | PPO-style clipping |
| `epsilon_high` | 0.28 | Asymmetric upper clip |

### Qwen2.5-VL-7B Model Specifics
- Vision encoder: ViT with 2D-RoPE (handles arbitrary image resolutions via dynamic patching)
- LLM backbone: 28 layers, 3584 hidden dim, 28 attn heads, GQA with 4 KV heads
- Context: up to 32K tokens (128K with YaRN)
- Supports: single image, multi-image, video frames
- Unsloth IDs: `unsloth/Qwen2.5-VL-7B-Instruct`, `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`

### Qwen3-4B Model Specifics
- Hybrid thinking: can switch between `<think>` mode and direct response
- ~4B parameters, efficient for RL training
- MoE variants also available
- Unsloth IDs: `unsloth/Qwen3-4B`, `unsloth/Qwen3-4B-bnb-4bit`

---

## 11. Unsloth ART / GRPO Trainer Plan

### Phase 1: Data Preparation

**Training Data Sources:**
1. OrigamiSpace dataset (471 auxiliary instances) - CP diagrams, fold sequences, 3D shapes
2. GamiBench dataset (777 samples, 186 patterns) - crease patterns with multi-view 3D
3. Synthetic data generation pipeline (following SpatialThinker approach):
   - Generate origami QA pairs with Claude/GPT
   - Validate with GPT-4o pass@2
   - Balance across difficulty levels

**Data Format for GRPO:**
```python
# Each training example = a prompt with origami task
{
    "prompt": [
        {"role": "user", "content": [
            {"type": "image", "image": cp_diagram_image},
            {"type": "text", "text": "Given this crease pattern, describe the folding sequence and predict the final 3D shape. Output your answer as a FOLD JSON."}
        ]}
    ]
}
```

### Phase 2: Reward Function Design

**Following SpatialThinker's lexicographic gating pattern, adapted for origami:**

```python
def origami_reward(prompt, response, ground_truth):
    # Component 1: Format reward (gate)
    r_format = check_valid_fold_json(response)  # 0 or 1

    # Component 2: Constraint satisfaction
    r_constraints = check_origami_constraints(response)
    # - Maekawa's theorem: |M-V| = 2
    # - Kawasaki's theorem: sum(alpha_i) = 2*pi
    # - Euler's formula: V - E + F = 2
    # - No self-intersection

    # Component 3: Topological similarity
    r_topology = compute_tss(response, ground_truth)
    # Vertex/edge/face counts, connectivity

    # Component 4: Geometric similarity
    r_geometry = compute_hausdorff_similarity(response, ground_truth)

    # Component 5: Final shape match
    r_shape = compute_folded_state_similarity(response, ground_truth)

    # Lexicographic gating
    if r_format == 0:
        return 0.0

    total = (0.1 * r_format +
             0.25 * r_constraints +
             0.2 * r_topology +
             0.2 * r_geometry +
             0.25 * r_shape)

    return total
```

### Phase 3: Training Infrastructure

**Option A: Unsloth (simpler, less VRAM)**
```python
from unsloth import FastVisionModel
from trl import GRPOConfig, GRPOTrainer

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct",
    load_in_4bit=True,
    fast_inference=True,
)

model = FastVisionModel.get_peft_model(model, r=16, lora_alpha=16)

config = GRPOConfig(
    loss_type="grpo",
    num_generations=8,
    max_new_tokens=2048,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=1e-6,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    reward_funcs=[origami_reward],
)

trainer.train()
```

**Option B: veRL/EasyR1 (following SpatialThinker, more control)**
- Uses veRL framework with GRPO
- vLLM backend for fast rollouts
- More complex but battle-tested for spatial reasoning
- Better for multi-turn rollouts

### Phase 4: Multi-Turn Rollouts

Following OrigamiSpace's environmental learning approach:
1. Model generates CP code / fold sequence
2. OpenEnv compiler validates and returns error feedback
3. Model refines based on error type (CSE/GIF/PSI/AFS)
4. Repeat up to 10 rounds
5. Final reward based on best attempt

**Environment class pattern:**
```python
class OrigamiEnv:
    def __init__(self, task):
        self.task = task
        self.state = task["initial_state"]  # FOLD JSON
        self.steps = 0
        self.max_steps = 10
        self.history = []

    def step(self, action: str):
        """Process model's fold action, return compiler feedback."""
        self.steps += 1
        # Validate through compiler (CSE/GIF/PSI/AFS checks)
        result = self.compile_and_validate(action)
        observation = f"Step {self.steps}: {result['error_type']}: {result['message']}"
        self.state = result.get("new_state", self.state)
        self.history.append((action, observation))
        done = self.steps >= self.max_steps or result.get("valid", False)
        reward = self.compute_reward() if done else 0.0
        return observation, reward, done

    def compute_reward(self):
        """4-dimensional evaluation: TSS + GS + CS + FFS."""
        return (0.25 * tss(self.state, self.task["target"]) +
                0.25 * gs(self.state, self.task["target"]) +
                0.25 * cs(self.state) +
                0.25 * ffs(self.state, self.task["target"]))

def multi_turn_reward(completions, prompts, **kwargs):
    """Wrap environment interaction into GRPO reward function."""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        env = OrigamiEnv(extract_task(prompt))
        actions = parse_actions(completion)
        total_reward = 0.0
        for action in actions:
            obs, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return rewards
```

### Phase 5: Evaluation

1. **GamiBench** - standard origami spatial reasoning benchmark
2. **OrigamiSpace tasks** - 4-task evaluation suite
3. **Custom metrics:**
   - Constraint satisfaction rate (Maekawa/Kawasaki)
   - Compilation success rate
   - Topological/geometric similarity scores

### Phase 6: Monitoring with Trackio

```python
import trackio

trackio.init(
    project="optigami-grpo",
    space_id="openenv-community/optigami-training",
    config={
        "model": "Qwen2.5-VL-7B",
        "lora_r": 16,
        "num_generations": 8,
        "learning_rate": 1e-6,
    }
)

# In training loop
trackio.log({
    "step": step,
    "reward/total": total_reward,
    "reward/format": format_reward,
    "reward/constraints": constraint_reward,
    "reward/topology": topology_reward,
    "reward/geometry": geometry_reward,
    "reward/shape": shape_reward,
    "loss": loss,
    "compilation_rate": compilation_rate,
})
```

---

## 12. GitHub Reference Repo (ianalin123/optigami)

Located at `.reference/optigami-github/` (gitignored, not pushed to HF).

### What It Contains
A complete research repository with detailed architecture docs and a reference 2048 GRPO implementation.

### Key Files

| File | Contents |
|------|----------|
| `research/plan/architecture.md` | **Full architecture spec**: action space, state, physics engine, reward functions, OpenEnv integration, rendering pipeline, project structure, implementation order |
| `research/openenv/2048_example.py` | **636-line reference implementation** of OpenEnv + GRPO for 2048 game (Unsloth + TRL) |
| `research/openenv/overview.md` | OpenEnv framework API, types, project structure, deployment to HF Spaces |
| `research/origami/fold_types_deep.md` | All fold operations, Huzita-Justin axioms, crane step-by-step, compression patterns |
| `research/origami/math_physics_deep.md` | Kawasaki/Maekawa theorems with code, bar-and-hinge model, energy formulas |
| `research/origami/rendering_research.md` | Rendering options comparison |
| `research/origami/fold_format.md` | FOLD file format details |

### Architecture Decisions (from GitHub repo)

| Decision | Choice |
|----------|--------|
| LLM interaction | **Code-as-policy** (LLM writes `fold_strategy()` function) |
| Action space | Named fold ops (valley/mountain + fold line + angle) |
| State format | FOLD-compatible JSON |
| Physics engine | Bar-and-hinge model (NumPy port of Ghassaei) |
| Validation | Kawasaki + Maekawa + triangle-triangle intersection |
| Primary task | Solar panel packing (Miura-ori discovery) |
| Training | GRPO via TRL + Unsloth |
| Deployment | Docker Space on HF Spaces |

### Action Space (Code-as-Policy)
The LLM generates a `fold_strategy(paper_state)` function returning fold instructions:
```python
def fold_strategy(paper_state: dict) -> list[dict]:
    # paper_state contains: vertices, edges, assignments, fold_angles, material, etc.
    return [
        {"type": "valley", "line": {"start": [0,0.5], "end": [1,0.5]}, "angle": 180},
        {"type": "mountain", "line": {"start": [0.5,0], "end": [0.5,0.5]}, "angle": 180},
    ]
```

### Reward Functions (3 from 2048 pattern, adapted for origami)

1. **`code_valid`**: +1.0 valid function, -0.5 exec fails, -2.0 syntax error
2. **`physically_valid`**: +1.0 all valid, -2.0 per Kawasaki/Maekawa violation, -5.0 self-intersection
3. **`fold_quality`**: +20.0 * compactness, +10.0 meets volume target, +5.0 deployable, -0.5 per fold

### Physics Engine (Bar-and-Hinge Model)
```python
E_total = E_bar + E_facet + E_fold
E_bar   = sum (1/2) * k_axial * (L - L0)^2      # stretching
E_facet = sum (1/2) * k_facet * l * (theta-pi)^2  # panel bending
E_fold  = sum (1/2) * k_fold * l * (rho-rho_t)^2  # crease folding
```

### Planned Project Structure
```
engine/                  # Core simulation (numpy/scipy)
  paper.py               # Paper data structure, FOLD I/O
  fold_engine.py          # Apply folds (quaternion rotation)
  physics.py              # Bar-and-hinge energy, strain
  validation.py           # Kawasaki, Maekawa, self-intersection
  metrics.py              # Deployment ratio, compactness
  materials.py            # Material definitions

environment/             # OpenEnv server
  models.py              # Action, Observation, State
  origami_environment.py  # Environment (reset/step/state)
  tasks.py               # Task pool / curriculum
  app.py                 # create_app()
  Dockerfile

client/                  # OpenEnv client + training bridge
  reward_functions.py     # code_valid, physically_valid, fold_quality

training/                # Colab notebook
  train_origami.ipynb     # GRPO training (Unsloth + TRL)
  prompts.py             # LLM prompt templates
```

### Implementation Order (from architecture.md)
1. **Phase 1: Engine** - paper.py, fold_engine.py, validation.py, metrics.py
2. **Phase 2: OpenEnv Server** - models.py, origami_environment.py, app.py, Dockerfile
3. **Phase 3: Reward + Training** - reward_functions.py, prompts.py, train_origami.ipynb
4. **Phase 4: Rendering + Demo** - matplotlib headless, React + R3F app

### 2048 Reference Implementation (Key Patterns)
The `2048_example.py` shows the exact Unsloth + OpenEnv + GRPO pattern:
- `PatchFastRL` not used (text model, not vision) - for our VLM use `FastVisionModel`
- `extract_function()` parses code from ```python blocks
- `create_locked_down_function()` sandboxes execution
- `check_python_modules()` prevents non-stdlib imports
- `execute_with_time_limit(5)` wraps strategy execution
- Dataset: 1000x replicated prompt, `report_to="trackio"`
- GRPOConfig: temp=1.0, lr=2e-4, max_steps=600, num_generations=2
- Three reward functions passed as list to `GRPOTrainer`

---

## 13. Current Project State

### Repository
- **Location:** HuggingFace Space `openenv-community/optigami`
- **Framework:** Create React App (React 19.1.0)
- **Status:** Fresh scaffold - default CRA boilerplate
- **Build:** `npm run build` -> `build/index.html` (HF Spaces static SDK)

### File Structure
```
optigami/
  package.json          # React app dependencies
  README.md             # CRA default + HF Space metadata
  public/               # Static assets (favicon, manifest)
  src/
    App.js              # Default CRA component (placeholder)
    App.css
    index.js            # Entry point
    index.css
    logo.svg
    reportWebVitals.js
    setupTests.js
    App.test.js
```

### What Needs to Be Built
1. **Python backend** - Paper Geometry Engine with Shapely, FOLD import/export, constraint checking
2. **GRPO training scripts** - Unsloth or veRL-based, with origami reward functions
3. **Data pipeline** - Load/process OrigamiSpace + GamiBench datasets
4. **Three.js frontend** - Replace CRA boilerplate with origami visualizer (possibly integrate OrigamiSimulator)
5. **OpenEnv server** - API connecting geometry engine to trainer

---

## Key Takeaways for Immediate Work (GRPO Trainer)

1. **Use Unsloth for simplicity** - 90% VRAM savings, built-in vLLM, QLoRA support for Qwen2.5-VL-7B
2. **Dense rewards with lexicographic gating** - format gate -> constraints -> topology -> geometry -> shape match (SpatialThinker pattern)
3. **OrigamiSpace's 4-error compiler** is the gold standard for reward signal generation
4. **Start with 500+ origami examples** - GamiBench (777) + OrigamiSpace (471) = 1248 examples
5. **8 generations per prompt**, temperature 1.0, 300+ training steps minimum
6. **Multi-turn: max 10 rounds** with compiler feedback (performance saturates after 8-10)
7. **Track with Trackio** - deploy dashboard to HF Spaces for real-time monitoring
8. **Evaluate on GamiBench** for standardized comparison against other MLLMs

---

## Cross-Reference: Tool Compatibility Matrix

| Component | FOLD | OrigamiSim | GamiBench | SpatialThinker | Unsloth | Trackio |
|-----------|------|------------|-----------|----------------|---------|---------|
| State representation | Core | Import | - | - | - | - |
| Visualization | Export | Core | - | - | - | - |
| Training data | - | - | Core | Augment | - | - |
| RL training | - | - | Eval | Template | Core | Monitor |
| Reward functions | Validate | Strain | - | Template | Integrate | Log |
| Constraint checking | Structure | Physics | Impossible set | - | - | - |
