"""
Instruction parser: converts human text into a structured origami task.

Handles variations like:
  - "make a paper crane"
  - "fold me a crane"
  - "i want to make a paper crane"
  - "crane please"
  - "can you fold a crane?"
  - "pack a 1m x 1m mylar sheet as compact as possible"
"""

from __future__ import annotations

import re
from planner.knowledge import ORIGAMI_MODELS, list_known_models


# ---------------------------------------------------------------------------
# Vocabulary for matching
# ---------------------------------------------------------------------------

# Model aliases → canonical name
_MODEL_ALIASES: dict[str, str] = {
    # Crane
    "crane": "crane",
    "tsuru": "crane",
    "bird": "crane",
    "orizuru": "crane",
    # Boat
    "boat": "boat",
    "hat": "boat",
    "paper boat": "boat",
    "paper hat": "boat",
    # Airplane
    "airplane": "airplane",
    "aeroplane": "airplane",
    "plane": "airplane",
    "paper airplane": "airplane",
    "paper plane": "airplane",
    "dart": "airplane",
    "paper dart": "airplane",
    # Box
    "box": "box",
    "masu": "box",
    "masu box": "box",
    "open box": "box",
    # Fortune teller
    "fortune teller": "fortune_teller",
    "fortune-teller": "fortune_teller",
    "cootie catcher": "fortune_teller",
    "cootie-catcher": "fortune_teller",
    "chatterbox": "fortune_teller",
    # Waterbomb / balloon
    "waterbomb": "waterbomb",
    "water bomb": "waterbomb",
    "balloon": "waterbomb",
    "paper balloon": "waterbomb",
    # Jumping frog
    "jumping frog": "jumping_frog",
    "frog": "jumping_frog",
    "leap frog": "jumping_frog",
}

# Sorted longest-first so multi-word aliases match before single-word ones
_ALIAS_KEYS_SORTED = sorted(_MODEL_ALIASES.keys(), key=len, reverse=True)

_MATERIALS = {
    "paper": "paper",
    "mylar": "mylar",
    "aluminum": "aluminum",
    "aluminium": "aluminum",
    "metal": "metal",
    "nitinol": "nitinol",
    "foil": "aluminum",
    "cardboard": "cardboard",
    "cardstock": "cardstock",
    "fabric": "fabric",
    "cloth": "fabric",
}

# Intent keywords
_FOLD_VERBS = {
    "make", "fold", "create", "build", "construct", "origami",
    "craft", "form", "shape", "assemble",
}
_PACK_VERBS = {
    "pack", "compress", "compact", "minimize", "reduce", "stow",
    "shrink", "deploy", "collapse",
}
_OPTIMIZE_PHRASES = [
    "as compact as possible",
    "minimize volume",
    "minimize packed volume",
    "minimize area",
    "solar panel",
    "stent",
    "deployable",
    "maximize compactness",
    "flatten",
]

# Dimension patterns
_DIM_PATTERNS = [
    # "10cm x 10cm", "10 cm x 10 cm"
    re.compile(
        r"(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet)\s*[xX\u00d7]\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet)",
        re.IGNORECASE,
    ),
    # "10cm square", "1 meter square"
    re.compile(
        r"(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|meter|meters|metre|metres)\s+square",
        re.IGNORECASE,
    ),
]

_UNIT_TO_M = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "feet": 0.3048,
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and strip extra whitespace."""
    return " ".join(text.lower().split())


def _detect_model(text: str) -> str | None:
    """Return canonical model name if one is mentioned, else None."""
    norm = _normalise(text)

    # Strip out constraint phrases that might contain model-name false positives
    # e.g. "into a 15cm x 15cm x 5cm box" should not match "box" as a model
    cleaned = re.sub(
        r"(?:fit\s+)?(?:in(?:to)?|inside)\s+(?:a\s+)?\d.*?box",
        "",
        norm,
    )

    for alias in _ALIAS_KEYS_SORTED:
        # Use word-boundary-aware search to avoid partial matches
        pattern = r"(?:^|\b)" + re.escape(alias) + r"(?:\b|$)"
        if re.search(pattern, cleaned):
            return _MODEL_ALIASES[alias]
    return None


def _detect_material(text: str) -> str:
    """Return detected material or default 'paper'."""
    norm = _normalise(text)
    # Check multi-word materials first (none currently, but future-proof)
    for keyword, canonical in sorted(_MATERIALS.items(), key=lambda kv: -len(kv[0])):
        if keyword in norm:
            return canonical
    return "paper"


def _detect_dimensions(text: str) -> dict[str, float]:
    """Parse explicit dimensions. Returns {"width": m, "height": m} or defaults."""
    for pat in _DIM_PATTERNS:
        m = pat.search(text)
        if m:
            groups = m.groups()
            if len(groups) == 4:
                # WxH pattern
                w = float(groups[0]) * _UNIT_TO_M.get(groups[1].lower(), 1.0)
                h = float(groups[2]) * _UNIT_TO_M.get(groups[3].lower(), 1.0)
                return {"width": round(w, 6), "height": round(h, 6)}
            elif len(groups) == 2:
                # "N unit square"
                side = float(groups[0]) * _UNIT_TO_M.get(groups[1].lower(), 1.0)
                return {"width": round(side, 6), "height": round(side, 6)}
    # Default: unit square
    return {"width": 1.0, "height": 1.0}


def _detect_constraints(text: str) -> dict:
    """Detect any explicit constraints mentioned in the instruction."""
    norm = _normalise(text)
    constraints: dict = {}

    # Target bounding box: "fit in a 15cm x 15cm x 5cm box"
    box_pat = re.compile(
        r"fit\s+(?:in(?:to)?|inside)\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*(cm|mm|m)\s*[xX\u00d7]\s*"
        r"(\d+(?:\.\d+)?)\s*(cm|mm|m)\s*[xX\u00d7]\s*(\d+(?:\.\d+)?)\s*(cm|mm|m)",
        re.IGNORECASE,
    )
    bm = box_pat.search(norm)
    if bm:
        g = bm.groups()
        constraints["target_box"] = [
            float(g[0]) * _UNIT_TO_M.get(g[1], 1.0),
            float(g[2]) * _UNIT_TO_M.get(g[3], 1.0),
            float(g[4]) * _UNIT_TO_M.get(g[5], 1.0),
        ]

    # Max folds
    folds_pat = re.compile(r"(?:max(?:imum)?|at most|no more than)\s+(\d+)\s+fold", re.IGNORECASE)
    fm = folds_pat.search(norm)
    if fm:
        constraints["max_folds"] = int(fm.group(1))

    # Compactness emphasis
    for phrase in _OPTIMIZE_PHRASES:
        if phrase in norm:
            constraints["optimize_compactness"] = True
            break

    # Must deploy
    if "deploy" in norm or "unfold" in norm and "clean" in norm:
        constraints["must_deploy"] = True

    return constraints


def _detect_intent(text: str, model_name: str | None, constraints: dict) -> str:
    """Determine the high-level intent of the instruction."""
    norm = _normalise(text)
    words = set(norm.split())

    # If packing / optimization phrases are present, it's an optimization task
    if constraints.get("optimize_compactness"):
        return "optimize_packing"
    if words & _PACK_VERBS:
        return "optimize_packing"

    # If a known model is detected, it's a fold_model task
    if model_name is not None:
        return "fold_model"

    # If fold verbs are present but no model, it's a free fold
    if words & _FOLD_VERBS:
        return "free_fold"

    # Fallback: if there's a pattern keyword
    pattern_words = {"miura", "tessellation", "pattern", "waterbomb tessellation", "pleat"}
    if words & pattern_words:
        return "fold_pattern"

    return "free_fold"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_instruction(text: str) -> dict:
    """
    Parse a human origami instruction into a structured task.

    Args:
        text: Natural-language instruction, e.g. "make a paper crane"

    Returns:
        {
            "intent": "fold_model" | "fold_pattern" | "optimize_packing" | "free_fold",
            "model_name": str or None,
            "material": str,
            "dimensions": {"width": float, "height": float},
            "constraints": {...},
            "raw_instruction": str,
        }
    """
    model_name = _detect_model(text)
    material = _detect_material(text)
    dimensions = _detect_dimensions(text)
    constraints = _detect_constraints(text)
    intent = _detect_intent(text, model_name, constraints)

    return {
        "intent": intent,
        "model_name": model_name,
        "material": material,
        "dimensions": dimensions,
        "constraints": constraints,
        "raw_instruction": text,
    }
