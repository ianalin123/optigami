"""
Origami knowledge base: bases, models, and fold operation primitives.

All fold coordinates are defined on a unit square (0,0) to (1,1).
Fold lines use {"start": [x, y], "end": [x, y]} format matching the
engine's FoldAction specification from architecture.md.
"""

# ---------------------------------------------------------------------------
# Primitive fold operations — mapping from named folds to engine-level dicts
# ---------------------------------------------------------------------------

FOLD_OPERATIONS = {
    "valley_fold": {
        "type": "valley",
        "angle": 180,
        "description": "Fold paper toward you along a crease line.",
    },
    "mountain_fold": {
        "type": "mountain",
        "angle": 180,
        "description": "Fold paper away from you along a crease line.",
    },
    "squash_fold": {
        "type": "squash",
        "angle": 180,
        "description": "Open a flap and flatten it symmetrically.",
        "primitives": [
            {"type": "valley", "angle": 90, "sub_op": "open_flap"},
            {"type": "valley", "angle": 180, "sub_op": "flatten"},
        ],
    },
    "petal_fold": {
        "type": "petal",
        "angle": 180,
        "description": "Lift a point while collapsing sides inward to create a narrow flap.",
        "primitives": [
            {"type": "valley", "angle": 180, "sub_op": "fold_left_edge_to_center"},
            {"type": "valley", "angle": 180, "sub_op": "fold_right_edge_to_center"},
            {"type": "valley", "angle": 180, "sub_op": "lift_bottom_point"},
            {"type": "mountain", "angle": 180, "sub_op": "collapse_left"},
            {"type": "mountain", "angle": 180, "sub_op": "collapse_right"},
        ],
    },
    "reverse_inside_fold": {
        "type": "reverse_inside",
        "angle": 180,
        "description": "Push a flap tip inward, reversing the spine crease.",
        "primitives": [
            {"type": "valley", "angle": 180, "sub_op": "new_crease_left"},
            {"type": "valley", "angle": 180, "sub_op": "new_crease_right"},
            {"type": "mountain", "angle": 180, "sub_op": "reverse_spine"},
        ],
    },
    "reverse_outside_fold": {
        "type": "reverse_outside",
        "angle": 180,
        "description": "Wrap a flap tip around the outside, reversing the spine crease.",
        "primitives": [
            {"type": "mountain", "angle": 180, "sub_op": "new_crease_left"},
            {"type": "mountain", "angle": 180, "sub_op": "new_crease_right"},
            {"type": "valley", "angle": 180, "sub_op": "reverse_spine"},
        ],
    },
    "crimp": {
        "type": "crimp",
        "angle": 180,
        "description": "Pair of reverse folds creating a zigzag step.",
        "primitives": [
            {"type": "valley", "angle": 180, "sub_op": "first_crease"},
            {"type": "mountain", "angle": 180, "sub_op": "second_crease"},
        ],
    },
    "pleat": {
        "type": "pleat",
        "angle": 180,
        "description": "Alternating valley and mountain folds creating an accordion.",
        "primitives": [
            {"type": "valley", "angle": 180, "sub_op": "valley_crease"},
            {"type": "mountain", "angle": 180, "sub_op": "mountain_crease"},
        ],
    },
    "rabbit_ear": {
        "type": "rabbit_ear",
        "angle": 180,
        "description": "Three creases meeting at a point, creating a triangular raised flap.",
        "primitives": [
            {"type": "valley", "angle": 180, "sub_op": "bisector_1"},
            {"type": "valley", "angle": 180, "sub_op": "bisector_2"},
            {"type": "mountain", "angle": 180, "sub_op": "ridge"},
        ],
    },
    "sink_fold": {
        "type": "sink",
        "angle": 180,
        "description": "Push a point into the interior of the model.",
        "primitives": [
            {"type": "mountain", "angle": 180, "sub_op": "reverse_creases"},
            {"type": "valley", "angle": 180, "sub_op": "reflatten"},
        ],
    },
    "turn_over": {
        "type": "turn_over",
        "angle": 0,
        "description": "Flip the paper over.",
    },
    "unfold": {
        "type": "unfold",
        "angle": 0,
        "description": "Reverse a previous fold (crease remains).",
    },
    "inflate": {
        "type": "inflate",
        "angle": 0,
        "description": "Gently open and puff out the model to create 3D form.",
    },
}


# ---------------------------------------------------------------------------
# Origami bases — fundamental starting configurations
# ---------------------------------------------------------------------------

ORIGAMI_BASES = {
    "preliminary_base": {
        "name": "Preliminary Base (Square Base)",
        "description": "Multi-layered diamond standing on a corner. Gateway to bird and frog bases.",
        "total_steps": 9,
        "steps": [
            {
                "description": "Fold in half diagonally (bottom-left to top-right)",
                "type": "valley",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold",
                "type": "unfold",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Fold in half on other diagonal (bottom-right to top-left)",
                "type": "valley",
                "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold",
                "type": "unfold",
                "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Fold in half horizontally",
                "type": "valley",
                "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold",
                "type": "unfold",
                "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Fold in half vertically",
                "type": "valley",
                "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold",
                "type": "unfold",
                "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Collapse into preliminary base: push sides in using existing creases",
                "type": "valley",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 180,
                "layer_select": "all",
                "simultaneous": [
                    {"type": "valley", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 180},
                    {"type": "mountain", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180},
                    {"type": "mountain", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 180},
                ],
            },
        ],
    },

    "waterbomb_base": {
        "name": "Waterbomb Base",
        "description": "Flat triangle with multiple layers. Inverse of preliminary base.",
        "total_steps": 9,
        "steps": [
            {
                "description": "Fold both diagonals — first diagonal",
                "type": "valley",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold first diagonal",
                "type": "unfold",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Fold second diagonal",
                "type": "valley",
                "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold second diagonal",
                "type": "unfold",
                "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Mountain fold horizontally",
                "type": "mountain",
                "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold horizontal",
                "type": "unfold",
                "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Mountain fold vertically",
                "type": "mountain",
                "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold vertical",
                "type": "unfold",
                "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Collapse into waterbomb base: fold top edge down, push sides in",
                "type": "mountain",
                "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]},
                "angle": 180,
                "layer_select": "all",
                "simultaneous": [
                    {"type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180},
                    {"type": "valley", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 180},
                ],
            },
        ],
    },

    "bird_base": {
        "name": "Bird Base (Crane Base)",
        "description": "Long diamond with 4 narrow flaps. Built from preliminary base + 2 petal folds.",
        "requires_base": "preliminary_base",
        "total_steps": 13,
        "steps": [
            # Steps 1-9 are the preliminary base (included by reference)
            # Steps 10-22 from the crane sequence build the bird base
            {
                "description": "Fold left edge of top layer to center line",
                "type": "valley",
                "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Fold right edge of top layer to center line",
                "type": "valley",
                "line": {"start": [0.75, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Fold top triangle down over kite flaps (crease only)",
                "type": "valley",
                "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Unfold top triangle",
                "type": "unfold",
                "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]},
                "angle": 0,
                "layer_select": "top",
            },
            {
                "description": "Unfold kite folds",
                "type": "unfold",
                "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]},
                "angle": 0,
                "layer_select": "top",
            },
            {
                "description": "Petal fold front: lift bottom point up, sides collapse inward",
                "type": "petal",
                "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Turn model over",
                "type": "turn_over",
                "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Fold left edge to center line (back)",
                "type": "valley",
                "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Fold right edge to center line (back)",
                "type": "valley",
                "line": {"start": [0.75, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Fold top triangle down (crease only, back)",
                "type": "valley",
                "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Unfold top triangle (back)",
                "type": "unfold",
                "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]},
                "angle": 0,
                "layer_select": "top",
            },
            {
                "description": "Unfold kite folds (back)",
                "type": "unfold",
                "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]},
                "angle": 0,
                "layer_select": "top",
            },
            {
                "description": "Petal fold back: lift bottom point up, sides collapse inward",
                "type": "petal",
                "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
        ],
    },

    "frog_base": {
        "name": "Frog Base",
        "description": "4 long narrow flaps radiating from center. Built from preliminary base + 4 squash + 4 petal folds.",
        "requires_base": "preliminary_base",
        "total_steps": 8,
        "steps": [
            {
                "description": "Squash fold front-left flap",
                "type": "squash",
                "line": {"start": [0.25, 0.5], "end": [0.5, 0.75]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Squash fold front-right flap",
                "type": "squash",
                "line": {"start": [0.75, 0.5], "end": [0.5, 0.75]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Squash fold back-left flap",
                "type": "squash",
                "line": {"start": [0.25, 0.5], "end": [0.5, 0.75]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Squash fold back-right flap",
                "type": "squash",
                "line": {"start": [0.75, 0.5], "end": [0.5, 0.75]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Petal fold first diamond",
                "type": "petal",
                "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Petal fold second diamond",
                "type": "petal",
                "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Petal fold third diamond",
                "type": "petal",
                "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
            {
                "description": "Petal fold fourth diamond",
                "type": "petal",
                "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]},
                "angle": 180,
                "layer_select": "top",
            },
        ],
    },

    "fish_base": {
        "name": "Fish Base",
        "description": "Diamond shape with 4 points. Built from kite folds + rabbit ears.",
        "total_steps": 7,
        "steps": [
            {
                "description": "Fold diagonal crease (reference line)",
                "type": "valley",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold diagonal",
                "type": "unfold",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Kite fold: fold bottom-left edge to diagonal",
                "type": "valley",
                "line": {"start": [0.0, 0.0], "end": [0.5, 0.5]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Kite fold: fold top-left edge to diagonal",
                "type": "valley",
                "line": {"start": [0.0, 1.0], "end": [0.5, 0.5]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Rabbit ear fold on bottom-right corner",
                "type": "rabbit_ear",
                "line": {"start": [0.5, 0.0], "end": [1.0, 0.5]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Rabbit ear fold on top-right corner",
                "type": "rabbit_ear",
                "line": {"start": [0.5, 1.0], "end": [1.0, 0.5]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Fold resulting flaps down flat",
                "type": "valley",
                "line": {"start": [0.5, 0.5], "end": [1.0, 0.5]},
                "angle": 180,
                "layer_select": "top",
            },
        ],
    },

    "kite_base": {
        "name": "Kite Base",
        "description": "Simplest base: a kite shape from two folds to a diagonal.",
        "total_steps": 3,
        "steps": [
            {
                "description": "Fold diagonal (reference crease)",
                "type": "valley",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 180,
                "layer_select": "all",
            },
            {
                "description": "Unfold diagonal",
                "type": "unfold",
                "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]},
                "angle": 0,
                "layer_select": "all",
            },
            {
                "description": "Fold bottom-left and top-left edges to lie on diagonal",
                "type": "valley",
                "line": {"start": [0.0, 0.0], "end": [0.5, 0.5]},
                "angle": 180,
                "layer_select": "all",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Complete origami models — full fold sequences from flat square to finished
# ---------------------------------------------------------------------------

ORIGAMI_MODELS = {
    "crane": {
        "name": "Paper Crane (Tsuru)",
        "difficulty": "intermediate",
        "base": "bird_base",
        "total_steps": 31,
        "description": "The traditional Japanese crane. 31 steps from flat square.",
        "steps": [
            # Phase 1: Pre-crease (steps 1-8)
            {"step": 1, "description": "Fold square in half diagonally (bottom-left to top-right)", "type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 2, "description": "Unfold", "type": "unfold", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 3, "description": "Fold in half on other diagonal", "type": "valley", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 4, "description": "Unfold", "type": "unfold", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 5, "description": "Fold in half horizontally", "type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 6, "description": "Unfold", "type": "unfold", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 0, "layer_select": "all"},
            {"step": 7, "description": "Fold in half vertically", "type": "valley", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 8, "description": "Unfold", "type": "unfold", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "all"},

            # Phase 2: Collapse into preliminary base (step 9)
            {"step": 9, "description": "Collapse into preliminary base: push left and right edges inward, fold top down", "type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180, "layer_select": "all"},

            # Phase 3: Front kite folds (steps 10-14)
            {"step": 10, "description": "Fold left edge of top layer to center line", "type": "valley", "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 11, "description": "Fold right edge of top layer to center line", "type": "valley", "line": {"start": [0.75, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 12, "description": "Fold top triangle down over kite flaps", "type": "valley", "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]}, "angle": 180, "layer_select": "top"},
            {"step": 13, "description": "Unfold step 12", "type": "unfold", "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]}, "angle": 0, "layer_select": "top"},
            {"step": 14, "description": "Unfold steps 10-11", "type": "unfold", "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "top"},

            # Phase 4: Front petal fold (step 15)
            {"step": 15, "description": "Petal fold: lift bottom point of top layer upward, sides collapse inward", "type": "petal", "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},

            # Phase 5: Repeat on back (steps 16-22)
            {"step": 16, "description": "Turn model over", "type": "turn_over", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 17, "description": "Fold left edge to center line", "type": "valley", "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 18, "description": "Fold right edge to center line", "type": "valley", "line": {"start": [0.75, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 19, "description": "Fold top triangle down", "type": "valley", "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]}, "angle": 180, "layer_select": "top"},
            {"step": 20, "description": "Unfold step 19", "type": "unfold", "line": {"start": [0.25, 0.75], "end": [0.75, 0.75]}, "angle": 0, "layer_select": "top"},
            {"step": 21, "description": "Unfold steps 17-18", "type": "unfold", "line": {"start": [0.25, 0.5], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "top"},
            {"step": 22, "description": "Petal fold back: lift bottom point up, collapse sides in. Bird base complete.", "type": "petal", "line": {"start": [0.5, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},

            # Phase 6: Narrow the legs (steps 23-27)
            {"step": 23, "description": "Fold left flap (front) edge to center", "type": "valley", "line": {"start": [0.375, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 24, "description": "Fold right flap (front) edge to center", "type": "valley", "line": {"start": [0.625, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 25, "description": "Turn over", "type": "turn_over", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 26, "description": "Fold left flap (back) edge to center", "type": "valley", "line": {"start": [0.375, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 27, "description": "Fold right flap (back) edge to center", "type": "valley", "line": {"start": [0.625, 0.5], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},

            # Phase 7: Form neck and tail (steps 28-29)
            {"step": 28, "description": "Inside reverse fold left flap upward to form neck", "type": "reverse_inside", "line": {"start": [0.35, 0.6], "end": [0.45, 0.85]}, "angle": 150, "layer_select": "all"},
            {"step": 29, "description": "Inside reverse fold right flap upward to form tail", "type": "reverse_inside", "line": {"start": [0.55, 0.6], "end": [0.65, 0.85]}, "angle": 150, "layer_select": "all"},

            # Phase 8: Head and finish (steps 30-31)
            {"step": 30, "description": "Inside reverse fold tip of neck downward to form head/beak", "type": "reverse_inside", "line": {"start": [0.38, 0.82], "end": [0.42, 0.9]}, "angle": 150, "layer_select": "all"},
            {"step": 31, "description": "Pull wings apart gently and press bottom to inflate body", "type": "inflate", "line": {"start": [0.5, 0.5], "end": [0.5, 0.7]}, "angle": 0, "layer_select": "all"},
        ],
    },

    "boat": {
        "name": "Simple Boat",
        "difficulty": "simple",
        "base": None,
        "total_steps": 9,
        "description": "A flat boat/hat from simple valley and mountain folds.",
        "steps": [
            {"step": 1, "description": "Fold in half horizontally (top to bottom)", "type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 2, "description": "Fold in half vertically (crease only)", "type": "valley", "line": {"start": [0.5, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 3, "description": "Unfold vertical", "type": "unfold", "line": {"start": [0.5, 0.0], "end": [0.5, 0.5]}, "angle": 0, "layer_select": "all"},
            {"step": 4, "description": "Fold top-left corner down to center mark", "type": "valley", "line": {"start": [0.15, 0.5], "end": [0.5, 0.35]}, "angle": 180, "layer_select": "top"},
            {"step": 5, "description": "Fold top-right corner down to center mark", "type": "valley", "line": {"start": [0.85, 0.5], "end": [0.5, 0.35]}, "angle": 180, "layer_select": "top"},
            {"step": 6, "description": "Fold bottom strip up (front layer)", "type": "valley", "line": {"start": [0.0, 0.15], "end": [1.0, 0.15]}, "angle": 180, "layer_select": "top"},
            {"step": 7, "description": "Turn over", "type": "turn_over", "line": {"start": [0.5, 0.0], "end": [0.5, 0.5]}, "angle": 0, "layer_select": "all"},
            {"step": 8, "description": "Fold bottom strip up (back layer)", "type": "valley", "line": {"start": [0.0, 0.15], "end": [1.0, 0.15]}, "angle": 180, "layer_select": "top"},
            {"step": 9, "description": "Open from bottom and flatten into boat shape", "type": "inflate", "line": {"start": [0.5, 0.0], "end": [0.5, 0.5]}, "angle": 0, "layer_select": "all"},
        ],
    },

    "airplane": {
        "name": "Paper Airplane (Dart)",
        "difficulty": "simple",
        "base": None,
        "total_steps": 6,
        "description": "Classic dart-style paper airplane using only valley folds.",
        "steps": [
            {"step": 1, "description": "Fold in half vertically (left to right)", "type": "valley", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 2, "description": "Unfold", "type": "unfold", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 3, "description": "Fold top-left corner to center line", "type": "valley", "line": {"start": [0.0, 1.0], "end": [0.5, 0.7]}, "angle": 180, "layer_select": "all"},
            {"step": 4, "description": "Fold top-right corner to center line", "type": "valley", "line": {"start": [1.0, 1.0], "end": [0.5, 0.7]}, "angle": 180, "layer_select": "all"},
            {"step": 5, "description": "Fold left angled edge to center line", "type": "valley", "line": {"start": [0.0, 0.7], "end": [0.5, 0.4]}, "angle": 180, "layer_select": "all"},
            {"step": 6, "description": "Fold right angled edge to center line", "type": "valley", "line": {"start": [1.0, 0.7], "end": [0.5, 0.4]}, "angle": 180, "layer_select": "all"},
        ],
    },

    "box": {
        "name": "Masu Box (Open-Top Box)",
        "difficulty": "low_intermediate",
        "base": None,
        "total_steps": 13,
        "description": "An open-top box. Uses preliminary base concept with tuck folds.",
        "steps": [
            {"step": 1, "description": "Fold in half horizontally", "type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 2, "description": "Unfold", "type": "unfold", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 0, "layer_select": "all"},
            {"step": 3, "description": "Fold in half vertically", "type": "valley", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 4, "description": "Unfold", "type": "unfold", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 5, "description": "Fold all four corners to center", "type": "valley", "line": {"start": [0.0, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 6, "description": "Fold bottom-right corner to center", "type": "valley", "line": {"start": [1.0, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 7, "description": "Fold top-left corner to center", "type": "valley", "line": {"start": [0.0, 1.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 8, "description": "Fold top-right corner to center", "type": "valley", "line": {"start": [1.0, 1.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 9, "description": "Fold top third down to center", "type": "valley", "line": {"start": [0.0, 0.667], "end": [1.0, 0.667]}, "angle": 180, "layer_select": "all"},
            {"step": 10, "description": "Fold bottom third up to center", "type": "valley", "line": {"start": [0.0, 0.333], "end": [1.0, 0.333]}, "angle": 180, "layer_select": "all"},
            {"step": 11, "description": "Unfold top and bottom, and unfold left/right corners", "type": "unfold", "line": {"start": [0.0, 0.333], "end": [1.0, 0.333]}, "angle": 0, "layer_select": "all"},
            {"step": 12, "description": "Raise left and right walls using existing creases, tuck corners in", "type": "valley", "line": {"start": [0.25, 0.0], "end": [0.25, 1.0]}, "angle": 90, "layer_select": "all"},
            {"step": 13, "description": "Fold flaps over into box and lock walls in place", "type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 90, "layer_select": "top"},
        ],
    },

    "fortune_teller": {
        "name": "Fortune Teller (Cootie Catcher)",
        "difficulty": "simple",
        "base": None,
        "total_steps": 8,
        "description": "Classic fortune teller: fold corners to center, flip, repeat.",
        "steps": [
            {"step": 1, "description": "Fold in half diagonally (crease only)", "type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 2, "description": "Unfold", "type": "unfold", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 3, "description": "Fold bottom-left corner to center", "type": "valley", "line": {"start": [0.0, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 4, "description": "Fold bottom-right corner to center", "type": "valley", "line": {"start": [1.0, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 5, "description": "Fold top-left corner to center", "type": "valley", "line": {"start": [0.0, 1.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 6, "description": "Fold top-right corner to center", "type": "valley", "line": {"start": [1.0, 1.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 7, "description": "Turn over", "type": "turn_over", "line": {"start": [0.5, 0.0], "end": [0.5, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 8, "description": "Fold all four new corners to center again", "type": "valley", "line": {"start": [0.25, 0.25], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "all"},
        ],
    },

    "waterbomb": {
        "name": "Waterbomb (Paper Balloon)",
        "difficulty": "simple",
        "base": "waterbomb_base",
        "total_steps": 12,
        "description": "Inflatable paper balloon built on the waterbomb base.",
        "steps": [
            # Phase 1: Waterbomb base (steps 1-9 same as waterbomb_base)
            {"step": 1, "description": "Fold first diagonal", "type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 2, "description": "Unfold", "type": "unfold", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 3, "description": "Fold second diagonal", "type": "valley", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 180, "layer_select": "all"},
            {"step": 4, "description": "Unfold", "type": "unfold", "line": {"start": [1.0, 0.0], "end": [0.0, 1.0]}, "angle": 0, "layer_select": "all"},
            {"step": 5, "description": "Fold in half horizontally", "type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 6, "description": "Collapse into waterbomb base (triangle)", "type": "valley", "line": {"start": [0.0, 0.0], "end": [1.0, 1.0]}, "angle": 180, "layer_select": "all"},
            # Phase 2: Fold flaps to top
            {"step": 7, "description": "Fold bottom-left corner of front layer up to top", "type": "valley", "line": {"start": [0.0, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "top"},
            {"step": 8, "description": "Fold bottom-right corner of front layer up to top", "type": "valley", "line": {"start": [1.0, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "top"},
            # Phase 3: Tuck flaps
            {"step": 9, "description": "Fold left and right points to center", "type": "valley", "line": {"start": [0.25, 0.25], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "top"},
            {"step": 10, "description": "Tuck small triangles into pockets", "type": "valley", "line": {"start": [0.35, 0.4], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "top"},
            {"step": 11, "description": "Repeat steps 7-10 on back", "type": "valley", "line": {"start": [0.0, 0.0], "end": [0.5, 0.5]}, "angle": 180, "layer_select": "top"},
            # Phase 4: Inflate
            {"step": 12, "description": "Blow into hole at bottom to inflate into cube/sphere", "type": "inflate", "line": {"start": [0.5, 0.0], "end": [0.5, 0.5]}, "angle": 0, "layer_select": "all"},
        ],
    },

    "jumping_frog": {
        "name": "Jumping Frog",
        "difficulty": "low_intermediate",
        "base": None,
        "total_steps": 15,
        "description": "A frog that jumps when you press its back. Uses pleats for the spring.",
        "steps": [
            {"step": 1, "description": "Fold in half horizontally", "type": "valley", "line": {"start": [0.0, 0.5], "end": [1.0, 0.5]}, "angle": 180, "layer_select": "all"},
            {"step": 2, "description": "Fold top-left corner to right edge", "type": "valley", "line": {"start": [0.0, 1.0], "end": [1.0, 0.75]}, "angle": 180, "layer_select": "all"},
            {"step": 3, "description": "Unfold", "type": "unfold", "line": {"start": [0.0, 1.0], "end": [1.0, 0.75]}, "angle": 0, "layer_select": "all"},
            {"step": 4, "description": "Fold top-right corner to left edge", "type": "valley", "line": {"start": [1.0, 1.0], "end": [0.0, 0.75]}, "angle": 180, "layer_select": "all"},
            {"step": 5, "description": "Unfold", "type": "unfold", "line": {"start": [1.0, 1.0], "end": [0.0, 0.75]}, "angle": 0, "layer_select": "all"},
            {"step": 6, "description": "Collapse top into waterbomb-like triangle", "type": "valley", "line": {"start": [0.0, 0.75], "end": [1.0, 0.75]}, "angle": 180, "layer_select": "all"},
            {"step": 7, "description": "Fold left point of triangle up and outward for front leg", "type": "valley", "line": {"start": [0.25, 0.75], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 8, "description": "Fold right point of triangle up and outward for front leg", "type": "valley", "line": {"start": [0.75, 0.75], "end": [0.5, 1.0]}, "angle": 180, "layer_select": "top"},
            {"step": 9, "description": "Fold bottom half up to meet triangle base", "type": "valley", "line": {"start": [0.0, 0.375], "end": [1.0, 0.375]}, "angle": 180, "layer_select": "all"},
            {"step": 10, "description": "Fold left side to center", "type": "valley", "line": {"start": [0.25, 0.0], "end": [0.25, 0.75]}, "angle": 180, "layer_select": "all"},
            {"step": 11, "description": "Fold right side to center", "type": "valley", "line": {"start": [0.75, 0.0], "end": [0.75, 0.75]}, "angle": 180, "layer_select": "all"},
            {"step": 12, "description": "Fold bottom up", "type": "valley", "line": {"start": [0.25, 0.25], "end": [0.75, 0.25]}, "angle": 180, "layer_select": "all"},
            {"step": 13, "description": "Pull back legs out to sides", "type": "valley", "line": {"start": [0.375, 0.0], "end": [0.375, 0.375]}, "angle": 180, "layer_select": "top"},
            {"step": 14, "description": "Pleat fold for spring: fold bottom half down", "type": "mountain", "line": {"start": [0.25, 0.15], "end": [0.75, 0.15]}, "angle": 180, "layer_select": "all"},
            {"step": 15, "description": "Valley fold bottom portion back up for spring action", "type": "valley", "line": {"start": [0.25, 0.08], "end": [0.75, 0.08]}, "angle": 180, "layer_select": "all"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_base_steps(base_name: str) -> list[dict]:
    """Return the fold steps for a given base, or empty list if not found."""
    base = ORIGAMI_BASES.get(base_name)
    if base is None:
        return []
    return list(base["steps"])


def get_model_steps(model_name: str) -> list[dict]:
    """Return the full fold steps for a known model, or empty list."""
    model = ORIGAMI_MODELS.get(model_name)
    if model is None:
        return []
    return list(model["steps"])


def list_known_models() -> list[str]:
    """Return names of all known origami models."""
    return list(ORIGAMI_MODELS.keys())


def list_known_bases() -> list[str]:
    """Return names of all known origami bases."""
    return list(ORIGAMI_BASES.keys())


def get_fold_operation(name: str) -> dict | None:
    """Look up a fold operation by name."""
    return FOLD_OPERATIONS.get(name)
