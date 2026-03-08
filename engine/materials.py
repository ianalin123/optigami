"""
Material definitions for origami simulation.

Provides dataclass-based material properties and preset materials
for paper, mylar, aluminum, and nitinol.
"""

from dataclasses import dataclass


@dataclass
class Material:
    name: str
    thickness_mm: float       # mm
    youngs_modulus_gpa: float  # GPa
    max_strain: float          # fraction (e.g. 0.03 = 3%)
    poissons_ratio: float = 0.3  # dimensionless

    @property
    def thickness_m(self) -> float:
        """Thickness in meters."""
        return self.thickness_mm / 1000.0

    @property
    def youngs_modulus_pa(self) -> float:
        """Young's modulus in Pascals."""
        return self.youngs_modulus_gpa * 1e9


# ── Preset materials ────────────────────────────────────────────────

MATERIAL_PRESETS: dict[str, Material] = {
    "paper": Material(
        name="paper",
        thickness_mm=0.1,
        youngs_modulus_gpa=2.0,
        max_strain=0.03,
        poissons_ratio=0.3,
    ),
    "mylar": Material(
        name="mylar",
        thickness_mm=0.05,
        youngs_modulus_gpa=4.0,
        max_strain=0.03,
        poissons_ratio=0.38,
    ),
    "aluminum": Material(
        name="aluminum",
        thickness_mm=0.1,
        youngs_modulus_gpa=69.0,
        max_strain=0.01,
        poissons_ratio=0.33,
    ),
    "nitinol": Material(
        name="nitinol",
        thickness_mm=0.1,
        youngs_modulus_gpa=75.0,
        max_strain=0.08,
        poissons_ratio=0.33,
    ),
}


def get_material(name: str) -> Material:
    """Return a copy of a preset material by name.

    Raises KeyError if name is not in MATERIAL_PRESETS.
    """
    if name not in MATERIAL_PRESETS:
        available = ", ".join(sorted(MATERIAL_PRESETS))
        raise KeyError(f"Unknown material '{name}'. Available: {available}")
    preset = MATERIAL_PRESETS[name]
    return Material(
        name=preset.name,
        thickness_mm=preset.thickness_mm,
        youngs_modulus_gpa=preset.youngs_modulus_gpa,
        max_strain=preset.max_strain,
        poissons_ratio=preset.poissons_ratio,
    )
