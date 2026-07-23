"""
TCAT / Xtacking-style Integrated Geometry Object Evaluator
==========================================================

설치:
    pip install numpy pandas matplotlib PySide6

실행:
    python tcat_xtacking_simple_evaluator.py

Rev 방향:
- YMTC Xtacking 개념도처럼 단순 3단 구조로 표현
    1) 하단 3D NAND Array Stack
    2) 중간 Bonding VIA / Metal VIA Interface
    3) 상단 Periphery Circuit Plane
- 모든 Geometry Object는 3D View에서 클릭 가능하게 등록
- 기존 Memory Hole / Channel Hole REF-COMP 분석 로직은 유지
- 나머지 Object는 구조 탐색 + 향후 계측 Mapping 정의 표시
- Object Ontology 탭에서 Object-Measurement-Parameter-Risk-Window 관계를 그래프로 표시
"""

from __future__ import annotations

import sys
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# =========================================================
# 1. Object Registry
# =========================================================

@dataclass(frozen=True)
class ObjectDefinition:
    object_id: str
    name: str
    category: str
    primary_process: str
    measurement_keys: Tuple[str, ...]
    geometry_parameters: Tuple[str, ...]
    risks: Tuple[str, ...]
    process_windows: Tuple[str, ...]
    description: str
    analysis_supported: bool = False


OBJECT_REGISTRY: Dict[str, ObjectDefinition] = {
    "PERIPHERY_CIRCUIT": ObjectDefinition(
        object_id="PERIPHERY_CIRCUIT",
        name="Periphery Circuit Plane",
        category="Xtacking Top Die",
        primary_process="Periphery CMOS / Logic Formation",
        measurement_keys=("PERI_THICKNESS", "PERI_PAD_PITCH", "PERI_BOND_OVERLAY"),
        geometry_parameters=("logic_plane_thickness", "bond_pad_pitch", "bond_overlay"),
        risks=("I/O Delay", "Thermal Coupling", "Bond Pad Misalignment"),
        process_windows=("Bond Pad Pitch × Overlay",),
        description=(
            "Xtacking 구조의 상단 Periphery/CMOS 회로 평면입니다. "
            "데이터 I/O와 Memory Cell Operation 회로가 위치하는 영역으로 단순화했습니다."
        ),
    ),
    "BONDING_VIA": ObjectDefinition(
        object_id="BONDING_VIA",
        name="Bonding VIA / Metal VIA",
        category="Xtacking Interface",
        primary_process="Wafer Bonding / Metal VIA Connection",
        measurement_keys=("BVIA_CD", "BVIA_HEIGHT", "BVIA_OVERLAY", "BVIA_RESISTANCE"),
        geometry_parameters=("via_cd", "via_height", "overlay", "resistance"),
        risks=("Via Open", "Resistance Increase", "Bonding Misalignment"),
        process_windows=("VIA CD × Overlay",),
        description=(
            "상단 Periphery와 하단 NAND Array를 전기적으로 연결하는 수직 Metal VIA 영역입니다. "
            "현재는 클릭 가능한 구조 Object와 향후 계측 Mapping 정의만 제공합니다."
        ),
    ),
    "BIT_LINE": ObjectDefinition(
        object_id="BIT_LINE",
        name="Bit Line / Array Top Metal",
        category="Array Interconnect",
        primary_process="Bit Line Patterning",
        measurement_keys=("BL_WIDTH", "BL_THICKNESS", "BL_OVERLAY"),
        geometry_parameters=("width", "thickness", "overlay"),
        risks=("Open", "Resistance Increase", "Overlay Margin"),
        process_windows=("Width × Overlay",),
        description="하단 NAND Array 상부의 Bit Line / top metal 배선 구조입니다.",
    ),
    "SOURCE_LINE": ObjectDefinition(
        object_id="SOURCE_LINE",
        name="Source Line",
        category="Array Interconnect",
        primary_process="Source Line Formation",
        measurement_keys=("SL_WIDTH", "SL_CONTACT_CD", "SL_OVERLAY"),
        geometry_parameters=("width", "contact_cd", "overlay"),
        risks=("Contact Open", "Resistance Increase"),
        process_windows=("Contact CD × Overlay",),
        description="String source 연결용 Array 내부 배선 구조입니다.",
    ),
    "SELECT_GATE": ObjectDefinition(
        object_id="SELECT_GATE",
        name="Select Gate",
        category="Array Gate",
        primary_process="Select Gate Etch",
        measurement_keys=("SG_THICKNESS", "SG_RECESS", "SG_OVERLAY"),
        geometry_parameters=("thickness", "recess", "overlay"),
        risks=("Select Failure", "Gate Leakage"),
        process_windows=("Recess × Thickness",),
        description="상부 String 선택용 Gate 구조입니다.",
    ),
    "CONTROL_GATE_STACK": ObjectDefinition(
        object_id="CONTROL_GATE_STACK",
        name="Control Gate / Word Line Stack",
        category="Repeated Array Stack",
        primary_process="Word Line Replacement",
        measurement_keys=("WL_THICKNESS", "WL_RECESS", "WL_PITCH"),
        geometry_parameters=("wl_thickness", "recess", "pitch"),
        risks=("WL Leakage", "Thickness Variation", "Stack Stress"),
        process_windows=("WL Thickness × Recess",),
        description="3D NAND Array의 Word Line / Control Gate 반복 적층 구조입니다.",
    ),
    "MEMORY_HOLE": ObjectDefinition(
        object_id="MEMORY_HOLE",
        name="Memory Hole / Channel Hole",
        category="Vertical Channel",
        primary_process="Channel Hole Etch",
        measurement_keys=(
            "CH_TOP_CD",
            "CH_MIDDLE_CD",
            "CH_BOTTOM_CD",
            "CH_DEPTH",
            "CH_BOWING",
            "CH_TILT",
        ),
        geometry_parameters=(
            "top_cd",
            "middle_cd",
            "bottom_cd",
            "depth",
            "bowing",
            "tilt_deg",
        ),
        risks=(
            "Etch Completion",
            "Bottom Open",
            "Poly Fill",
            "Alignment / Proximity",
        ),
        process_windows=(
            "Bottom CD × Tilt",
            "Bottom CD × Bowing",
            "Depth × Bottom CD",
        ),
        description=(
            "Channel Hole Etch의 핵심 Geometry Object입니다. "
            "Lot 그룹의 실제 계측값을 형상 파라미터에 매핑해 구조 Risk와 Process Window를 평가합니다."
        ),
        analysis_supported=True,
    ),
    "PIPE_CONNECTION": ObjectDefinition(
        object_id="PIPE_CONNECTION",
        name="Pipe / Bottom Connection",
        category="Array Bottom Connection",
        primary_process="Pipe Etch / Bottom Connection",
        measurement_keys=("PIPE_CD", "PIPE_DEPTH", "PIPE_OVERLAY"),
        geometry_parameters=("pipe_cd", "depth", "overlay"),
        risks=("Connection Open", "Local Void"),
        process_windows=("Pipe CD × Depth",),
        description="하단 Channel 연결부 / bottom connection 구조입니다.",
    ),
    "ARRAY_CHIP": ObjectDefinition(
        object_id="ARRAY_CHIP",
        name="3D NAND Array Chip / Base",
        category="Xtacking Bottom Die",
        primary_process="Array Wafer Formation",
        measurement_keys=("ARRAY_STACK_HEIGHT", "ARRAY_DIE_BOW", "ARRAY_BOND_OVERLAY"),
        geometry_parameters=("stack_height", "die_bow", "bond_overlay"),
        risks=("Array Die Bow", "Bond Overlay Margin", "Stack Mechanical Stress"),
        process_windows=("Die Bow × Bond Overlay",),
        description=(
            "Xtacking 구조의 하단 3D NAND Array chip/base입니다. "
            "Word Line stack과 vertical channel array가 포함되는 하부 die 영역으로 단순화했습니다."
        ),
    ),
}


# =========================================================
# 1-1. Object Ontology Links
# =========================================================

# Simple ontology edge list.  Each tuple means:
#   source object -> target object, relationship label
# The graph view uses both forward and reverse links so selecting either side
# exposes relevant structural neighbors.
OBJECT_ONTOLOGY_LINKS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "PERIPHERY_CIRCUIT": (
        ("BONDING_VIA", "bonded through"),
        ("ARRAY_CHIP", "operates / drives"),
    ),
    "BONDING_VIA": (
        ("BIT_LINE", "lands on array top metal"),
        ("ARRAY_CHIP", "connects top die to"),
        ("MEMORY_HOLE", "read path proximity"),
    ),
    "ARRAY_CHIP": (
        ("CONTROL_GATE_STACK", "contains"),
        ("MEMORY_HOLE", "contains channel array"),
        ("PIPE_CONNECTION", "contains bottom connection"),
        ("SELECT_GATE", "contains select layer"),
        ("BIT_LINE", "contains top metal"),
        ("SOURCE_LINE", "contains source rail"),
    ),
    "CONTROL_GATE_STACK": (
        ("MEMORY_HOLE", "surrounds / gated by"),
        ("SELECT_GATE", "stack interface"),
    ),
    "MEMORY_HOLE": (
        ("BIT_LINE", "upper read path"),
        ("SOURCE_LINE", "source path"),
        ("PIPE_CONNECTION", "bottom connected to"),
        ("SELECT_GATE", "selected by"),
    ),
    "SOURCE_LINE": (
        ("PIPE_CONNECTION", "source / bottom path"),
    ),
}


def get_ontology_neighbors(object_id: str) -> List[Tuple[str, str]]:
    """Return neighbor object ids and relation labels for the ontology graph."""
    neighbors: List[Tuple[str, str]] = []
    seen = set()

    for target_id, relation in OBJECT_ONTOLOGY_LINKS.get(object_id, ()):  # forward
        if target_id in OBJECT_REGISTRY and target_id not in seen:
            neighbors.append((target_id, relation))
            seen.add(target_id)

    for source_id, edges in OBJECT_ONTOLOGY_LINKS.items():  # reverse
        for target_id, relation in edges:
            if target_id == object_id and source_id in OBJECT_REGISTRY and source_id not in seen:
                neighbors.append((source_id, relation))
                seen.add(source_id)

    return neighbors


# =========================================================
# 2. Mock Measurement Data
# =========================================================

def generate_mock_data(seed: int = 42, lots_per_group: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    groups = {
        "REF_GROUP_A": {
            "recipe": "ETCH_RECIPE_A",
            "source_power": 1000,
            "bias_power": 480,
            "pressure": 22.0,
            "etch_time": 180,
            "means": {
                "CH_TOP_CD": 120.0,
                "CH_MIDDLE_CD": 130.0,
                "CH_BOTTOM_CD": 86.0,
                "CH_DEPTH": 8000.0,
                "CH_BOWING": 5.0,
                "CH_TILT": 0.20,
            },
        },
        "COMP_GROUP_B": {
            "recipe": "ETCH_RECIPE_B",
            "source_power": 1045,
            "bias_power": 462,
            "pressure": 24.0,
            "etch_time": 185,
            "means": {
                "CH_TOP_CD": 118.0,
                "CH_MIDDLE_CD": 132.0,
                "CH_BOTTOM_CD": 81.5,
                "CH_DEPTH": 8350.0,
                "CH_BOWING": 8.5,
                "CH_TILT": 0.31,
            },
        },
        "COMP_GROUP_C": {
            "recipe": "ETCH_RECIPE_C",
            "source_power": 1018,
            "bias_power": 474,
            "pressure": 22.8,
            "etch_time": 182,
            "means": {
                "CH_TOP_CD": 119.0,
                "CH_MIDDLE_CD": 131.0,
                "CH_BOTTOM_CD": 84.5,
                "CH_DEPTH": 8150.0,
                "CH_BOWING": 6.4,
                "CH_TILT": 0.24,
            },
        },
    }

    sigma = {
        "CH_TOP_CD": 1.1,
        "CH_MIDDLE_CD": 1.5,
        "CH_BOTTOM_CD": 1.3,
        "CH_DEPTH": 75.0,
        "CH_BOWING": 0.7,
        "CH_TILT": 0.022,
    }


    # Additional object-level mock measurements so every clickable object can
    # show REF/COMP group means immediately after selection.
    extra_means = {
        "REF_GROUP_A": {
            "PERI_THICKNESS": 720.0,
            "PERI_PAD_PITCH": 180.0,
            "PERI_BOND_OVERLAY": 4.2,
            "BVIA_CD": 42.0,
            "BVIA_HEIGHT": 900.0,
            "BVIA_OVERLAY": 3.6,
            "BVIA_RESISTANCE": 12.0,
            "BL_WIDTH": 38.0,
            "BL_THICKNESS": 52.0,
            "BL_OVERLAY": 3.2,
            "SL_WIDTH": 46.0,
            "SL_CONTACT_CD": 64.0,
            "SL_OVERLAY": 3.8,
            "SG_THICKNESS": 68.0,
            "SG_RECESS": 7.8,
            "SG_OVERLAY": 3.4,
            "WL_THICKNESS": 36.0,
            "WL_RECESS": 5.2,
            "WL_PITCH": 58.0,
            "PIPE_CD": 78.0,
            "PIPE_DEPTH": 420.0,
            "PIPE_OVERLAY": 4.0,
            "ARRAY_STACK_HEIGHT": 14800.0,
            "ARRAY_DIE_BOW": 18.0,
            "ARRAY_BOND_OVERLAY": 4.5,
        },
        "COMP_GROUP_B": {
            "PERI_THICKNESS": 735.0,
            "PERI_PAD_PITCH": 180.0,
            "PERI_BOND_OVERLAY": 5.8,
            "BVIA_CD": 39.5,
            "BVIA_HEIGHT": 925.0,
            "BVIA_OVERLAY": 5.4,
            "BVIA_RESISTANCE": 14.8,
            "BL_WIDTH": 36.5,
            "BL_THICKNESS": 51.0,
            "BL_OVERLAY": 4.4,
            "SL_WIDTH": 44.8,
            "SL_CONTACT_CD": 61.5,
            "SL_OVERLAY": 4.9,
            "SG_THICKNESS": 66.8,
            "SG_RECESS": 9.1,
            "SG_OVERLAY": 4.5,
            "WL_THICKNESS": 35.2,
            "WL_RECESS": 6.4,
            "WL_PITCH": 58.4,
            "PIPE_CD": 74.5,
            "PIPE_DEPTH": 438.0,
            "PIPE_OVERLAY": 5.2,
            "ARRAY_STACK_HEIGHT": 15120.0,
            "ARRAY_DIE_BOW": 24.0,
            "ARRAY_BOND_OVERLAY": 5.9,
        },
        "COMP_GROUP_C": {
            "PERI_THICKNESS": 728.0,
            "PERI_PAD_PITCH": 180.0,
            "PERI_BOND_OVERLAY": 4.9,
            "BVIA_CD": 40.8,
            "BVIA_HEIGHT": 910.0,
            "BVIA_OVERLAY": 4.3,
            "BVIA_RESISTANCE": 13.1,
            "BL_WIDTH": 37.2,
            "BL_THICKNESS": 51.5,
            "BL_OVERLAY": 3.9,
            "SL_WIDTH": 45.5,
            "SL_CONTACT_CD": 62.8,
            "SL_OVERLAY": 4.2,
            "SG_THICKNESS": 67.4,
            "SG_RECESS": 8.5,
            "SG_OVERLAY": 3.9,
            "WL_THICKNESS": 35.7,
            "WL_RECESS": 5.8,
            "WL_PITCH": 58.2,
            "PIPE_CD": 76.2,
            "PIPE_DEPTH": 429.0,
            "PIPE_OVERLAY": 4.6,
            "ARRAY_STACK_HEIGHT": 14980.0,
            "ARRAY_DIE_BOW": 21.0,
            "ARRAY_BOND_OVERLAY": 5.1,
        },
    }

    extra_sigma = {
        "PERI_THICKNESS": 6.5,
        "PERI_PAD_PITCH": 0.8,
        "PERI_BOND_OVERLAY": 0.35,
        "BVIA_CD": 0.9,
        "BVIA_HEIGHT": 9.0,
        "BVIA_OVERLAY": 0.35,
        "BVIA_RESISTANCE": 0.35,
        "BL_WIDTH": 0.55,
        "BL_THICKNESS": 0.8,
        "BL_OVERLAY": 0.25,
        "SL_WIDTH": 0.65,
        "SL_CONTACT_CD": 0.95,
        "SL_OVERLAY": 0.30,
        "SG_THICKNESS": 0.75,
        "SG_RECESS": 0.30,
        "SG_OVERLAY": 0.25,
        "WL_THICKNESS": 0.45,
        "WL_RECESS": 0.25,
        "WL_PITCH": 0.25,
        "PIPE_CD": 1.0,
        "PIPE_DEPTH": 5.5,
        "PIPE_OVERLAY": 0.35,
        "ARRAY_STACK_HEIGHT": 85.0,
        "ARRAY_DIE_BOW": 1.5,
        "ARRAY_BOND_OVERLAY": 0.35,
    }

    rows: List[dict] = []

    for group_name, config in groups.items():
        for idx in range(1, lots_per_group + 1):
            row = {
                "group": group_name,
                "lot_id": f"{group_name[:4]}_LOT_{idx:03d}",
                "recipe_id": config["recipe"],
                "source_power": rng.normal(config["source_power"], 7),
                "bias_power": rng.normal(config["bias_power"], 5),
                "pressure": rng.normal(config["pressure"], 0.35),
                "etch_time": rng.normal(config["etch_time"], 1.2),
            }

            for key, mean in config["means"].items():
                row[key] = rng.normal(mean, sigma[key])

            for key, mean in extra_means.get(group_name, {}).items():
                row[key] = rng.normal(mean, extra_sigma.get(key, max(abs(mean) * 0.01, 0.001)))

            rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# 3. Memory Hole Geometry & Risk
# =========================================================

MEASUREMENT_TO_PARAMETER = {
    "CH_TOP_CD": "top_cd",
    "CH_MIDDLE_CD": "middle_cd",
    "CH_BOTTOM_CD": "bottom_cd",
    "CH_DEPTH": "depth",
    "CH_BOWING": "bowing",
    "CH_TILT": "tilt_deg",
}


@dataclass(frozen=True)
class ChannelHoleGeometry:
    object_id: str
    group_name: str
    top_cd: float
    middle_cd: float
    bottom_cd: float
    depth: float
    bowing: float
    tilt_deg: float
    wl_count: int = 256


@dataclass
class GeometryEvaluation:
    minimum_cd: float
    aspect_ratio: float
    bottom_offset: float
    taper_delta: float
    critical_depth: float
    total_risk: float
    level: str
    critical_wl: int
    main_driver: str
    risk_items: Dict[str, float]


def group_to_channel_geometry(df: pd.DataFrame, group_name: str) -> ChannelHoleGeometry:
    group_df = df[df["group"] == group_name]

    if group_df.empty:
        raise ValueError(f"그룹 데이터가 없습니다: {group_name}")

    means = group_df[list(MEASUREMENT_TO_PARAMETER.keys())].mean()

    return ChannelHoleGeometry(
        object_id="MEMORY_HOLE",
        group_name=group_name,
        top_cd=float(means["CH_TOP_CD"]),
        middle_cd=float(means["CH_MIDDLE_CD"]),
        bottom_cd=float(means["CH_BOTTOM_CD"]),
        depth=float(means["CH_DEPTH"]),
        bowing=float(means["CH_BOWING"]),
        tilt_deg=float(means["CH_TILT"]),
    )


def cd_profile(z: np.ndarray, geometry: ChannelHoleGeometry) -> np.ndarray:
    t = np.clip(z / max(geometry.depth, 1.0), 0.0, 1.0)

    linear = geometry.top_cd + (geometry.bottom_cd - geometry.top_cd) * t
    linear_mid = (geometry.top_cd + geometry.bottom_cd) / 2.0
    measured_mid_bow = geometry.middle_cd - linear_mid
    return np.maximum(
        # Anchor the reconstructed wall profile to the measured top, middle,
        # and bottom CDs.  Bowing remains an independent measured risk input;
        # adding it here as well would double-count the middle-zone expansion.
        linear + measured_mid_bow * np.sin(np.pi * t) ** 2,
        1.0,
    )


def center_offset(z: np.ndarray, geometry: ChannelHoleGeometry) -> np.ndarray:
    return np.tan(np.deg2rad(geometry.tilt_deg)) * z


def low_limit_score(value: float, caution: float, risk: float) -> float:
    if value >= caution:
        return 0.0
    if value <= risk:
        return 1.0
    return float((caution - value) / (caution - risk))


def high_limit_score(value: float, caution: float, risk: float) -> float:
    if value <= caution:
        return 0.0
    if value >= risk:
        return 1.0
    return float((value - caution) / (risk - caution))


def risk_level(score: float) -> str:
    if score < 0.35:
        return "SAFE"
    if score < 0.60:
        return "CAUTION"
    return "RISK"


def evaluate_geometry(geometry: ChannelHoleGeometry) -> GeometryEvaluation:
    z = np.linspace(0.0, geometry.depth, 500)
    cd = cd_profile(z, geometry)
    offset = np.abs(center_offset(z, geometry))

    minimum_cd = float(np.min(cd))
    min_index = int(np.argmin(cd))

    aspect_ratio = geometry.depth / max(minimum_cd, 1.0)
    bottom_offset = float(offset[-1])
    taper_delta = geometry.top_cd - geometry.bottom_cd

    bottom_cd_risk = low_limit_score(geometry.bottom_cd, 88.0, 74.0)
    minimum_cd_risk = low_limit_score(minimum_cd, 86.0, 72.0)
    ar_risk = high_limit_score(aspect_ratio, 88.0, 125.0)
    depth_risk = high_limit_score(geometry.depth, 8200.0, 9800.0)
    bowing_risk = high_limit_score(geometry.bowing, 7.0, 16.0)
    tilt_risk = high_limit_score(geometry.tilt_deg, 0.20, 0.52)
    offset_risk = high_limit_score(bottom_offset, 22.0, 52.0)
    taper_risk = high_limit_score(taper_delta, 32.0, 52.0)

    etch_completion = float(np.clip(
        0.50 * ar_risk + 0.30 * depth_risk + 0.20 * minimum_cd_risk,
        0.0,
        1.0,
    ))
    bottom_open = float(np.clip(
        0.60 * bottom_cd_risk + 0.25 * taper_risk + 0.15 * ar_risk,
        0.0,
        1.0,
    ))
    poly_fill = float(np.clip(
        0.50 * minimum_cd_risk + 0.30 * bowing_risk + 0.20 * ar_risk,
        0.0,
        1.0,
    ))
    alignment = float(np.clip(
        0.60 * tilt_risk + 0.40 * offset_risk,
        0.0,
        1.0,
    ))

    risk_items = {
        "Etch Completion": etch_completion,
        "Bottom Open": bottom_open,
        "Poly Fill": poly_fill,
        "Alignment / Proximity": alignment,
    }

    total_risk = float(
        0.30 * etch_completion
        + 0.30 * bottom_open
        + 0.22 * poly_fill
        + 0.18 * alignment
    )

    driver_scores = {
        "Bottom CD": bottom_cd_risk,
        "Aspect Ratio": max(ar_risk, depth_risk),
        "Bowing": bowing_risk,
        "Tilt": max(tilt_risk, offset_risk),
    }
    main_driver = max(driver_scores, key=driver_scores.get)

    critical_depth = float(z[min_index])
    critical_wl = int(
        round(
            critical_depth
            / max(geometry.depth, 1.0)
            * max(geometry.wl_count - 1, 1)
        ) + 1
    )
    critical_wl = max(1, min(geometry.wl_count, critical_wl))

    return GeometryEvaluation(
        minimum_cd=minimum_cd,
        aspect_ratio=aspect_ratio,
        bottom_offset=bottom_offset,
        taper_delta=taper_delta,
        critical_depth=critical_depth,
        total_risk=total_risk,
        level=risk_level(total_risk),
        critical_wl=critical_wl,
        main_driver=main_driver,
        risk_items=risk_items,
    )


def build_process_window(
    base_geometry: ChannelHoleGeometry,
    grid_size: int = 35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bottom_cd_values = np.linspace(72.0, 100.0, grid_size)
    tilt_values = np.linspace(0.0, 0.60, grid_size)
    levels = np.zeros((grid_size, grid_size), dtype=int)

    for row, tilt in enumerate(tilt_values):
        for col, bottom_cd in enumerate(bottom_cd_values):
            candidate = replace(
                base_geometry,
                bottom_cd=float(bottom_cd),
                tilt_deg=float(tilt),
            )
            score = evaluate_geometry(candidate).total_risk

            if score < 0.35:
                levels[row, col] = 0
            elif score < 0.60:
                levels[row, col] = 1
            else:
                levels[row, col] = 2

    return bottom_cd_values, tilt_values, levels


# =========================================================
# 4. Full Structure Canvas
# =========================================================

class FullStructureCanvas(FigureCanvasQTAgg):
    CHANNEL_LATERAL_SCALE = 0.0024
    NOMINAL_CHANNEL_DEPTH = 8000.0

    def __init__(self, object_selected_callback):
        self.figure = Figure(figsize=(10, 7), tight_layout=True)
        super().__init__(self.figure)

        # REF와 COMP는 같은 axes에 겹치거나 한 scene 안에 배치하지 않고,
        # 독립된 3D axes/subplot으로 각각 렌더링합니다.
        self.ax = None
        self.ref_ax = None
        self.comp_ax = None
        self.structure_axes: List[object] = []

        self.object_selected_callback = object_selected_callback

        self.artist_to_object: Dict[object, str] = {}
        self.object_artists: Dict[str, List[object]] = {}
        self.default_alpha: Dict[object, float] = {}
        self.ref_group_name = "REF_GROUP_A"
        self.comp_group_name = "COMP_GROUP_B"
        self.ref_geometry: ChannelHoleGeometry | None = None
        self.comp_geometry: ChannelHoleGeometry | None = None
        self.selected_object_id: str | None = None
        self.focus_object_id: str | None = None
        self.current_view_name = "3d"

        self.mpl_connect("pick_event", self._on_pick)
        self.draw_structure()

    def _register(self, artist, object_id: str, alpha: float) -> None:
        artist.set_picker(True)
        self.artist_to_object[artist] = object_id
        self.object_artists.setdefault(object_id, []).append(artist)
        self.default_alpha[artist] = alpha

    def add_extruded_polygon(
        self,
        footprint: List[Tuple[float, float]],
        z0: float,
        z1: float,
        color: str,
        object_id: str,
        alpha: float,
        edge_color: str = "black",
        linewidth: float = 0.25,
    ) -> None:
        bottom = np.array([[x, y, z0] for x, y in footprint], dtype=float)
        top = np.array([[x, y, z1] for x, y in footprint], dtype=float)

        faces = []
        faces.append([point for point in bottom[::-1]])
        faces.append([point for point in top])

        for idx in range(len(footprint)):
            nxt = (idx + 1) % len(footprint)
            faces.append([bottom[idx], bottom[nxt], top[nxt], top[idx]])

        poly = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors=edge_color,
            linewidths=linewidth,
            alpha=alpha,
        )
        self.ax.add_collection3d(poly)
        self._register(poly, object_id, alpha)

    def add_box(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        color: str,
        object_id: str,
        alpha: float,
        edge_color: str = "black",
        linewidth: float = 0.25,
    ) -> None:
        cx, cy, cz = center
        sx, sy, sz = size

        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2

        footprint = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self.add_extruded_polygon(
            footprint=footprint,
            z0=z0,
            z1=z1,
            color=color,
            object_id=object_id,
            alpha=alpha,
            edge_color=edge_color,
            linewidth=linewidth,
        )

    def add_cutaway_box(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        color: str,
        object_id: str,
        alpha: float,
        cut_ratio_x: float = 0.28,
        cut_ratio_y: float = 0.30,
        edge_color: str = "black",
        linewidth: float = 0.25,
    ) -> None:
        """Box with one top-view corner chamfered away for cutaway-style illustration."""
        cx, cy, cz = center
        sx, sy, sz = size

        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2

        cut_x = x1 - sx * cut_ratio_x
        cut_y = y1 - sy * cut_ratio_y
        footprint = [(x0, y0), (x1, y0), (x1, cut_y), (cut_x, y1), (x0, y1)]
        self.add_extruded_polygon(
            footprint=footprint,
            z0=z0,
            z1=z1,
            color=color,
            object_id=object_id,
            alpha=alpha,
            edge_color=edge_color,
            linewidth=linewidth,
        )


    def add_y_parallel_global_cut_box(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        color: str,
        object_id: str,
        alpha: float,
        x_at_z_ref: float,
        z_ref: float,
        slope_dx_dz: float,
        edge_color: str = "black",
        linewidth: float = 0.25,
    ) -> None:
        """Box clipped by one common plane: x = x_at_z_ref + slope * (z - z_ref).

        The cut plane is parallel to Y and tilted only in X-Z.
        Multiple slabs use the same x_at_z_ref / z_ref / slope, so they look
        like one continuous control-gate stack removed by a single plane.
        """
        cx, cy, cz = center
        sx, sy, sz = size

        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2

        def clipped_x(z_value: float) -> float:
            raw = x_at_z_ref + slope_dx_dz * (z_value - z_ref)
            return float(np.clip(raw, x0 + 0.18 * sx, x1 - 0.05 * sx))

        x_bottom = clipped_x(z0)
        x_top = clipped_x(z1)

        bottom = np.array([
            [x0, y0, z0],
            [x_bottom, y0, z0],
            [x_bottom, y1, z0],
            [x0, y1, z0],
        ], dtype=float)
        top = np.array([
            [x0, y0, z1],
            [x_top, y0, z1],
            [x_top, y1, z1],
            [x0, y1, z1],
        ], dtype=float)

        faces = [
            [point for point in bottom[::-1]],
            [point for point in top],
            [bottom[0], bottom[1], top[1], top[0]],
            [bottom[1], bottom[2], top[2], top[1]],  # single shared slanted cut face
            [bottom[2], bottom[3], top[3], top[2]],
            [bottom[3], bottom[0], top[0], top[3]],
        ]

        poly = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors=edge_color,
            linewidths=linewidth,
            alpha=alpha,
        )
        self.ax.add_collection3d(poly)
        self._register(poly, object_id, alpha)

    def add_cylinder(
        self,
        x: float,
        y: float,
        z0: float,
        z1: float,
        radius: float,
        color: str,
        object_id: str,
        alpha: float,
        segments: int = 20,
    ) -> None:
        """Clickable cylinder implemented as Poly3DCollection.

        plot_surface보다 pick event가 안정적이라 모든 VIA/Channel Object에 사용합니다.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        bottom = np.column_stack([
            x + radius * np.cos(theta),
            y + radius * np.sin(theta),
            np.full_like(theta, z0),
        ])
        top = np.column_stack([
            x + radius * np.cos(theta),
            y + radius * np.sin(theta),
            np.full_like(theta, z1),
        ])

        faces = []
        for idx in range(segments):
            nxt = (idx + 1) % segments
            faces.append([bottom[idx], bottom[nxt], top[nxt], top[idx]])

        faces.append([point for point in bottom[::-1]])
        faces.append([point for point in top])

        poly = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="black",
            linewidths=0.15,
            alpha=alpha,
        )
        self.ax.add_collection3d(poly)
        self._register(poly, object_id, alpha)

    def add_measured_memory_hole(
        self,
        x: float,
        y: float,
        top_z: float,
        nominal_bottom_z: float,
        geometry: ChannelHoleGeometry,
        color: str,
        object_id: str = "MEMORY_HOLE",
        alpha: float = 0.92,
        rings: int = 18,
        segments: int = 18,
    ) -> None:
        """Render a channel wall from measured CD, depth, and tilt values.

        The architecture view compresses the vertical NAND stack and enlarges
        lateral dimensions so that nanometer-scale profile differences remain
        visible.  REF and COMP use the same conversion factors, preserving the
        direction and relative magnitude of their measured differences.
        """
        measured_depth = max(float(geometry.depth), 1.0)
        nominal_depth = self.NOMINAL_CHANNEL_DEPTH
        vertical_span = top_z - nominal_bottom_z
        bottom_z = top_z - vertical_span * measured_depth / nominal_depth
        bottom_z = float(np.clip(bottom_z, 0.16, top_z - 0.20))

        measured_z = np.linspace(0.0, measured_depth, rings)
        display_z = np.linspace(top_z, bottom_z, rings)
        measured_cd = cd_profile(measured_z, geometry)
        measured_offset = center_offset(measured_z, geometry)

        # One shared lateral conversion makes CD and tilt differences directly
        # comparable between the two panels.  Vertical scale is independently
        # compressed to fit the full stack in the architecture illustration.
        lateral_scale = self.CHANNEL_LATERAL_SCALE
        theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        ring_points = []

        for z_value, cd_value, offset_value in zip(
            display_z,
            measured_cd,
            measured_offset,
        ):
            radius = max(float(cd_value) / 2.0 * lateral_scale, 0.018)
            center_x = x + float(offset_value) * lateral_scale
            ring_points.append(np.column_stack([
                center_x + radius * np.cos(theta),
                y + radius * np.sin(theta),
                np.full_like(theta, z_value),
            ]))

        faces = []
        for ring_idx in range(rings - 1):
            lower = ring_points[ring_idx]
            upper = ring_points[ring_idx + 1]
            for segment_idx in range(segments):
                nxt = (segment_idx + 1) % segments
                faces.append([
                    lower[segment_idx],
                    lower[nxt],
                    upper[nxt],
                    upper[segment_idx],
                ])

        faces.append([point for point in ring_points[0][::-1]])
        faces.append([point for point in ring_points[-1]])

        poly = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="#5b5214",
            linewidths=0.12,
            alpha=alpha,
        )
        self.ax.add_collection3d(poly)
        self._register(poly, object_id, alpha)

    def add_label(self, x: float, y: float, z: float, text: str) -> None:
        self.ax.text(
            x,
            y,
            z,
            text,
            fontsize=8,
            ha="center",
            va="center",
            color="#222222",
        )

    def draw_xtacking_unit(
        self,
        x_offset: float,
        group_role: str,
        group_name: str,
        channel_geometry: ChannelHoleGeometry | None = None,
    ) -> None:
        """Draw one Xtacking structure instance at x_offset.

        The same object_id is registered on both REF and COMP instances, so a
        click on either side selects the shared geometry object and highlights
        both group views.
        """
        def ox(x_value: float) -> float:
            return x_value + x_offset

        x_positions = [ox(v) for v in [-1.85, -0.70, 0.45, 1.45]]
        y_positions = [-1.30, -0.40, 0.50, 1.35]

        # Bottom die: full array base and pipe connection.
        self.add_box(
            center=(ox(0), 0, 0.10),
            size=(6.0, 4.9, 0.20),
            color="#2f2f2f",
            object_id="ARRAY_CHIP",
            alpha=0.78,
        )

        self.add_box(
            center=(ox(0), 0, 0.40),
            size=(5.5, 4.4, 0.32),
            color="#65b8b8",
            object_id="PIPE_CONNECTION",
            alpha=0.74,
        )

        # Control gate / word line stack removed by ONE shared cut plane.
        cg_cut_z_ref = 0.72
        cg_cut_x_at_ref = ox(1.62)
        cg_cut_slope_dx_dz = -1.2

        for idx, z in enumerate(np.linspace(0.84, 3.30, 8)):
            color = "#61b6ac" if idx % 2 == 0 else "#86cec4"
            self.add_y_parallel_global_cut_box(
                center=(ox(0), 0, float(z)),
                size=(5.35, 4.25, 0.18),
                color=color,
                object_id="CONTROL_GATE_STACK",
                alpha=0.82,
                x_at_z_ref=cg_cut_x_at_ref,
                z_ref=cg_cut_z_ref,
                slope_dx_dz=cg_cut_slope_dx_dz,
                linewidth=0.20,
            )

        # Memory holes use group measurements when available.  All other
        # objects remain a stable architecture reference for the demo.
        channel_color = "#4f81bd" if group_role == "REF" else "#d99032"
        for x in x_positions:
            for y in y_positions:
                if channel_geometry is None:
                    self.add_cylinder(
                        x=x,
                        y=y,
                        z0=0.42,
                        z1=3.78,
                        radius=0.13,
                        color=channel_color,
                        object_id="MEMORY_HOLE",
                        alpha=0.92,
                    )
                else:
                    self.add_measured_memory_hole(
                        x=x,
                        y=y,
                        top_z=3.78,
                        nominal_bottom_z=0.42,
                        geometry=channel_geometry,
                        color=channel_color,
                    )

        self.add_box(
            center=(ox(0), 0, 3.72),
            size=(5.3, 4.2, 0.30),
            color="#2d7d42",
            object_id="SELECT_GATE",
            alpha=0.86,
        )

        self.add_box(
            center=(ox(0.55), 0.32, 4.04),
            size=(3.35, 0.55, 0.16),
            color="#8fd6a0",
            object_id="SOURCE_LINE",
            alpha=0.88,
        )
        for x in [ox(-1.15), ox(0.0), ox(1.05)]:
            self.add_cylinder(
                x=x,
                y=0.32,
                z0=3.94,
                z1=4.24,
                radius=0.11,
                color="#8fd6a0",
                object_id="SOURCE_LINE",
                alpha=0.90,
            )

        for y in [-1.30, -0.40, 0.50, 1.35]:
            self.add_box(
                center=(ox(-0.15), y, 4.35),
                size=(4.7, 0.20, 0.18),
                color="#c85858",
                object_id="BIT_LINE",
                alpha=0.90,
            )

        # Xtacking interface: Bonding VIA / metal VIA.
        via_x_positions = [ox(v) for v in [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0]]
        via_y_positions = [-1.5, -0.75, 0.0, 0.75, 1.5]

        for x in via_x_positions:
            for y in via_y_positions:
                self.add_cylinder(
                    x=x,
                    y=y,
                    z0=4.45,
                    z1=5.38,
                    radius=0.075,
                    color="#9ce39b",
                    object_id="BONDING_VIA",
                    alpha=0.92,
                )
                self.add_box(
                    center=(x, y, 4.43),
                    size=(0.24, 0.24, 0.06),
                    color="#b9f2b2",
                    object_id="BONDING_VIA",
                    alpha=0.88,
                    linewidth=0.12,
                )
                self.add_box(
                    center=(x, y, 5.41),
                    size=(0.24, 0.24, 0.06),
                    color="#b9f2b2",
                    object_id="BONDING_VIA",
                    alpha=0.88,
                    linewidth=0.12,
                )

        # Top die: Periphery / CMOS Circuit plane.
        self.add_box(
            center=(ox(0.10), 0.05, 5.72),
            size=(5.9, 4.7, 0.24),
            color="#3a3a3a",
            object_id="PERIPHERY_CIRCUIT",
            alpha=0.90,
            linewidth=0.30,
        )

        for center, size, color in [
            ((-1.80, -1.00, 5.48), (1.20, 0.72, 0.18), "#595959"),
            ((0.00, -1.00, 5.48), (1.30, 0.72, 0.18), "#505050"),
            ((1.80, -1.00, 5.48), (1.20, 0.72, 0.18), "#595959"),
            ((-1.10, 1.02, 5.48), (1.40, 0.72, 0.18), "#505050"),
            ((1.20, 1.02, 5.48), (1.50, 0.72, 0.18), "#595959"),
        ]:
            (cx, cy, cz), size_value, color_value = center, size, color
            self.add_box(
                center=(ox(cx), cy, cz),
                size=size_value,
                color=color_value,
                object_id="PERIPHERY_CIRCUIT",
                alpha=0.88,
                edge_color="#111111",
                linewidth=0.18,
            )

        self.add_label(ox(-2.95), -2.25, 5.70, "Periphery\nCircuit")
        self.add_label(ox(-2.95), -2.25, 4.95, "Bonding\nVIA")
        self.add_label(ox(-2.95), -2.25, 2.10, "3D NAND\nArray")
        self.add_label(ox(1.95), 1.85, 2.75, "Shared X-Z\ncut plane")

        self.ax.text(
            ox(0),
            -2.62,
            6.00,
            f"{group_role}\n{group_name}",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#555555", alpha=0.90),
        )

        if channel_geometry is not None:
            self.ax.text2D(
                0.02,
                0.02,
                f"MH T/M/B CD {channel_geometry.top_cd:.1f} / "
                f"{channel_geometry.middle_cd:.1f} / {channel_geometry.bottom_cd:.1f} nm\n"
                f"Depth {channel_geometry.depth:.0f} nm  ·  "
                f"Bowing {channel_geometry.bowing:.1f} nm  ·  "
                f"Tilt {channel_geometry.tilt_deg:.3f}°",
                transform=self.ax.transAxes,
                fontsize=7.2,
                ha="left",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.32",
                    facecolor="white",
                    edgecolor=channel_color,
                    alpha=0.90,
                ),
            )

    def _draw_memory_hole_focus_panel(
        self,
        ax,
        geometry: ChannelHoleGeometry,
        group_role: str,
        group_name: str,
        channel_color: str,
    ) -> None:
        """Draw two representative measured channels with local WL context."""
        self.ax = ax
        top_z = 5.15
        nominal_bottom_z = 0.70
        hole_positions = (-0.58, 0.58)

        # Only a few translucent WL plates remain in focus mode.  They provide
        # structural context without shrinking the selected Memory Holes.
        for layer_z in np.linspace(1.15, 4.70, 7):
            self.add_box(
                center=(0.0, 0.0, float(layer_z)),
                size=(3.0, 1.45, 0.10),
                color="#66b8ae",
                object_id="CONTROL_GATE_STACK",
                alpha=0.17,
                edge_color="#367970",
                linewidth=0.20,
            )

        for base_x in hole_positions:
            self.add_measured_memory_hole(
                x=base_x,
                y=0.0,
                top_z=top_z,
                nominal_bottom_z=nominal_bottom_z,
                geometry=geometry,
                color=channel_color,
                alpha=0.96,
                rings=24,
                segments=22,
            )

            measured_z = np.linspace(0.0, geometry.depth, 80)
            vertical_span = top_z - nominal_bottom_z
            bottom_z = top_z - (
                vertical_span
                * geometry.depth
                / self.NOMINAL_CHANNEL_DEPTH
            )
            bottom_z = float(np.clip(bottom_z, 0.16, top_z - 0.20))
            display_z = np.linspace(top_z, bottom_z, len(measured_z))
            display_x = (
                base_x
                + center_offset(measured_z, geometry)
                * self.CHANNEL_LATERAL_SCALE
            )
            ax.plot(
                display_x,
                np.zeros_like(display_x),
                display_z,
                linestyle=":",
                linewidth=1.3,
                color="#2d2d2d",
                alpha=0.78,
            )

        evaluation = evaluate_geometry(geometry)
        actual_bottom_z = top_z - (
            (top_z - nominal_bottom_z)
            * geometry.depth
            / self.NOMINAL_CHANNEL_DEPTH
        )
        actual_bottom_z = float(np.clip(actual_bottom_z, 0.16, top_z - 0.20))
        critical_fraction = float(np.clip(
            evaluation.critical_depth / max(geometry.depth, 1.0),
            0.0,
            1.0,
        ))
        critical_z = top_z + (actual_bottom_z - top_z) * critical_fraction

        self.add_box(
            center=(0.0, 0.0, critical_z),
            size=(2.75, 1.28, 0.045),
            color="#d62728",
            object_id="MEMORY_HOLE",
            alpha=0.18,
            edge_color="#a51f1f",
            linewidth=0.35,
        )
        ax.text(
            1.12,
            0.60,
            critical_z,
            f"Critical WL {evaluation.critical_wl}",
            color="#a51f1f",
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.set_xlim(-1.35, 1.65)
        ax.set_ylim(-0.90, 0.90)
        ax.set_zlim(0.15, 5.45)
        ax.set_box_aspect((1.05, 0.68, 1.35))
        ax.set_title(
            f"{group_role} Focus\n{group_name}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Hole pitch / tilt")
        ax.set_ylabel("Local array")
        ax.set_zlabel("Vertical stack")
        ax.grid(False)

        elev, azim = self._view_angles(self.current_view_name)
        ax.view_init(elev=elev, azim=azim)
        ax.text2D(
            0.02,
            0.02,
            f"Bottom CD {geometry.bottom_cd:.1f} nm  ·  "
            f"Tilt {geometry.tilt_deg:.3f}°\n"
            f"{evaluation.level}  ·  Critical WL {evaluation.critical_wl}",
            transform=ax.transAxes,
            fontsize=8.0,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=channel_color,
                alpha=0.92,
            ),
        )

    def draw_memory_hole_focus(self) -> None:
        if self.ref_geometry is None or self.comp_geometry is None:
            return

        self.figure.clear()
        self.artist_to_object.clear()
        self.object_artists.clear()
        self.default_alpha.clear()

        self.ref_ax = self.figure.add_subplot(1, 2, 1, projection="3d")
        self.comp_ax = self.figure.add_subplot(1, 2, 2, projection="3d")
        self.structure_axes = [self.ref_ax, self.comp_ax]

        for ax in self.structure_axes:
            ax.disable_mouse_rotation()

        self._draw_memory_hole_focus_panel(
            self.ref_ax,
            self.ref_geometry,
            "REF",
            self.ref_group_name,
            "#4f81bd",
        )
        self._draw_memory_hole_focus_panel(
            self.comp_ax,
            self.comp_geometry,
            "COMP",
            self.comp_group_name,
            "#d99032",
        )

        self.figure.suptitle(
            "Selected Object Focus · Memory Hole · 2 Representative Channels",
            fontsize=12,
            fontweight="bold",
        )
        self.highlight_object("MEMORY_HOLE")

    def _draw_generic_object_focus_panel(
        self,
        ax,
        object_id: str,
        group_role: str,
        group_name: str,
    ) -> None:
        """Draw an object-specific close-up with only essential neighbors."""
        self.ax = ax
        definition = OBJECT_REGISTRY[object_id]

        if object_id == "PERIPHERY_CIRCUIT":
            self.add_box(
                center=(0.0, 0.0, 4.05),
                size=(3.45, 2.10, 0.28),
                color="#3a3a3a",
                object_id=object_id,
                alpha=0.92,
            )
            for x, y, sx, sy in [
                (-0.95, -0.48, 0.85, 0.48),
                (0.0, -0.48, 0.70, 0.48),
                (0.95, -0.48, 0.85, 0.48),
                (-0.58, 0.48, 0.95, 0.48),
                (0.62, 0.48, 1.00, 0.48),
            ]:
                self.add_box(
                    center=(x, y, 4.35),
                    size=(sx, sy, 0.24),
                    color="#565656",
                    object_id=object_id,
                    alpha=0.94,
                )
            for x in (-0.90, 0.0, 0.90):
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=2.10,
                    z1=3.88,
                    radius=0.10,
                    color="#9ce39b",
                    object_id="BONDING_VIA",
                    alpha=0.52,
                )
            self.add_box(
                center=(0.0, 0.0, 1.95),
                size=(3.10, 1.75, 0.16),
                color="#c85858",
                object_id="BIT_LINE",
                alpha=0.32,
            )

        elif object_id == "BONDING_VIA":
            self.add_box(
                center=(0.0, 0.0, 4.45),
                size=(3.10, 1.85, 0.18),
                color="#4a4a4a",
                object_id="PERIPHERY_CIRCUIT",
                alpha=0.30,
            )
            self.add_box(
                center=(0.0, 0.0, 1.10),
                size=(3.10, 1.85, 0.18),
                color="#c85858",
                object_id="BIT_LINE",
                alpha=0.30,
            )
            for x in (-0.62, 0.62):
                for y in (-0.38, 0.38):
                    self.add_cylinder(
                        x=x,
                        y=y,
                        z0=1.25,
                        z1=4.30,
                        radius=0.13,
                        color="#83d889",
                        object_id=object_id,
                        alpha=0.95,
                        segments=24,
                    )
                    for pad_z in (1.18, 4.37):
                        self.add_box(
                            center=(x, y, pad_z),
                            size=(0.38, 0.38, 0.10),
                            color="#b9f2b2",
                            object_id=object_id,
                            alpha=0.92,
                        )

        elif object_id == "BIT_LINE":
            for y in (-0.55, 0.0, 0.55):
                self.add_box(
                    center=(0.0, y, 4.05),
                    size=(3.40, 0.18, 0.22),
                    color="#c85858",
                    object_id=object_id,
                    alpha=0.95,
                )
            for x in (-0.78, 0.78):
                for y in (-0.55, 0.0, 0.55):
                    self.add_cylinder(
                        x=x,
                        y=y,
                        z0=1.05,
                        z1=3.90,
                        radius=0.095,
                        color="#d7c73f",
                        object_id="MEMORY_HOLE",
                        alpha=0.48,
                    )
            self.add_box(
                center=(0.0, 0.0, 3.55),
                size=(3.05, 1.65, 0.16),
                color="#2d7d42",
                object_id="SELECT_GATE",
                alpha=0.25,
            )

        elif object_id == "SOURCE_LINE":
            self.add_box(
                center=(0.0, 0.0, 3.75),
                size=(3.20, 0.46, 0.24),
                color="#8fd6a0",
                object_id=object_id,
                alpha=0.96,
            )
            for x in (-0.92, 0.0, 0.92):
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=3.30,
                    z1=4.10,
                    radius=0.13,
                    color="#67be7b",
                    object_id=object_id,
                    alpha=0.96,
                    segments=24,
                )
            for x in (-0.72, 0.72):
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=0.95,
                    z1=3.25,
                    radius=0.095,
                    color="#d7c73f",
                    object_id="MEMORY_HOLE",
                    alpha=0.45,
                )
            self.add_box(
                center=(0.0, 0.0, 0.78),
                size=(2.80, 1.35, 0.18),
                color="#65b8b8",
                object_id="PIPE_CONNECTION",
                alpha=0.28,
            )

        elif object_id == "SELECT_GATE":
            self.add_box(
                center=(0.0, 0.0, 3.95),
                size=(3.15, 1.75, 0.30),
                color="#2d7d42",
                object_id=object_id,
                alpha=0.94,
            )
            for x in (-0.62, 0.62):
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=0.75,
                    z1=4.45,
                    radius=0.105,
                    color="#d7c73f",
                    object_id="MEMORY_HOLE",
                    alpha=0.62,
                )
            for layer_z in (2.20, 2.65, 3.10):
                self.add_box(
                    center=(0.0, 0.0, layer_z),
                    size=(3.00, 1.60, 0.12),
                    color="#66b8ae",
                    object_id="CONTROL_GATE_STACK",
                    alpha=0.25,
                )
            self.add_box(
                center=(0.0, 0.0, 4.45),
                size=(3.20, 0.18, 0.18),
                color="#c85858",
                object_id="BIT_LINE",
                alpha=0.25,
            )

        elif object_id == "CONTROL_GATE_STACK":
            cut_z_ref = 0.85
            cut_x_at_ref = 1.15
            for idx, layer_z in enumerate(np.linspace(1.05, 4.25, 8)):
                self.add_y_parallel_global_cut_box(
                    center=(0.0, 0.0, float(layer_z)),
                    size=(3.25, 1.75, 0.18),
                    color="#61b6ac" if idx % 2 == 0 else "#86cec4",
                    object_id=object_id,
                    alpha=0.90,
                    x_at_z_ref=cut_x_at_ref,
                    z_ref=cut_z_ref,
                    slope_dx_dz=-1.2,
                    linewidth=0.25,
                )
            for x in (-0.62, 0.62):
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=0.72,
                    z1=4.70,
                    radius=0.10,
                    color="#d7c73f",
                    object_id="MEMORY_HOLE",
                    alpha=0.66,
                )

        elif object_id == "PIPE_CONNECTION":
            self.add_box(
                center=(0.0, 0.0, 0.82),
                size=(3.20, 1.65, 0.32),
                color="#65b8b8",
                object_id=object_id,
                alpha=0.94,
            )
            for x in (-0.65, 0.65):
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=0.55,
                    z1=1.35,
                    radius=0.16,
                    color="#4aa1a4",
                    object_id=object_id,
                    alpha=0.96,
                    segments=24,
                )
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=1.20,
                    z1=4.55,
                    radius=0.095,
                    color="#d7c73f",
                    object_id="MEMORY_HOLE",
                    alpha=0.52,
                )
            self.add_box(
                center=(0.0, 0.0, 0.38),
                size=(3.45, 1.90, 0.18),
                color="#2f2f2f",
                object_id="ARRAY_CHIP",
                alpha=0.32,
            )

        elif object_id == "ARRAY_CHIP":
            self.add_box(
                center=(0.0, 0.0, 0.42),
                size=(3.55, 2.05, 0.30),
                color="#2f2f2f",
                object_id=object_id,
                alpha=0.94,
            )
            self.add_box(
                center=(0.0, 0.0, 0.78),
                size=(3.20, 1.75, 0.24),
                color="#65b8b8",
                object_id="PIPE_CONNECTION",
                alpha=0.32,
            )
            for layer_z in (1.25, 1.80, 2.35, 2.90, 3.45):
                self.add_box(
                    center=(0.0, 0.0, layer_z),
                    size=(3.05, 1.65, 0.12),
                    color="#66b8ae",
                    object_id="CONTROL_GATE_STACK",
                    alpha=0.24,
                )
            for x in (-0.70, 0.70):
                self.add_cylinder(
                    x=x,
                    y=0.0,
                    z0=0.82,
                    z1=3.85,
                    radius=0.095,
                    color="#d7c73f",
                    object_id="MEMORY_HOLE",
                    alpha=0.48,
                )

        selected_color = {
            "PERIPHERY_CIRCUIT": "#3a3a3a",
            "BONDING_VIA": "#67be7b",
            "BIT_LINE": "#c85858",
            "SOURCE_LINE": "#67be7b",
            "SELECT_GATE": "#2d7d42",
            "CONTROL_GATE_STACK": "#61b6ac",
            "PIPE_CONNECTION": "#4aa1a4",
            "ARRAY_CHIP": "#2f2f2f",
        }[object_id]

        ax.set_xlim(-1.90, 1.90)
        ax.set_ylim(-1.20, 1.20)
        ax.set_zlim(0.10, 5.05)
        ax.set_box_aspect((1.05, 0.72, 1.25))
        ax.set_title(
            f"{group_role} Focus\n{group_name}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Local X")
        ax.set_ylabel("Local Y")
        ax.set_zlabel("Vertical stack")
        ax.grid(False)
        elev, azim = self._view_angles(self.current_view_name)
        ax.view_init(elev=elev, azim=azim)
        ax.text2D(
            0.02,
            0.02,
            f"{definition.name}\n{definition.primary_process}",
            transform=ax.transAxes,
            fontsize=8.0,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=selected_color,
                alpha=0.92,
            ),
        )

    def draw_generic_object_focus(self, object_id: str) -> None:
        self.figure.clear()
        self.artist_to_object.clear()
        self.object_artists.clear()
        self.default_alpha.clear()

        self.ref_ax = self.figure.add_subplot(1, 2, 1, projection="3d")
        self.comp_ax = self.figure.add_subplot(1, 2, 2, projection="3d")
        self.structure_axes = [self.ref_ax, self.comp_ax]

        for ax in self.structure_axes:
            ax.disable_mouse_rotation()

        self._draw_generic_object_focus_panel(
            self.ref_ax,
            object_id,
            "REF",
            self.ref_group_name,
        )
        self._draw_generic_object_focus_panel(
            self.comp_ax,
            object_id,
            "COMP",
            self.comp_group_name,
        )

        definition = OBJECT_REGISTRY[object_id]
        self.figure.suptitle(
            f"Selected Object Focus · {definition.name}",
            fontsize=12,
            fontweight="bold",
        )
        self.highlight_object(object_id)

    def _format_structure_axis(self, ax, panel_title: str) -> None:
        ax.set_xlim(-3.25, 3.25)
        ax.set_ylim(-2.85, 2.85)
        ax.set_zlim(0.0, 6.10)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Vertical Stack")
        ax.set_box_aspect((1.20, 1.0, 1.12))
        ax.grid(False)

        elev, azim = self._view_angles(self.current_view_name)
        ax.view_init(elev=elev, azim=azim)

    def draw_structure(self) -> None:
        if (
            self.focus_object_id == "MEMORY_HOLE"
            and self.ref_geometry is not None
            and self.comp_geometry is not None
        ):
            self.draw_memory_hole_focus()
            return
        if (
            self.focus_object_id in OBJECT_REGISTRY
            and self.focus_object_id != "MEMORY_HOLE"
        ):
            self.draw_generic_object_focus(self.focus_object_id)
            return

        self.figure.clear()
        self.artist_to_object.clear()
        self.object_artists.clear()
        self.default_alpha.clear()

        self.ref_ax = self.figure.add_subplot(1, 2, 1, projection="3d")
        self.comp_ax = self.figure.add_subplot(1, 2, 2, projection="3d")
        self.structure_axes = [self.ref_ax, self.comp_ax]

        for ax in self.structure_axes:
            ax.disable_mouse_rotation()

        # REF structure: independent 3D axes.
        self.ax = self.ref_ax
        self.draw_xtacking_unit(
            0.0,
            "REF",
            self.ref_group_name,
            self.ref_geometry,
        )
        self._format_structure_axis(self.ref_ax, f"REF Structure\n{self.ref_group_name}")

        # COMP structure: independent 3D axes.
        self.ax = self.comp_ax
        self.draw_xtacking_unit(
            0.0,
            "COMP",
            self.comp_group_name,
            self.comp_geometry,
        )
        self._format_structure_axis(self.comp_ax, f"COMP Structure\n{self.comp_group_name}")

        self.figure.suptitle(
            "Independent REF / COMP Xtacking Structures",
            fontsize=12,
            fontweight="bold",
        )

        if self.selected_object_id:
            self.highlight_object(self.selected_object_id)
        else:
            self.draw_idle()

    def _view_angles(self, view_name: str) -> Tuple[float, float]:
        presets = {
            "3d": (23, -56),
            "side": (0, -90),
            "front": (0, 0),
            "top": (90, -90),
        }
        return presets.get(view_name, presets["3d"])

    def set_groups(
        self,
        ref_group_name: str,
        comp_group_name: str,
        ref_geometry: ChannelHoleGeometry | None = None,
        comp_geometry: ChannelHoleGeometry | None = None,
    ) -> None:
        self.ref_group_name = ref_group_name
        self.comp_group_name = comp_group_name
        self.ref_geometry = ref_geometry
        self.comp_geometry = comp_geometry
        self.draw_structure()

    def select_object(self, object_id: str, focus: bool = True) -> None:
        """Select an object and enter focus mode when that view is supported."""
        self.selected_object_id = object_id
        desired_focus = object_id if focus and object_id in OBJECT_REGISTRY else None

        if self.focus_object_id != desired_focus:
            self.focus_object_id = desired_focus
            self.draw_structure()
        else:
            self.highlight_object(object_id)

    def focus_selected_object(self) -> bool:
        if self.selected_object_id not in OBJECT_REGISTRY:
            return False
        if (
            self.selected_object_id == "MEMORY_HOLE"
            and (self.ref_geometry is None or self.comp_geometry is None)
        ):
            return False

        self.focus_object_id = self.selected_object_id
        self.draw_structure()
        return True

    def show_full_structure(self) -> None:
        self.focus_object_id = None
        self.draw_structure()

    def set_view(self, view_name: str) -> None:
        self.current_view_name = view_name
        elev, azim = self._view_angles(view_name)

        for ax in self.structure_axes:
            ax.view_init(elev=elev, azim=azim)

        self.draw_idle()

    def _on_pick(self, event) -> None:
        object_id = self.artist_to_object.get(event.artist)
        if object_id:
            self.highlight_object(object_id)
            self.object_selected_callback(object_id)

    def highlight_object(self, object_id: str) -> None:
        self.selected_object_id = object_id
        for current_id, artists in self.object_artists.items():
            selected = current_id == object_id

            for artist in artists:
                base_alpha = self.default_alpha[artist]
                artist.set_alpha(
                    min(1.0, base_alpha + 0.08)
                    if selected
                    else 0.10
                )

        self.draw_idle()

    def reset_highlight(self) -> None:
        for artist, alpha in self.default_alpha.items():
            artist.set_alpha(alpha)
        self.draw_idle()


# =========================================================
# 5. Object Ontology Canvas
# =========================================================

class OntologyGraphCanvas(FigureCanvasQTAgg):
    """Visual ontology graph for the selected geometry object.

    This intentionally avoids an external graph library so the app still only
    depends on numpy, pandas, matplotlib, and PySide6.
    """

    def __init__(self, object_selected_callback):
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        super().__init__(self.figure)
        self.object_selected_callback = object_selected_callback
        self.artist_to_object: Dict[object, str] = {}
        self.mpl_connect("pick_event", self._on_pick)

    def _register_pickable_object(self, artist, object_id: str) -> None:
        artist.set_picker(True)
        self.artist_to_object[artist] = object_id

    def _on_pick(self, event) -> None:
        object_id = self.artist_to_object.get(event.artist)
        if object_id:
            self.object_selected_callback(object_id)

    def _node_style(self, kind: str) -> dict:
        styles = {
            "selected": dict(boxstyle="round,pad=0.50", facecolor="#fff2a8", edgecolor="#1f1f1f", linewidth=1.8),
            "object": dict(boxstyle="round,pad=0.38", facecolor="#d9ecff", edgecolor="#2f5f8f", linewidth=1.2),
            "process": dict(boxstyle="round,pad=0.35", facecolor="#eadcff", edgecolor="#6e4aa8", linewidth=1.1),
            "measurement": dict(boxstyle="round,pad=0.25", facecolor="#e4f7e4", edgecolor="#387d3c", linewidth=1.0),
            "parameter": dict(boxstyle="round,pad=0.25", facecolor="#fff0d6", edgecolor="#9a6a16", linewidth=1.0),
            "risk": dict(boxstyle="round,pad=0.25", facecolor="#ffe3e0", edgecolor="#a33b32", linewidth=1.0),
            "window": dict(boxstyle="round,pad=0.25", facecolor="#e9e9e9", edgecolor="#666666", linewidth=1.0),
        }
        return styles.get(kind, styles["object"])

    def _draw_node(
        self,
        ax,
        x: float,
        y: float,
        title: str,
        subtitle: str = "",
        kind: str = "object",
        object_id: str | None = None,
        fontsize: int = 8,
    ):
        label = title if not subtitle else f"{title}\n{subtitle}"
        artist = ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            bbox=self._node_style(kind),
            zorder=3,
        )
        if object_id is not None:
            self._register_pickable_object(artist, object_id)
        return artist

    def _draw_edge(
        self,
        ax,
        start: Tuple[float, float],
        end: Tuple[float, float],
        label: str = "",
        linewidth: float = 0.9,
    ) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="-",
                linewidth=linewidth,
                color="#777777",
                shrinkA=18,
                shrinkB=18,
                alpha=0.70,
            ),
            zorder=1,
        )
        if label:
            mx = (start[0] + end[0]) / 2.0
            my = (start[1] + end[1]) / 2.0
            ax.text(
                mx,
                my,
                label,
                fontsize=6.6,
                color="#555555",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.72),
                zorder=2,
            )

    def _spread_y(self, count: int, top: float, bottom: float) -> List[float]:
        if count <= 0:
            return []
        if count == 1:
            return [(top + bottom) / 2.0]
        return [float(v) for v in np.linspace(top, bottom, count)]

    def draw_object_graph(self, selected_object_id: str) -> None:
        self.figure.clear()
        self.artist_to_object.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.set_xlim(-5.0, 5.2)
        ax.set_ylim(-3.25, 3.25)

        definition = OBJECT_REGISTRY[selected_object_id]
        selected_pos = (0.0, 0.05)

        ax.set_title(
            "Object Ontology Graph: Structure ↔ Measurement ↔ Geometry ↔ Risk",
            fontsize=12,
            pad=12,
        )

        self._draw_node(
            ax,
            *selected_pos,
            title=definition.name,
            subtitle=f"[{definition.object_id}]\n{definition.category}",
            kind="selected",
            object_id=selected_object_id,
            fontsize=9,
        )

        # Architecture / structural neighbors on the left.
        neighbors = get_ontology_neighbors(selected_object_id)
        neighbor_ys = self._spread_y(len(neighbors), 2.35, -2.05)
        for (neighbor_id, relation), y in zip(neighbors, neighbor_ys):
            neighbor = OBJECT_REGISTRY[neighbor_id]
            pos = (-3.35, y)
            self._draw_node(
                ax,
                *pos,
                title=neighbor.name,
                subtitle=f"[{neighbor_id}]",
                kind="object",
                object_id=neighbor_id,
                fontsize=7.4,
            )
            self._draw_edge(ax, pos, selected_pos, relation)

        # Process node at the top.
        process_pos = (0.0, 2.72)
        self._draw_node(
            ax,
            *process_pos,
            title="Primary Process",
            subtitle=definition.primary_process,
            kind="process",
            fontsize=8,
        )
        self._draw_edge(ax, selected_pos, process_pos, "formed by")

        # Measurement → Geometry Parameter mapping on the right.
        measurements = list(definition.measurement_keys)
        parameters = list(definition.geometry_parameters)
        max_rows = max(len(measurements), len(parameters))
        ys = self._spread_y(max_rows, 2.18, -0.82)

        measurement_x = 1.85
        parameter_x = 3.95
        for idx, y in enumerate(ys):
            measurement = measurements[idx] if idx < len(measurements) else ""
            parameter = parameters[idx] if idx < len(parameters) else ""

            if measurement:
                self._draw_node(
                    ax,
                    measurement_x,
                    y,
                    title=measurement,
                    kind="measurement",
                    fontsize=7.2,
                )
                self._draw_edge(ax, selected_pos, (measurement_x, y), "measured by", linewidth=0.7)

            if parameter:
                self._draw_node(
                    ax,
                    parameter_x,
                    y,
                    title=parameter,
                    kind="parameter",
                    fontsize=7.2,
                )

            if measurement and parameter:
                self._draw_edge(ax, (measurement_x, y), (parameter_x, y), "maps to", linewidth=0.8)
            elif parameter:
                self._draw_edge(ax, selected_pos, (parameter_x, y), "has parameter", linewidth=0.7)

        # Risks and process windows at the bottom.
        risk_ys = self._spread_y(len(definition.risks), -1.25, -2.75)
        for risk, y in zip(definition.risks, risk_ys):
            pos = (-0.95, y)
            self._draw_node(ax, *pos, title=risk, kind="risk", fontsize=7.2)
            self._draw_edge(ax, selected_pos, pos, "drives risk", linewidth=0.75)

        window_ys = self._spread_y(len(definition.process_windows), -1.25, -2.75)
        for window, y in zip(definition.process_windows, window_ys):
            pos = (2.35, y)
            self._draw_node(ax, *pos, title=window, kind="window", fontsize=7.2)
            self._draw_edge(ax, selected_pos, pos, "evaluated as", linewidth=0.75)

        # Small legend.
        legend_items = [
            ("Object", "object"),
            ("Measurement", "measurement"),
            ("Parameter", "parameter"),
            ("Risk", "risk"),
            ("Window", "window"),
        ]
        for idx, (label, kind) in enumerate(legend_items):
            self._draw_node(
                ax,
                -4.35 + idx * 1.05,
                -3.07,
                title=label,
                kind=kind,
                fontsize=6.4,
            )

        ax.text(
            4.85,
            -3.08,
            "Object nodes are clickable",
            ha="right",
            va="center",
            fontsize=7,
            color="#555555",
        )

        self.draw_idle()


# =========================================================
# 5. Comparison Canvas
# =========================================================

class ObjectComparisonCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.figure)

    def draw_empty(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=12,
            wrap=True,
        )
        self.draw_idle()

    def draw_memory_hole_comparison(
        self,
        ref_geometry: ChannelHoleGeometry,
        comp_geometry: ChannelHoleGeometry,
        ref_eval: GeometryEvaluation,
        comp_eval: GeometryEvaluation,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        style_map = [
            (ref_geometry, "--", "#1f77b4", f"REF profile: {ref_geometry.group_name}"),
            (comp_geometry, "-", "#d62728", f"COMP profile: {comp_geometry.group_name}"),
        ]

        for geometry, style, color, label in style_map:
            z = np.linspace(0.0, geometry.depth, 350)
            cd = cd_profile(z, geometry)
            radius = cd / 2.0
            offset = center_offset(z, geometry)

            ax.plot(
                offset - radius,
                z,
                style,
                color=color,
                linewidth=2.0,
                label=label,
            )
            ax.plot(
                offset + radius,
                z,
                style,
                color=color,
                linewidth=2.0,
            )

        ax.axhline(
            comp_eval.critical_depth,
            color="#333333",
            linestyle=":",
            linewidth=1.6,
            label=f"Critical WL {comp_eval.critical_wl}: COMP minimum-CD zone",
        )

        ax.text(
            0.02,
            0.02,
            "Line guide\n"
            "Blue dashed = REF channel wall profile\n"
            "Red solid = COMP channel wall profile\n"
            "Black dotted = critical WL / minimum-CD location",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#777777", alpha=0.88),
        )

        ax.set_title("Selected Object Geometry Comparison")
        ax.set_xlabel("Horizontal position (nm)")
        ax.set_ylabel("Depth (nm)")
        ax.invert_yaxis()
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=8)

        self.draw_idle()


class ProcessWindowCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.figure)

    def draw_empty(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, wrap=True)
        self.draw_idle()

    def draw_memory_hole_window(
        self,
        ref_geometry: ChannelHoleGeometry,
        comp_geometry: ChannelHoleGeometry,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x_values, y_values, levels = build_process_window(comp_geometry)

        safe_color = "#b7e4c7"
        caution_color = "#ffe08a"
        risk_color = "#f28b82"
        cmap = ListedColormap([safe_color, caution_color, risk_color])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        ax.imshow(
            levels,
            origin="lower",
            aspect="auto",
            extent=[
                x_values[0],
                x_values[-1],
                y_values[0],
                y_values[-1],
            ],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        ax.scatter(
            ref_geometry.bottom_cd,
            ref_geometry.tilt_deg,
            s=90,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            label="REF point",
        )
        ax.scatter(
            comp_geometry.bottom_cd,
            comp_geometry.tilt_deg,
            s=100,
            marker="X",
            facecolors="black",
            edgecolors="black",
            linewidths=1.0,
            label="COMP point",
        )

        ax.annotate(
            "",
            xy=(comp_geometry.bottom_cd, comp_geometry.tilt_deg),
            xytext=(ref_geometry.bottom_cd, ref_geometry.tilt_deg),
            arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#333333"),
        )

        legend_handles = [
            Patch(facecolor=safe_color, edgecolor="#777777", label="Green: SAFE, total risk < 0.35"),
            Patch(facecolor=caution_color, edgecolor="#777777", label="Yellow: CAUTION, 0.35–0.60"),
            Patch(facecolor=risk_color, edgecolor="#777777", label="Red: RISK, total risk ≥ 0.60"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markersize=7, linewidth=0, label="REF group position"),
            Line2D([0], [0], marker="X", color="black", markerfacecolor="black", markersize=7, linewidth=0, label="COMP group position"),
            Line2D([0], [0], color="#333333", linewidth=1.6, label="REF → COMP movement"),
        ]

        ax.text(
            0.02,
            0.02,
            "Window guide\n"
            "X-axis = Bottom CD margin\n"
            "Y-axis = Tilt margin\n"
            "Arrow = REF → COMP movement",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#777777", alpha=0.88),
        )

        ax.set_title("Major Process Window: Bottom CD × Tilt")
        ax.set_xlabel("Bottom CD (nm)")
        ax.set_ylabel("Tilt (deg)")
        ax.grid(alpha=0.18)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.92)

        self.draw_idle()


# =========================================================
# 6. Main Application
# =========================================================

class IntegratedEvaluator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TCAT / Xtacking Integrated Geometry Object Evaluator")
        self.resize(1750, 980)

        self.data = generate_mock_data()
        self.selected_object_id = "MEMORY_HOLE"

        self._build_ui()
        self.select_object("MEMORY_HOLE", focus_structure=False)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(main_splitter)

        # ---------------------------------------------
        # Left controller
        # ---------------------------------------------
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(380)
        left_layout = QVBoxLayout(left)

        title = QLabel("Geometry Object Evaluation")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        left_layout.addWidget(title)

        description = QLabel(
            "Xtacking-style 구조에서 Object를 선택한 후 "
            "REF/COMP 그룹을 지정하고 Evaluate를 실행합니다."
        )
        description.setStyleSheet("color: #666666;")
        left_layout.addWidget(description)

        object_group = QGroupBox("1. Geometry Object")
        object_layout = QVBoxLayout(object_group)

        self.object_list = QListWidget()
        for object_id, definition in OBJECT_REGISTRY.items():
            flag = "  ✓" if definition.analysis_supported else ""
            self.object_list.addItem(f"{definition.name} [{object_id}]{flag}")
        self.object_list.currentRowChanged.connect(self.on_object_row_changed)
        self.object_list.itemClicked.connect(self.on_object_item_clicked)
        object_layout.addWidget(self.object_list)
        left_layout.addWidget(object_group)

        group_box = QGroupBox("2. REF / COMP Group")
        group_layout = QVBoxLayout(group_box)

        group_layout.addWidget(QLabel("Reference group"))
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(sorted(self.data["group"].unique()))
        self.ref_combo.setCurrentText("REF_GROUP_A")
        group_layout.addWidget(self.ref_combo)

        group_layout.addWidget(QLabel("Comparison group"))
        self.comp_combo = QComboBox()
        self.comp_combo.addItems(sorted(self.data["group"].unique()))
        self.comp_combo.setCurrentText("COMP_GROUP_B")
        group_layout.addWidget(self.comp_combo)

        left_layout.addWidget(group_box)

        evaluate_button = QPushButton("3. Evaluate Selected Object")
        evaluate_button.setMinimumHeight(44)
        evaluate_button.setStyleSheet("font-weight: bold;")
        evaluate_button.clicked.connect(self.evaluate_selected_object)
        left_layout.addWidget(evaluate_button)

        regenerate_button = QPushButton("Regenerate Mock Data")
        regenerate_button.clicked.connect(self.regenerate_mock_data)
        left_layout.addWidget(regenerate_button)

        view_group = QGroupBox("3D View Preset")
        view_layout = QVBoxLayout(view_group)

        view_row_1 = QHBoxLayout()
        view_row_2 = QHBoxLayout()
        view_row_3 = QHBoxLayout()

        self.view_3d_button = QPushButton("3D View")
        self.view_side_button = QPushButton("Side View")
        self.view_front_button = QPushButton("Front View")
        self.view_top_button = QPushButton("Top View")
        self.full_structure_button = QPushButton("Full Structure")
        self.focus_selected_button = QPushButton("Focus Selected")

        self.view_3d_button.clicked.connect(lambda: self.set_structure_view("3d"))
        self.view_side_button.clicked.connect(lambda: self.set_structure_view("side"))
        self.view_front_button.clicked.connect(lambda: self.set_structure_view("front"))
        self.view_top_button.clicked.connect(lambda: self.set_structure_view("top"))
        self.full_structure_button.clicked.connect(self.show_full_structure)
        self.focus_selected_button.clicked.connect(self.focus_selected_object)

        view_row_1.addWidget(self.view_3d_button)
        view_row_1.addWidget(self.view_side_button)
        view_row_2.addWidget(self.view_front_button)
        view_row_2.addWidget(self.view_top_button)
        view_row_3.addWidget(self.full_structure_button)
        view_row_3.addWidget(self.focus_selected_button)

        view_layout.addLayout(view_row_1)
        view_layout.addLayout(view_row_2)
        view_layout.addLayout(view_row_3)
        left_layout.addWidget(view_group)

        object_info_group = QGroupBox("Selected Object")
        object_info_layout = QVBoxLayout(object_info_group)

        self.selected_object_label = QLabel()
        self.selected_object_label.setStyleSheet(
            "font-size: 17px; font-weight: bold;"
        )
        object_info_layout.addWidget(self.selected_object_label)

        self.object_description = QTextEdit()
        self.object_description.setReadOnly(True)
        self.object_description.setMaximumHeight(190)
        object_info_layout.addWidget(self.object_description)

        left_layout.addWidget(object_info_group)
        left_layout.addStretch(1)

        main_splitter.addWidget(left)

        # ---------------------------------------------
        # Center: full 3D and comparison
        # ---------------------------------------------
        center_tabs = QTabWidget()

        self.structure_canvas = FullStructureCanvas(self.select_object)
        self.structure_canvas.set_groups(
            self.ref_combo.currentText(),
            self.comp_combo.currentText(),
            group_to_channel_geometry(self.data, self.ref_combo.currentText()),
            group_to_channel_geometry(self.data, self.comp_combo.currentText()),
        )
        self.ref_combo.currentTextChanged.connect(self.on_group_selection_changed)
        self.comp_combo.currentTextChanged.connect(self.on_group_selection_changed)
        center_tabs.addTab(self.structure_canvas, "REF/COMP 3D Structure")

        self.ontology_canvas = OntologyGraphCanvas(self.select_object)
        center_tabs.addTab(self.ontology_canvas, "Object Ontology")

        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)

        comparison_splitter = QSplitter(Qt.Orientation.Vertical)

        self.comparison_canvas = ObjectComparisonCanvas()
        comparison_splitter.addWidget(self.comparison_canvas)

        self.window_canvas = ProcessWindowCanvas()
        comparison_splitter.addWidget(self.window_canvas)

        comparison_splitter.setSizes([460, 430])
        comparison_layout.addWidget(comparison_splitter)

        center_tabs.addTab(comparison_tab, "Object Evaluation")

        self.center_tabs = center_tabs
        main_splitter.addWidget(center_tabs)

        # ---------------------------------------------
        # Right result panel
        # ---------------------------------------------
        right = QWidget()
        right.setMinimumWidth(390)
        right.setMaximumWidth(520)
        right_layout = QVBoxLayout(right)

        decision_group = QGroupBox("Decision Summary")
        decision_layout = QVBoxLayout(decision_group)

        self.summary_status_label = QLabel("NOT EVALUATED")
        self.summary_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_status_label.setMinimumHeight(54)
        self.summary_status_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #555555; "
            "background: #eeeeee; border: 1px solid #bdbdbd; border-radius: 6px;"
        )
        decision_layout.addWidget(self.summary_status_label)

        self.interpretation_text = QTextEdit()
        self.interpretation_text.setReadOnly(True)
        self.interpretation_text.setMaximumHeight(250)
        decision_layout.addWidget(self.interpretation_text)
        right_layout.addWidget(decision_group)

        self.detail_toggle_button = QPushButton("Show Measure / Risk Details")
        self.detail_toggle_button.setCheckable(True)
        self.detail_toggle_button.toggled.connect(self.on_detail_toggled)
        right_layout.addWidget(self.detail_toggle_button)

        self.details_tabs = QTabWidget()

        mapping_tab = QWidget()
        mapping_layout = QVBoxLayout(mapping_tab)
        self.mapping_table = QTableWidget(0, 5)
        self.mapping_table.setHorizontalHeaderLabels(
            ["Measurement", "Parameter", "REF", "COMP", "Delta"]
        )
        mapping_layout.addWidget(self.mapping_table)
        self.details_tabs.addTab(mapping_tab, "Measure Mapping")

        risk_tab = QWidget()
        risk_layout = QVBoxLayout(risk_tab)
        self.risk_table = QTableWidget(0, 4)
        self.risk_table.setHorizontalHeaderLabels(
            ["Risk", "REF", "COMP", "COMP Level"]
        )
        risk_layout.addWidget(self.risk_table)
        self.details_tabs.addTab(risk_tab, "Risk Breakdown")

        lot_tab = QWidget()
        lot_layout = QVBoxLayout(lot_tab)
        self.lot_table = QTableWidget()
        lot_layout.addWidget(self.lot_table)
        self.details_tabs.addTab(lot_tab, "Lot / WF Data")

        self.details_tabs.setVisible(False)
        right_layout.addWidget(self.details_tabs, 1)
        right_layout.addStretch(1)

        main_splitter.addWidget(right)
        main_splitter.setSizes([340, 850, 570])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)

    def object_ids(self) -> List[str]:
        return list(OBJECT_REGISTRY.keys())

    def on_detail_toggled(self, visible: bool) -> None:
        self.details_tabs.setVisible(visible)
        self.detail_toggle_button.setText(
            "Hide Measure / Risk Details"
            if visible
            else "Show Measure / Risk Details"
        )

    def set_summary_status(self, status: str) -> None:
        palette = {
            "SAFE": ("#176b38", "#dff3e5", "#75b98b"),
            "CAUTION": ("#765400", "#fff0bd", "#d4ad3f"),
            "RISK": ("#9f241d", "#fde0dd", "#d8746d"),
            "MAPPING ONLY": ("#4f4f4f", "#eeeeee", "#bdbdbd"),
            "RE-EVALUATE": ("#655000", "#fff6d8", "#d8bf69"),
            "NOT EVALUATED": ("#555555", "#eeeeee", "#bdbdbd"),
        }
        foreground, background, border = palette.get(
            status,
            palette["NOT EVALUATED"],
        )
        self.summary_status_label.setText(status)
        self.summary_status_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; "
            f"color: {foreground}; background: {background}; "
            f"border: 1px solid {border}; border-radius: 6px;"
        )

    def on_object_row_changed(self, row: int) -> None:
        ids = self.object_ids()
        if 0 <= row < len(ids):
            self.select_object(ids[row])

    def on_object_item_clicked(self, item) -> None:
        """Focus an already-selected row, which emits no row-change signal."""
        row = self.object_list.row(item)
        ids = self.object_ids()
        if not 0 <= row < len(ids):
            return

        object_id = ids[row]
        if self.structure_canvas.focus_object_id != object_id:
            self.select_object(object_id)

    def select_object(
        self,
        object_id: str,
        focus_structure: bool = True,
    ) -> None:
        previous_object_id = self.selected_object_id
        self.selected_object_id = object_id
        definition = OBJECT_REGISTRY[object_id]

        self.structure_canvas.select_object(object_id, focus=focus_structure)
        if focus_structure:
            self.center_tabs.setCurrentIndex(0)
        self.ontology_canvas.draw_object_graph(object_id)

        support_text = "REF/COMP 분석 지원" if definition.analysis_supported else "구조 탐색 / Mapping 정의"
        self.selected_object_label.setText(
            f"{definition.name}\n{definition.primary_process}"
        )
        self.object_description.setPlainText(
            self.format_object_description(definition, support_text)
        )

        self.update_object_measurement_table(definition)
        self.update_lot_table(
            self.ref_combo.currentText(),
            self.comp_combo.currentText(),
            definition.measurement_keys,
        )

        row = self.object_ids().index(object_id)
        self.object_list.blockSignals(True)
        self.object_list.setCurrentRow(row)
        self.object_list.blockSignals(False)

        if not definition.analysis_supported:
            self.comparison_canvas.draw_empty(
                f"{definition.name}\n\n현재 MVP에서는 3D Object 탐색과 향후 계측 Mapping 정의를 표시합니다.\n"
                "REF/COMP 정량 평가는 Memory Hole부터 연결되어 있습니다."
            )
            self.window_canvas.draw_empty(
                f"{definition.name}\nProcess Window 확장 예정"
            )
            self.clear_risk_table()
            self.set_summary_status("MAPPING ONLY")
            self.interpretation_text.setPlainText(
                f"Selected Object: {definition.name}\n\n"
                "현재 데모에서는 구조와 Measure Mapping만 제공합니다.\n"
                "Memory Hole을 선택하면 REF/COMP Risk 평가를 실행할 수 있습니다."
            )
        else:
            result_is_current = (
                previous_object_id == object_id
                and self.summary_status_label.text() in {"SAFE", "CAUTION", "RISK"}
            )
            if not result_is_current:
                self.clear_risk_table()
                self.set_summary_status("NOT EVALUATED")
                self.interpretation_text.setPlainText(
                    "REF/COMP의 Measure 기반 3D 형상은 현재 선택에 맞춰 표시됩니다.\n\n"
                    "Evaluate를 실행하면 상태, REF 차이, 주요 원인과 위험 위치를 요약합니다."
                )

    def format_object_description(self, definition: ObjectDefinition, support_text: str) -> str:
        return (
            f"설명:\n{definition.description}\n\n"
            f"상태:\n{support_text}\n\n"
            "자동 연결 계측:\n- "
            + "\n- ".join(definition.measurement_keys)
            + "\n\nGeometry Parameter:\n- "
            + "\n- ".join(definition.geometry_parameters)
            + "\n\n주요 Risk:\n- "
            + "\n- ".join(definition.risks)
            + "\n\nProcess Window:\n- "
            + "\n- ".join(definition.process_windows)
        )

    def evaluate_selected_object(self) -> None:
        definition = OBJECT_REGISTRY[self.selected_object_id]

        if not definition.analysis_supported:
            QMessageBox.information(
                self,
                "MVP 안내",
                f"{definition.name}은 현재 구조 탐색과 Mapping 정의만 지원합니다. "
                "Memory Hole을 선택해 REF/COMP 평가를 실행해 주세요.",
            )
            return

        ref_group = self.ref_combo.currentText()
        comp_group = self.comp_combo.currentText()

        if ref_group == comp_group:
            QMessageBox.warning(
                self,
                "그룹 선택 확인",
                "REF와 COMP 그룹이 동일합니다. 서로 다른 그룹을 선택해 주세요.",
            )
            return

        ref_geometry = group_to_channel_geometry(self.data, ref_group)
        comp_geometry = group_to_channel_geometry(self.data, comp_group)

        ref_eval = evaluate_geometry(ref_geometry)
        comp_eval = evaluate_geometry(comp_geometry)

        self.update_mapping_table(ref_geometry, comp_geometry)
        self.update_risk_table(ref_eval, comp_eval)
        self.update_lot_table(ref_group, comp_group)

        self.comparison_canvas.draw_memory_hole_comparison(
            ref_geometry,
            comp_geometry,
            ref_eval,
            comp_eval,
        )
        self.window_canvas.draw_memory_hole_window(
            ref_geometry,
            comp_geometry,
        )

        self.interpretation_text.setPlainText(
            self.build_interpretation(
                ref_geometry,
                comp_geometry,
                ref_eval,
                comp_eval,
            )
        )
        self.set_summary_status(comp_eval.level)

        self.center_tabs.setCurrentIndex(2)

    def _group_mean(self, group_name: str, measurement_key: str) -> float | None:
        if measurement_key not in self.data.columns:
            return None
        group_df = self.data[self.data["group"] == group_name]
        if group_df.empty:
            return None
        value = group_df[measurement_key].mean()
        if pd.isna(value):
            return None
        return float(value)

    def _format_value(self, value: float | None, signed: bool = False) -> str:
        if value is None:
            return "N/A"
        abs_value = abs(value)
        if abs_value >= 1000:
            formatted = f"{value:.1f}"
        elif abs_value >= 10:
            formatted = f"{value:.2f}"
        else:
            formatted = f"{value:.3f}"
        if signed and value > 0:
            return "+" + formatted
        return formatted

    def update_object_measurement_table(self, definition: ObjectDefinition) -> None:
        ref_group = self.ref_combo.currentText()
        comp_group = self.comp_combo.currentText()

        self.mapping_table.setHorizontalHeaderLabels(
            ["Measurement", "Parameter", f"REF\n{ref_group}", f"COMP\n{comp_group}", "Delta"]
        )

        row_count = max(len(definition.measurement_keys), len(definition.geometry_parameters))
        self.mapping_table.setRowCount(row_count)

        for row_idx in range(row_count):
            measurement = definition.measurement_keys[row_idx] if row_idx < len(definition.measurement_keys) else "-"
            parameter = definition.geometry_parameters[row_idx] if row_idx < len(definition.geometry_parameters) else "-"

            ref_value = self._group_mean(ref_group, measurement) if measurement != "-" else None
            comp_value = self._group_mean(comp_group, measurement) if measurement != "-" else None
            delta = comp_value - ref_value if ref_value is not None and comp_value is not None else None

            values = [
                measurement,
                parameter,
                self._format_value(ref_value),
                self._format_value(comp_value),
                self._format_value(delta, signed=True),
            ]

            for col_idx, value in enumerate(values):
                self.mapping_table.setItem(row_idx, col_idx, QTableWidgetItem(value))

        self.mapping_table.resizeColumnsToContents()

    def update_definition_mapping_table(self, definition: ObjectDefinition) -> None:
        self.update_object_measurement_table(definition)

    def update_mapping_table(
        self,
        ref_geometry: ChannelHoleGeometry,
        comp_geometry: ChannelHoleGeometry,
    ) -> None:
        self.mapping_table.setHorizontalHeaderLabels(
            [
                "Measurement",
                "Parameter",
                f"REF\n{ref_geometry.group_name}",
                f"COMP\n{comp_geometry.group_name}",
                "Delta",
            ]
        )

        rows = list(MEASUREMENT_TO_PARAMETER.items())
        self.mapping_table.setRowCount(len(rows))

        for row_idx, (measurement, parameter) in enumerate(rows):
            ref_value = float(getattr(ref_geometry, parameter))
            comp_value = float(getattr(comp_geometry, parameter))

            values = [
                measurement,
                parameter,
                self._format_value(ref_value),
                self._format_value(comp_value),
                self._format_value(comp_value - ref_value, signed=True),
            ]

            for col_idx, value in enumerate(values):
                self.mapping_table.setItem(
                    row_idx,
                    col_idx,
                    QTableWidgetItem(value),
                )

        self.mapping_table.resizeColumnsToContents()

    def update_risk_table(
        self,
        ref_eval: GeometryEvaluation,
        comp_eval: GeometryEvaluation,
    ) -> None:
        names = list(comp_eval.risk_items.keys())
        self.risk_table.setRowCount(len(names))

        for row_idx, name in enumerate(names):
            ref_score = ref_eval.risk_items[name]
            comp_score = comp_eval.risk_items[name]

            values = [
                name,
                f"{ref_score:.3f}",
                f"{comp_score:.3f}",
                risk_level(comp_score),
            ]

            for col_idx, value in enumerate(values):
                self.risk_table.setItem(
                    row_idx,
                    col_idx,
                    QTableWidgetItem(value),
                )

        self.risk_table.resizeColumnsToContents()

    def update_lot_table(
        self,
        ref_group: str,
        comp_group: str,
        measurement_keys: Tuple[str, ...] | None = None,
    ) -> None:
        subset = self.data[self.data["group"].isin([ref_group, comp_group])]

        if measurement_keys is None:
            definition = OBJECT_REGISTRY.get(self.selected_object_id)
            measurement_keys = definition.measurement_keys if definition else tuple(MEASUREMENT_TO_PARAMETER.keys())

        visible_measurements = [key for key in measurement_keys if key in self.data.columns]
        columns = ["group", "lot_id", "recipe_id"] + visible_measurements

        self.lot_table.setRowCount(len(subset))
        self.lot_table.setColumnCount(len(columns))
        self.lot_table.setHorizontalHeaderLabels(columns)

        for row_idx, (_, row) in enumerate(subset.iterrows()):
            for col_idx, column in enumerate(columns):
                value = row[column]
                if isinstance(value, (float, np.floating)):
                    text = self._format_value(float(value))
                else:
                    text = str(value)
                self.lot_table.setItem(
                    row_idx,
                    col_idx,
                    QTableWidgetItem(text),
                )

        self.lot_table.resizeColumnsToContents()

    def build_interpretation(
        self,
        ref_geometry: ChannelHoleGeometry,
        comp_geometry: ChannelHoleGeometry,
        ref_eval: GeometryEvaluation,
        comp_eval: GeometryEvaluation,
    ) -> str:
        recommendation = {
            "Bottom CD": "Bottom CD 하한 확보와 하부 Etch profile 개선을 우선 검토하십시오.",
            "Aspect Ratio": "Etch Depth와 최소 CD 조합을 재검토해 AR margin을 확보하십시오.",
            "Bowing": "중앙부 Bowing과 Profile 조건을 우선 보정하십시오.",
            "Tilt": "Tilt 감소와 하부 중심축 이탈 margin 확보를 우선 검토하십시오.",
        }[comp_eval.main_driver]

        driver_values = {
            "Bottom CD": (
                ref_geometry.bottom_cd,
                comp_geometry.bottom_cd,
                "nm",
                2,
            ),
            "Aspect Ratio": (
                ref_eval.aspect_ratio,
                comp_eval.aspect_ratio,
                "",
                2,
            ),
            "Bowing": (
                ref_geometry.bowing,
                comp_geometry.bowing,
                "nm",
                2,
            ),
            "Tilt": (
                ref_geometry.tilt_deg,
                comp_geometry.tilt_deg,
                "deg",
                3,
            ),
        }
        ref_driver, comp_driver, unit, precision = driver_values[comp_eval.main_driver]
        driver_delta = comp_driver - ref_driver
        risk_delta = comp_eval.total_risk - ref_eval.total_risk
        return (
            f"COMP: {comp_geometry.group_name}\n"
            f"REF 대비 Risk: {risk_delta:+.3f} "
            f"({ref_eval.total_risk:.3f} → {comp_eval.total_risk:.3f})\n\n"
            f"Main Driver: {comp_eval.main_driver}\n"
            f"Driver 변화: {ref_driver:.{precision}f} → "
            f"{comp_driver:.{precision}f} {unit} ({driver_delta:+.{precision}f})\n"
            f"Critical Location: WL {comp_eval.critical_wl} / "
            f"Depth {comp_eval.critical_depth:.0f} nm\n"
            "\n"
            f"Recommended Check: {recommendation}"
        )

    def clear_result_tables(self) -> None:
        self.mapping_table.setRowCount(0)
        self.risk_table.setRowCount(0)
        self.lot_table.setRowCount(0)
        self.lot_table.setColumnCount(0)

    def clear_risk_table(self) -> None:
        self.risk_table.setRowCount(0)

    def clear_risk_and_lot_tables(self) -> None:
        self.clear_risk_table()
        self.lot_table.setRowCount(0)
        self.lot_table.setColumnCount(0)

    def on_group_selection_changed(self) -> None:
        if not hasattr(self, "structure_canvas"):
            return

        ref_group = self.ref_combo.currentText()
        comp_group = self.comp_combo.currentText()
        ref_geometry = group_to_channel_geometry(self.data, ref_group)
        comp_geometry = group_to_channel_geometry(self.data, comp_group)
        self.structure_canvas.set_groups(
            ref_group,
            comp_group,
            ref_geometry,
            comp_geometry,
        )
        self.structure_canvas.highlight_object(self.selected_object_id)

        if hasattr(self, "mapping_table"):
            definition = OBJECT_REGISTRY[self.selected_object_id]
            self.update_object_measurement_table(definition)
            self.update_lot_table(ref_group, comp_group, definition.measurement_keys)
            self.clear_risk_table()

            if definition.analysis_supported:
                self.set_summary_status("RE-EVALUATE")
                self.interpretation_text.setPlainText(
                    "REF/COMP 선택이 변경되었습니다.\n\n"
                    "3D Memory Hole 형상은 새 Measure 평균으로 갱신되었습니다. "
                    "Evaluate를 실행해 Risk 결론을 갱신하십시오."
                )
                self.comparison_canvas.draw_empty(
                    "REF/COMP selection changed\nEvaluate to refresh the comparison."
                )
                self.window_canvas.draw_empty(
                    "REF/COMP selection changed\nEvaluate to refresh the process window."
                )

    def regenerate_mock_data(self) -> None:
        seed = int(np.random.randint(0, 1_000_000))
        self.data = generate_mock_data(seed=seed)
        definition = OBJECT_REGISTRY[self.selected_object_id]
        self.update_object_measurement_table(definition)
        self.update_lot_table(
            self.ref_combo.currentText(),
            self.comp_combo.currentText(),
            definition.measurement_keys,
        )
        ref_group = self.ref_combo.currentText()
        comp_group = self.comp_combo.currentText()
        self.structure_canvas.set_groups(
            ref_group,
            comp_group,
            group_to_channel_geometry(self.data, ref_group),
            group_to_channel_geometry(self.data, comp_group),
        )
        self.structure_canvas.highlight_object(self.selected_object_id)
        self.clear_risk_table()
        self.comparison_canvas.draw_empty(
            "Mock data regenerated\nEvaluate to refresh the comparison."
        )
        self.window_canvas.draw_empty(
            "Mock data regenerated\nEvaluate to refresh the process window."
        )
        self.set_summary_status(
            "RE-EVALUATE" if definition.analysis_supported else "MAPPING ONLY"
        )
        self.interpretation_text.setPlainText(
            f"Mock data를 새로 생성했습니다. Seed: {seed}\n"
            "3D Memory Hole 형상도 새 Measure 평균으로 갱신했습니다. "
            "Memory Hole을 선택하고 Evaluate를 실행하면 Risk 평가가 갱신됩니다."
        )

    def show_full_structure(self) -> None:
        self.structure_canvas.show_full_structure()
        self.center_tabs.setCurrentIndex(0)

    def focus_selected_object(self) -> None:
        if not self.structure_canvas.focus_selected_object():
            QMessageBox.information(
                self,
                "Selected Object Focus",
                "선택한 Object의 확대 보기를 생성할 수 없습니다.",
            )
            return
        self.center_tabs.setCurrentIndex(0)

    def set_structure_view(self, view_name: str) -> None:
        self.structure_canvas.set_view(view_name)
        self.center_tabs.setCurrentIndex(0)

    def reset_structure_view(self) -> None:
        self.structure_canvas.show_full_structure()
        self.set_structure_view("3d")


def main() -> None:
    app = QApplication(sys.argv)

    window = IntegratedEvaluator()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
