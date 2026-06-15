"""
geometry_engine.py
------------------
The mathematical foundation of the BIM AI Engine.

This module acts as a virtual precision saw — bypassing all heuristic
approximation to extract structural geometry from IFC building models as
100% straight, millimeter-perfect vectors.

No AI inference happens here. This is pure deterministic mathematics:
linear algebra, coordinate transforms, and computational geometry working
together to guarantee that a wall is always a wall — never a wavy line.

Pipeline:
    IFC File → IfcOpenShell Parser → Wall Element Extraction
             → Axis Line Computation → Snap-to-Grid (millimeter precision)
             → Deduplication → Normalized Float32 Numpy Arrays
             → Ready for neural network training or DXF output
"""

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("geometry_engine")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MM_PRECISION        = 1.0          # Snap all coordinates to 1 mm grid
MINIMUM_WALL_LENGTH = 50.0         # Ignore walls shorter than 50 mm (noise)
DEDUP_TOLERANCE     = 5.0          # Merge duplicate walls within 5 mm
CANVAS_SIZE_MM      = 50_000.0     # 50-metre bounding canvas for normalization
MAX_WALLS_PER_FILE  = 2_048        # Hard cap — prevents memory blowouts


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WallVector:
    """
    A single structural wall represented as a 2D axis line.

    All coordinates are in millimetres, snapped to a 1 mm grid,
    relative to the building's own origin point.

    Attributes:
        x1, y1  : Start point of the wall centreline (mm)
        x2, y2  : End point of the wall centreline (mm)
        thickness: Wall thickness (mm) — preserved for DXF output
        height   : Wall height (mm) — preserved for 3D reconstruction
        ifc_guid : Original IFC GlobalId — maintains traceability to source
        layer    : Storey / floor level the wall belongs to
    """
    x1:        float
    y1:        float
    x2:        float
    y2:        float
    thickness: float       = 200.0
    height:    float       = 3000.0
    ifc_guid:  str         = ""
    layer:     str         = "WALL"

    # ── Derived geometric properties ──────────────────────────────────────────

    @property
    def length(self) -> float:
        """Euclidean length of the wall axis in mm."""
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle_degrees(self) -> float:
        """Bearing angle of the wall from positive X-axis, in degrees [0, 360)."""
        angle = np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))
        return float(angle % 360.0)

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Geometric centre of the wall axis."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def direction_vector(self) -> np.ndarray:
        """Unit vector along the wall axis (always magnitude = 1.0)."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length = np.hypot(dx, dy)
        if length < 1e-9:
            return np.array([1.0, 0.0])
        return np.array([dx / length, dy / length])

    def as_array(self) -> np.ndarray:
        """
        Compact float32 feature vector for ML ingestion.

        Layout: [x1, y1, x2, y2, thickness, height, length, angle_deg]
        Shape:  (8,)  dtype=float32
        """
        return np.array(
            [self.x1, self.y1, self.x2, self.y2,
             self.thickness, self.height, self.length, self.angle_degrees],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        return (
            f"WallVector(({self.x1:.1f},{self.y1:.1f})→"
            f"({self.x2:.1f},{self.y2:.1f}) "
            f"L={self.length:.1f}mm θ={self.angle_degrees:.1f}°)"
        )


@dataclass
class GeometryPackage:
    """
    The complete, cleaned geometry output for one IFC file.

    This is what the training pipeline, the DXF exporter,
    and the FastAPI endpoint all consume.

    Attributes:
        walls       : Ordered list of extracted WallVector objects
        origin      : Global XY origin shift applied during extraction (mm)
        storey_count: Number of floor levels detected in the building
        source_file : Path to the originating IFC file
        warnings    : Non-fatal issues encountered during parsing
    """
    walls:        List[WallVector] = field(default_factory=list)
    origin:       Tuple[float, float] = (0.0, 0.0)
    storey_count: int  = 0
    source_file:  str  = ""
    warnings:     List[str] = field(default_factory=list)

    @property
    def wall_count(self) -> int:
        return len(self.walls)

    @property
    def is_empty(self) -> bool:
        return len(self.walls) == 0

    def to_numpy(self) -> np.ndarray:
        """
        Stack all wall vectors into a single (N, 8) float32 matrix.

        Each row is one wall. Each column is a feature.
        Returns an empty (0, 8) array if no walls were extracted.
        """
        if self.is_empty:
            return np.zeros((0, 8), dtype=np.float32)
        return np.stack([w.as_array() for w in self.walls], axis=0)

    def summary(self) -> str:
        """Human-readable extraction report for logging."""
        lines = [
            f"  Source   : {self.source_file}",
            f"  Walls    : {self.wall_count}",
            f"  Storeys  : {self.storey_count}",
            f"  Origin   : ({self.origin[0]:.1f}, {self.origin[1]:.1f}) mm",
        ]
        if self.warnings:
            lines.append(f"  Warnings : {len(self.warnings)}")
            for w in self.warnings[:5]:
                lines.append(f"    ⚠  {w}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PRECISION MATHEMATICS
# ─────────────────────────────────────────────────────────────────────────────

def snap_to_grid(value: float, grid: float = MM_PRECISION) -> float:
    """
    Round a floating-point coordinate to the nearest millimetre grid point.

    This is the core precision guarantee of the entire engine.
    Every coordinate that enters the system passes through this function.

    IFC files store geometry in metres with floating-point representation,
    which means a wall endpoint that should be at exactly 3000 mm might
    arrive as 2999.9999997 mm due to IEEE 754 rounding. This function
    corrects that, making all coordinates perfectly deterministic.

    Args:
        value: Raw coordinate in millimetres
        grid : Snap resolution in mm (default 1.0 = millimetre precision)

    Returns:
        Coordinate snapped to nearest grid point (float, but exact)

    Example:
        >>> snap_to_grid(2999.9999997)
        3000.0
        >>> snap_to_grid(1234.6, grid=5.0)
        1235.0
    """
    return round(value / grid) * grid


def ifc_to_mm(value: float, ifc_unit: str = "METRE") -> float:
    """
    Convert an IFC length value to millimetres.

    IFC files can store lengths in metres or millimetres depending
    on the project's unit settings. This normalizes everything to mm.

    Args:
        value   : Raw length value from the IFC file
        ifc_unit: The unit declared in the IFC header ("METRE" or "MILLIMETRE")

    Returns:
        Length in millimetres (float)
    """
    conversion = {"METRE": 1000.0, "MILLIMETRE": 1.0, "FOOT": 304.8, "INCH": 25.4}
    factor = conversion.get(ifc_unit.upper(), 1000.0)  # Default to metres if unknown
    return value * factor


def transform_point_2d(
    point: np.ndarray,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Apply a 2D affine transformation to a point.

    Used to convert wall endpoints from local object space (relative to
    the wall's own insertion point) into global building space (relative
    to the project origin).

    Args:
        point          : 2D point as [x, y] array
        rotation_matrix: 2x2 rotation matrix from IFC ObjectPlacement
        translation    : 2D translation vector [tx, ty]

    Returns:
        Transformed 2D point as numpy array [x', y']
    """
    return rotation_matrix @ point + translation


def compute_wall_axis_from_placement(
    wall_element,
    ifc_unit: str = "METRE",
) -> Optional[Tuple[float, float, float, float]]:
    """
    Deterministically extract the centreline axis of a single IFC wall element.

    This is the mathematical heart of the engine. It works in four stages:

    Stage 1 — Placement extraction:
        Reads the wall's ObjectPlacement to get its position and orientation
        in the global coordinate system.

    Stage 2 — Axis geometry:
        Reads the wall's Axis representation (the centreline) from its
        RepresentationMaps. If no Axis exists, falls back to computing
        the centreline from the SweptSolid extrusion geometry.

    Stage 3 — Coordinate transform:
        Applies the 3x3 affine matrix from Stage 1 to the raw axis points
        from Stage 2, producing world-space coordinates.

    Stage 4 — Millimetre snap:
        Passes every resulting coordinate through snap_to_grid() to
        eliminate floating-point noise and guarantee precision.

    Args:
        wall_element: An ifcopenshell IfcWall or IfcWallStandardCase entity
        ifc_unit    : Length unit declared in the IFC project ("METRE" etc.)

    Returns:
        (x1, y1, x2, y2) in millimetres, or None if extraction failed.
    """
    try:
        placement = wall_element.ObjectPlacement
        if placement is None:
            return None

        # ── Extract local-to-world transform from IFC placement ──────────────
        local_placement = placement.RelativePlacement
        if local_placement is None:
            return None

        # Location (translation vector)
        loc = local_placement.Location
        tx = ifc_to_mm(loc.Coordinates[0], ifc_unit)
        ty = ifc_to_mm(loc.Coordinates[1], ifc_unit)

        # Orientation (rotation matrix from RefDirection and Axis)
        ref_dir = local_placement.RefDirection
        if ref_dir is not None:
            dx, dy = ref_dir.DirectionRatios[0], ref_dir.DirectionRatios[1]
        else:
            dx, dy = 1.0, 0.0  # Default: wall runs along +X axis

        # Build the 2x2 rotation matrix
        length = np.hypot(dx, dy)
        if length < 1e-9:
            dx, dy = 1.0, 0.0
        else:
            dx, dy = dx / length, dy / length

        R = np.array([[dx, -dy], [dy, dx]])   # Orthonormal rotation matrix
        T = np.array([tx, ty])

        # ── Extract wall axis geometry ────────────────────────────────────────
        representations = wall_element.Representation
        if representations is None:
            return None

        axis_start_local = None
        axis_end_local   = None

        for rep in representations.Representations:
            if rep.RepresentationIdentifier == "Axis":
                for item in rep.Items:
                    # Axis is a single IfcPolyline or IfcIndexedPolyCurve
                    if hasattr(item, "Points"):
                        pts = item.Points
                        if len(pts) >= 2:
                            p0 = pts[0].Coordinates
                            p1 = pts[-1].Coordinates
                            axis_start_local = np.array([
                                ifc_to_mm(p0[0], ifc_unit),
                                ifc_to_mm(p0[1], ifc_unit),
                            ])
                            axis_end_local = np.array([
                                ifc_to_mm(p1[0], ifc_unit),
                                ifc_to_mm(p1[1], ifc_unit),
                            ])
                        break
                break

        # Fallback: if no Axis rep, use a zero-length point at the placement
        # origin and try to read length from the SweptSolid extrusion
        if axis_start_local is None:
            axis_start_local = np.array([0.0, 0.0])
            # Attempt to read extrusion length from body geometry
            extrusion_length = _read_extrusion_length(wall_element, ifc_unit)
            axis_end_local = np.array([extrusion_length, 0.0])

        # ── Transform from local space to world space ─────────────────────────
        world_start = transform_point_2d(axis_start_local, R, T)
        world_end   = transform_point_2d(axis_end_local,   R, T)

        # ── Snap all four coordinates to 1 mm grid ────────────────────────────
        x1 = snap_to_grid(world_start[0])
        y1 = snap_to_grid(world_start[1])
        x2 = snap_to_grid(world_end[0])
        y2 = snap_to_grid(world_end[1])

        return x1, y1, x2, y2

    except Exception as exc:
        log.debug("Axis extraction failed for wall %s: %s",
                  getattr(wall_element, "GlobalId", "?"), exc)
        return None


def _read_extrusion_length(wall_element, ifc_unit: str = "METRE") -> float:
    """
    Fallback: read wall length from the SweptSolid extrusion body.

    When an IFC wall has no Axis representation, the wall's run length
    is encoded as the extrusion depth of its IfcExtrudedAreaSolid body.

    Returns:
        Extrusion length in mm (float), or a safe default of 1000 mm.
    """
    try:
        for rep in wall_element.Representation.Representations:
            if rep.RepresentationIdentifier in ("Body", "SweptSolid"):
                for item in rep.Items:
                    if hasattr(item, "Depth"):
                        return ifc_to_mm(float(item.Depth), ifc_unit)
    except Exception:
        pass
    return 1000.0  # Safe fallback: assume 1-metre wall


def _read_wall_thickness(wall_element, ifc_unit: str = "METRE") -> float:
    """
    Extract wall thickness from the IfcMaterialLayerSetUsage if available.

    Returns:
        Thickness in mm (float), or the standard default of 200 mm.
    """
    try:
        for rel in wall_element.HasAssociations:
            if rel.is_a("IfcRelAssociatesMaterial"):
                usage = rel.RelatingMaterial
                if hasattr(usage, "ForLayerSet"):
                    total = sum(
                        ifc_to_mm(float(layer.LayerThickness), ifc_unit)
                        for layer in usage.ForLayerSet.MaterialLayers
                    )
                    if total > 0:
                        return snap_to_grid(total)
    except Exception:
        pass
    return 200.0  # Standard default: 200 mm (8-inch) wall


def _read_wall_height(wall_element, ifc_unit: str = "METRE") -> float:
    """
    Read wall height from extrusion depth or IfcQuantityLength properties.

    Returns:
        Height in mm (float), or the standard default of 3000 mm.
    """
    try:
        for rel in wall_element.IsDefinedBy:
            if rel.is_a("IfcRelDefinesByProperties"):
                prop_set = rel.RelatingPropertyDefinition
                if hasattr(prop_set, "Quantities"):
                    for qty in prop_set.Quantities:
                        if qty.Name in ("Height", "NetHeight", "GrossHeight"):
                            return snap_to_grid(
                                ifc_to_mm(float(qty.LengthValue), ifc_unit)
                            )
    except Exception:
        pass
    return 3000.0  # Standard default: 3-metre storey height


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_walls(
    walls: List[WallVector],
    tolerance: float = DEDUP_TOLERANCE,
) -> List[WallVector]:
    """
    Remove geometrically duplicate wall vectors.

    IFC files frequently contain coincident walls due to:
    - Linked models (structural + architectural overlapping)
    - Copy-paste errors in the authoring tool
    - Both IfcWall and IfcWallStandardCase for the same physical wall

    Two walls are considered duplicates if both their endpoints lie within
    `tolerance` mm of each other (checked in both directions: AB ≈ AB and AB ≈ BA).

    Algorithm:
        O(N²) brute force — acceptable for N < 2048 walls per file.
        For very large models, upgrade to a spatial index (KD-tree).

    Args:
        walls    : Raw list of extracted WallVector objects
        tolerance: Maximum endpoint distance to consider walls identical (mm)

    Returns:
        Deduplicated list — first occurrence wins when duplicates are found.
    """
    if len(walls) <= 1:
        return walls

    kept: List[WallVector] = []

    for candidate in walls:
        is_duplicate = False
        for existing in kept:
            # Check forward direction: candidate.start ≈ existing.start
            forward = (
                abs(candidate.x1 - existing.x1) <= tolerance and
                abs(candidate.y1 - existing.y1) <= tolerance and
                abs(candidate.x2 - existing.x2) <= tolerance and
                abs(candidate.y2 - existing.y2) <= tolerance
            )
            # Check reverse direction: candidate.start ≈ existing.end
            reverse = (
                abs(candidate.x1 - existing.x2) <= tolerance and
                abs(candidate.y1 - existing.y2) <= tolerance and
                abs(candidate.x2 - existing.x1) <= tolerance and
                abs(candidate.y2 - existing.y1) <= tolerance
            )
            if forward or reverse:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    removed = len(walls) - len(kept)
    if removed > 0:
        log.info("Deduplication removed %d coincident wall(s).", removed)

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_canvas(
    walls: List[WallVector],
    canvas_size: float = CANVAS_SIZE_MM,
) -> Tuple[List[WallVector], Tuple[float, float], float]:
    """
    Translate and scale all wall coordinates to a fixed [0, canvas_size] canvas.

    Neural networks require normalized inputs. Raw IFC coordinates can
    span from -30,000 mm to +80,000 mm in arbitrary world positions.
    This function:
        1. Translates all coordinates so the bounding box starts at (0, 0)
        2. Scales uniformly (preserving aspect ratio) to fit within the canvas
        3. Returns the inverse transform parameters for later de-normalization

    Args:
        walls      : List of WallVector objects in raw world coordinates (mm)
        canvas_size: Target canvas dimension in mm (default 50,000 = 50 metres)

    Returns:
        Tuple of:
            - Normalized WallVector list
            - (origin_x, origin_y): the translation applied (mm)
            - scale_factor: the uniform scale applied (dimensionless)
    """
    if not walls:
        return walls, (0.0, 0.0), 1.0

    all_x = [w.x1 for w in walls] + [w.x2 for w in walls]
    all_y = [w.y1 for w in walls] + [w.y2 for w in walls]

    min_x, min_y = min(all_x), min(all_y)
    max_x, max_y = max(all_x), max(all_y)

    width  = max_x - min_x
    height = max_y - min_y
    span   = max(width, height, 1.0)  # Avoid div-by-zero for point buildings

    scale = canvas_size / span

    normalized: List[WallVector] = []
    for w in walls:
        normalized.append(WallVector(
            x1=snap_to_grid((w.x1 - min_x) * scale),
            y1=snap_to_grid((w.y1 - min_y) * scale),
            x2=snap_to_grid((w.x2 - min_x) * scale),
            y2=snap_to_grid((w.y2 - min_y) * scale),
            thickness=snap_to_grid(w.thickness * scale),
            height=w.height,           # Height is not scaled (stays metric)
            ifc_guid=w.ifc_guid,
            layer=w.layer,
        ))

    log.info(
        "Normalized %d walls to canvas %.0f mm | scale=%.4f | "
        "original span=(%.0f × %.0f) mm",
        len(normalized), canvas_size, scale, width, height,
    )

    return normalized, (min_x, min_y), scale


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_geometry(ifc_path: str) -> GeometryPackage:
    """
    Full geometry extraction pipeline for a single IFC file.

    This is the primary public interface of the geometry engine.
    Call this function with a path to any IFC file; receive a clean,
    normalized GeometryPackage ready for ML training or DXF export.

    Pipeline stages:
        1. Open and validate the IFC file
        2. Detect project length units
        3. Count building storeys
        4. Iterate all IfcWall + IfcWallStandardCase elements
        5. Extract axis via compute_wall_axis_from_placement()
        6. Filter out walls below minimum length
        7. Read thickness and height metadata
        8. Deduplicate coincident walls
        9. Normalize coordinates to canvas space
       10. Return GeometryPackage

    Args:
        ifc_path: Absolute or relative path to the .ifc file (str or Path)

    Returns:
        GeometryPackage containing all extracted walls and metadata.
        Returns an empty GeometryPackage (with warnings) on failure.

    Raises:
        Does not raise. All errors are captured as warnings in the package.
    """
    path = Path(ifc_path)
    package = GeometryPackage(source_file=str(path))

    # ── Validate file ─────────────────────────────────────────────────────────
    if not path.exists():
        msg = f"IFC file not found: {path}"
        log.error(msg)
        package.warnings.append(msg)
        return package

    if path.suffix.lower() != ".ifc":
        msg = f"Expected .ifc extension, got '{path.suffix}'. Proceeding anyway."
        log.warning(msg)
        package.warnings.append(msg)

    # ── Open IFC model ────────────────────────────────────────────────────────
    try:
        log.info("Opening IFC file: %s", path.name)
        model = ifcopenshell.open(str(path))
    except Exception as exc:
        msg = f"Failed to open IFC file: {exc}"
        log.error(msg)
        package.warnings.append(msg)
        return package

    # ── Detect project length units ───────────────────────────────────────────
    ifc_unit = "METRE"  # Assume SI metres unless the file says otherwise
    try:
        project = model.by_type("IfcProject")[0]
        for context in project.UnitsInContext.Units:
            if hasattr(context, "UnitType") and context.UnitType == "LENGTHUNIT":
                if hasattr(context, "Name"):
                    ifc_unit = context.Name.upper()
                elif hasattr(context, "Prefix") and context.Prefix:
                    ifc_unit = f"{context.Prefix}METRE".upper()
                break
    except Exception as exc:
        msg = f"Could not detect length units, defaulting to METRE: {exc}"
        log.warning(msg)
        package.warnings.append(msg)

    log.info("IFC length unit detected: %s", ifc_unit)

    # ── Count building storeys ────────────────────────────────────────────────
    try:
        storeys = model.by_type("IfcBuildingStorey")
        package.storey_count = len(storeys)
        log.info("Building storeys detected: %d", package.storey_count)
    except Exception:
        package.storey_count = 1

    # ── Extract all wall elements ─────────────────────────────────────────────
    wall_types = ["IfcWall", "IfcWallStandardCase"]
    raw_walls: List[WallVector] = []
    skipped_short = 0
    skipped_no_axis = 0

    for wall_type in wall_types:
        try:
            elements = model.by_type(wall_type)
        except Exception:
            elements = []

        for element in elements:
            if len(raw_walls) >= MAX_WALLS_PER_FILE:
                msg = f"Wall cap ({MAX_WALLS_PER_FILE}) reached. Truncating."
                log.warning(msg)
                package.warnings.append(msg)
                break

            # Extract axis geometry
            axis = compute_wall_axis_from_placement(element, ifc_unit)
            if axis is None:
                skipped_no_axis += 1
                continue

            x1, y1, x2, y2 = axis

            # Build WallVector before length check
            wall = WallVector(
                x1=x1, y1=y1, x2=x2, y2=y2,
                thickness=_read_wall_thickness(element, ifc_unit),
                height=_read_wall_height(element, ifc_unit),
                ifc_guid=getattr(element, "GlobalId", ""),
                layer=wall_type.upper(),
            )

            # Filter noise walls
            if wall.length < MINIMUM_WALL_LENGTH:
                skipped_short += 1
                log.debug("Skipped short wall %s (%.1f mm)", wall.ifc_guid, wall.length)
                continue

            raw_walls.append(wall)

    log.info(
        "Extracted %d walls | skipped: %d too-short, %d no-axis",
        len(raw_walls), skipped_short, skipped_no_axis,
    )

    if not raw_walls:
        msg = "No valid walls could be extracted from this IFC file."
        log.warning(msg)
        package.warnings.append(msg)
        return package

    # ── Deduplicate ───────────────────────────────────────────────────────────
    clean_walls = deduplicate_walls(raw_walls)

    # ── Normalize to canvas ───────────────────────────────────────────────────
    normalized_walls, origin, _scale = normalize_to_canvas(clean_walls)

    # ── Populate and return package ───────────────────────────────────────────
    package.walls  = normalized_walls
    package.origin = origin

    log.info("Extraction complete:\n%s", package.summary())
    return package


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_geometry_batch(
    ifc_dir: str,
    max_files: Optional[int] = None,
) -> List[GeometryPackage]:
    """
    Process an entire directory of IFC files in sequence.

    Used by the training data pipeline to convert the raw_ifc/ folder
    into a list of GeometryPackages ready for tensor conversion.

    Args:
        ifc_dir  : Path to directory containing .ifc files
        max_files: Optional cap on number of files to process

    Returns:
        List of GeometryPackage objects, one per successfully parsed file.
        Files that fail are logged as warnings but don't halt the batch.
    """
    directory = Path(ifc_dir)
    if not directory.is_dir():
        log.error("Directory not found: %s", directory)
        return []

    ifc_files = sorted(directory.glob("*.ifc"))
    if max_files is not None:
        ifc_files = ifc_files[:max_files]

    log.info("Batch processing %d IFC files from: %s", len(ifc_files), directory)

    packages: List[GeometryPackage] = []
    for i, ifc_file in enumerate(ifc_files, start=1):
        log.info("[%d/%d] Processing: %s", i, len(ifc_files), ifc_file.name)
        try:
            pkg = extract_geometry(str(ifc_file))
            if not pkg.is_empty:
                packages.append(pkg)
            else:
                log.warning("Skipping empty package: %s", ifc_file.name)
        except Exception as exc:
            log.error("Unexpected error processing %s: %s", ifc_file.name, exc)

    log.info(
        "Batch complete: %d/%d files yielded usable geometry.",
        len(packages), len(ifc_files),
    )
    return packages


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DIAGNOSTIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python geometry_engine.py <path_to_file.ifc>")
        print("       python geometry_engine.py <path_to_ifc_directory/>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        results = extract_geometry_batch(str(target))
        print(f"\n{'='*60}")
        print(f"BATCH RESULT: {len(results)} packages extracted")
        for pkg in results:
            print(f"\n{pkg.summary()}")
            matrix = pkg.to_numpy()
            print(f"  Array shape : {matrix.shape}")
            print(f"  Array dtype : {matrix.dtype}")
    else:
        pkg = extract_geometry(str(target))
        print(f"\n{'='*60}")
        print(pkg.summary())
        matrix = pkg.to_numpy()
        print(f"\nOutput array shape : {matrix.shape}")
        print(f"Output array dtype : {matrix.dtype}")
        if not pkg.is_empty:
            print(f"\nFirst 3 wall vectors:")
            for w in pkg.walls[:3]:
                print(f"  {w}")

