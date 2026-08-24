"""Typed schemas shared by the scan pipeline, worker, API, and report exporter."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Reject accidental fields at trust boundaries instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class InputMode(str, Enum):
    PHOTOS = "photos"
    VIDEO = "video"
    SINGLE_IMAGE = "single_image"


class QualityPreset(str, Enum):
    DRAFT = "draft"
    BALANCED = "balanced"
    HIGH = "high"


class VideoSelectionSettings(StrictModel):
    maximum_duration_seconds: float = Field(default=300.0, gt=0, le=3_600)
    candidate_frames_per_second: float = Field(default=3.0, gt=0, le=10)
    maximum_candidate_frames: int = Field(default=600, ge=10, le=2_000)
    target_frames: int = Field(default=40, ge=8, le=200)
    minimum_spacing_seconds: float = Field(default=0.25, ge=0, le=30)
    maximum_similarity: float = Field(default=0.985, ge=0.5, le=1)
    minimum_viewpoint_change: float = Field(default=0.015, ge=0, le=1)


class ReconstructionReuse(StrictModel):
    """Trusted internal reference to immutable geometry from a completed scan."""

    source_job_id: UUID
    registered_cameras: int = Field(ge=3)
    sparse_points: int = Field(ge=0)
    reprojection_error_px: float | None = Field(default=None, ge=0)
    source_textured: bool = False
    tool_versions: dict[str, str] = Field(default_factory=dict)
    reconstruction_warnings: tuple[str, ...] = ()


class ScanConfiguration(StrictModel):
    mode: InputMode
    quality_preset: QualityPreset = QualityPreset.BALANCED
    feature_matcher: Literal["exhaustive", "sequential"] = "exhaustive"
    mesher: Literal["poisson", "delaunay"] = "poisson"
    use_gpu: bool = True
    generate_cad: bool = True
    video: VideoSelectionSettings = Field(default_factory=VideoSelectionSettings)
    scale: ScaleMeasurement | None = None
    single_image_provider: str | None = None
    reconstruction_reuse: ReconstructionReuse | None = None


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    EXTRACTING_FRAMES = "extracting_frames"
    ANALYZING_IMAGES = "analyzing_images"
    ESTIMATING_CAMERAS = "estimating_cameras"
    BUILDING_DENSE_CLOUD = "building_dense_cloud"
    BUILDING_MESH = "building_mesh"
    TEXTURING = "texturing"
    APPLYING_SCALE = "applying_scale"
    REPAIRING_MESH = "repairing_mesh"
    EXPORTING = "exporting"
    FITTING_CAD = "fitting_cad"
    VALIDATING_OUTPUTS = "validating_outputs"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_STAGES = frozenset(
    stage for stage in JobStage if stage not in {JobStage.COMPLETED, JobStage.FAILED, JobStage.CANCELLED}
)


class ArtifactKind(str, Enum):
    SPARSE_POINT_CLOUD = "sparse_point_cloud"
    DENSE_POINT_CLOUD = "dense_point_cloud"
    TRIANGLE_MESH = "triangle_mesh"
    TEXTURED_MODEL = "visualization_model"
    PRINTABLE_MESH = "watertight_printable_mesh"
    ANALYTIC_CAD = "analytic_cad_brep"
    CAD_SCRIPT = "editable_cad_script"
    CONTACT_SHEET = "selected_frame_contact_sheet"
    PREVIEW = "interactive_preview"
    TEXTURE_RESOURCE = "texture_resource"
    REPORT = "reconstruction_report"
    MANIFEST = "reproducibility_manifest"
    BUNDLE = "complete_bundle"
    LOG = "processing_log"


class StructuredNotice(StrictModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    message: str = Field(min_length=1, max_length=1000)
    stage: JobStage | None = None
    details: dict[str, object] = Field(default_factory=dict)


class ToolCapability(StrictModel):
    name: str
    available: bool
    executable: str | None = None
    version: str | None = None
    reason: str | None = None
    install_hint: str | None = None


class ToolchainCapabilities(StrictModel):
    tools: dict[str, ToolCapability]
    photo_reconstruction: bool
    video_ingest: bool
    dense_reconstruction: bool
    texture_generation: bool
    mesh_processing: bool
    analytic_cad: bool
    standard_pipeline_uses_paid_cloud: Literal[False] = False


class ImageQuality(StrictModel):
    source_name: str
    normalized_name: str | None = None
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    blur_score: float = Field(ge=0)
    shadow_fraction: float = Field(ge=0, le=1)
    highlight_fraction: float = Field(ge=0, le=1)
    feature_count: int = Field(ge=0)
    perceptual_hash: str
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VideoMetadata(StrictModel):
    codec: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    duration_seconds: float = Field(gt=0)
    frame_rate: float = Field(gt=0)
    frame_count: int | None = Field(default=None, ge=1)
    size_bytes: int = Field(ge=1)


class VideoFrame(StrictModel):
    source_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    filename: str
    blur_score: float = Field(ge=0)
    similarity_to_previous: float | None = Field(default=None, ge=0, le=1)
    viewpoint_change: float | None = Field(default=None, ge=0, le=1)


class ScaleMeasurement(StrictModel):
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]
    real_distance: Annotated[float, Field(gt=0)]
    unit: Literal["mm", "cm", "m", "in"]
    selection_uncertainty: Annotated[float, Field(ge=0)] = 0.0

    @field_validator("point_a", "point_b")
    @classmethod
    def finite_point(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        import math

        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError("calibration points must contain finite coordinates")
        return value


class ScaleInformation(StrictModel):
    calibrated: bool = False
    original_units: str = "arbitrary reconstruction units"
    output_unit: Literal["mm", "cm", "m", "in"] | None = None
    scale_factor: float | None = None
    calibration_method: str | None = None
    user_distance: float | None = None
    estimated_uncertainty: float | None = None
    warning: str = "Scale is unknown; do not use dimensions for manufacturing."


class FeatureFit(StrictModel):
    feature_type: Literal[
        "plane", "cylinder", "cone", "sphere", "box", "extrusion", "circular_hole"
    ]
    parameters: dict[str, object]
    inlier_ratio: float = Field(ge=0, le=1)
    rms_residual: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    accepted: bool


class ReconstructionMetrics(StrictModel):
    uploaded_images: int = Field(ge=0)
    accepted_images: int = Field(ge=0)
    registered_cameras: int = Field(ge=0)
    registration_percentage: float = Field(ge=0, le=100)
    sparse_points: int = Field(ge=0)
    dense_points: int = Field(ge=0)
    reprojection_error_px: float | None = Field(default=None, ge=0)
    bounding_box: tuple[float, float, float] | None = None
    vertices: int = Field(ge=0)
    triangles: int = Field(ge=0)
    connected_components: int = Field(ge=0)
    boundary_edges: int = Field(ge=0)
    non_manifold_edges: int = Field(ge=0)
    watertight: bool
    texture_resolution: tuple[int, int] | None = None


class StageTiming(StrictModel):
    stage: JobStage
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)


class ArtifactMetadata(StrictModel):
    artifact_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    kind: ArtifactKind
    filename: str
    media_type: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    textured: bool | None = None
    metric_scale: bool | None = None
    validated: bool = True


class ReconstructionReport(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    job_id: str
    mode: InputMode
    status: JobStatus
    quality_class: Literal["excellent", "usable", "weak", "failed"]
    configuration: dict[str, object]
    capabilities: ToolchainCapabilities
    image_quality: list[ImageQuality]
    video: VideoMetadata | None = None
    selected_frames: list[VideoFrame] = Field(default_factory=list)
    metrics: ReconstructionMetrics
    scale: ScaleInformation
    cad_features: list[FeatureFit] = Field(default_factory=list)
    cad_status: Literal["generated", "skipped", "failed"] = "skipped"
    cad_explanation: str
    warnings: list[StructuredNotice] = Field(default_factory=list)
    errors: list[StructuredNotice] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    tool_versions: dict[str, str]
    artifacts: list[ArtifactMetadata] = Field(default_factory=list)
    logs_included: bool = True
    inferred_surfaces: bool = False


class ReproducibilityManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str
    input_sha256: dict[str, str]
    configuration: dict[str, object]
    commands: list[list[str]]
    tool_versions: dict[str, str]
    artifacts: list[ArtifactMetadata]
    created_at: datetime
