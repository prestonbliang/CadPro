"""Trainable image-to-parametric-CAD depth prediction.

The neural network learns a bounded depth-to-width ratio from labeled object
images.  Inference combines that learned hidden-depth estimate with CadPro's
measured image silhouette; OpenCascade remains responsible for constructing
and validating the resulting STEP solid.

The implementation intentionally uses NumPy rather than executable pickle
checkpoints or a heavyweight runtime.  Checkpoints are strict, data-only NPZ
archives and can be loaded with ``allow_pickle=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence

import cv2
import numpy as np

from cadpro.media import extract_silhouette, validated_image_size


CHECKPOINT_VERSION = 1
FEATURE_EDGE = 24
EXTRA_FEATURES = 5
INPUT_FEATURES = FEATURE_EDGE * FEATURE_EDGE + EXTRA_FEATURES
HIDDEN_ONE = 64
HIDDEN_TWO = 32
MIN_DEPTH_RATIO = 0.01
MAX_DEPTH_RATIO = 4.0
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_UNCOMPRESSED_CHECKPOINT_BYTES = 8 * 1024 * 1024
MIN_TRAINING_EXAMPLES = 4
DEFAULT_CHECKPOINT = Path("cadpro-depth-model.npz")


class NeuralModelError(RuntimeError):
    """Base class for safe neural training and inference failures."""


class TrainingDataError(NeuralModelError):
    """Raised when a training manifest or sample is invalid."""


class NeuralCheckpointError(NeuralModelError):
    """Raised when a checkpoint is missing, malformed, or incompatible."""


@dataclass(frozen=True)
class NeuralConfig:
    """Opt-in website configuration for a trained local checkpoint."""

    enabled: bool = False
    checkpoint: Path | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "NeuralConfig":
        values = os.environ if environ is None else environ
        raw_path = (values.get("CADPRO_NEURAL_CHECKPOINT") or "").strip()
        return cls(
            enabled=_env_flag(values.get("CADPRO_NEURAL_ENABLED")),
            checkpoint=Path(raw_path).expanduser() if raw_path else None,
        )

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.checkpoint and self.checkpoint.is_file())


@dataclass(frozen=True)
class TrainingExample:
    image_path: Path
    width_mm: float
    depth_mm: float

    @property
    def depth_ratio(self) -> float:
        return self.depth_mm / self.width_mm


@dataclass(frozen=True)
class DepthPrediction:
    depth_mm: float
    depth_ratio: float
    confidence_score: float
    measured_width_mm: float
    checkpoint_version: int
    trained_examples: int
    validation_examples: int
    validation_relative_mae: float

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "completed",
            "model_type": "numpy_mlp_depth_regressor",
            "checkpoint_version": self.checkpoint_version,
            "trained_examples": self.trained_examples,
            "validation_examples": self.validation_examples,
            "measured_width_mm": self.measured_width_mm,
            "predicted_depth_mm": self.depth_mm,
            "predicted_depth_ratio": self.depth_ratio,
            "confidence_score": self.confidence_score,
            "confidence_kind": "heuristic_not_probability",
            "validation_relative_mae": self.validation_relative_mae,
            "changes_geometry": True,
            "manufacturing_verified": False,
            "warnings": [
                "Depth is a learned estimate from one visible view, not a measurement.",
                "The network cannot recover hidden pockets, side features, or backside topology.",
                "Verify the predicted depth and every critical feature before manufacturing.",
            ],
        }


@dataclass(frozen=True)
class TrainingSummary:
    checkpoint: Path
    examples: int
    training_examples: int
    validation_examples: int
    epochs: int
    final_training_loss: float
    validation_relative_mae: float


class NeuralDepthModel:
    """A compact two-hidden-layer MLP with a bounded scalar CAD parameter output."""

    def __init__(
        self,
        *,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        trained_examples: int,
        validation_relative_mae: float,
        validation_examples: int = 0,
    ) -> None:
        arrays = {
            "w1": np.asarray(w1, dtype=np.float32),
            "b1": np.asarray(b1, dtype=np.float32),
            "w2": np.asarray(w2, dtype=np.float32),
            "b2": np.asarray(b2, dtype=np.float32),
            "w3": np.asarray(w3, dtype=np.float32),
            "b3": np.asarray(b3, dtype=np.float32),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_scale": np.asarray(feature_scale, dtype=np.float32),
        }
        _validate_parameter_shapes(arrays)
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise NeuralCheckpointError(f"Checkpoint array {name} contains non-finite values")
        if (
            isinstance(trained_examples, bool)
            or not isinstance(trained_examples, int)
            or trained_examples < MIN_TRAINING_EXAMPLES
        ):
            raise NeuralCheckpointError(
                f"Checkpoint must record at least {MIN_TRAINING_EXAMPLES} training examples"
            )
        if (
            isinstance(validation_examples, bool)
            or not isinstance(validation_examples, int)
            or validation_examples < 0
        ):
            raise NeuralCheckpointError("Checkpoint validation example count is invalid")
        if (
            isinstance(validation_relative_mae, bool)
            or not isinstance(validation_relative_mae, (int, float))
            or not math.isfinite(validation_relative_mae)
            or validation_relative_mae < 0
        ):
            raise NeuralCheckpointError("Checkpoint validation error must be finite and non-negative")
        self.w1 = arrays["w1"]
        self.b1 = arrays["b1"]
        self.w2 = arrays["w2"]
        self.b2 = arrays["b2"]
        self.w3 = arrays["w3"]
        self.b3 = arrays["b3"]
        self.feature_mean = arrays["feature_mean"]
        self.feature_scale = arrays["feature_scale"]
        self.trained_examples = int(trained_examples)
        self.validation_examples = int(validation_examples)
        self.validation_relative_mae = float(validation_relative_mae)

    def predict(self, image_path: str | Path, measured_width_mm: float) -> DepthPrediction:
        width = _positive_number(measured_width_mm, "measured_width_mm")
        features = extract_image_features(image_path)
        normalized = np.clip(
            (features - self.feature_mean) / self.feature_scale,
            -8.0,
            8.0,
        )
        output = float(_forward(normalized[None, :], self)[0][0, 0])
        ratio = _decode_ratio(output)
        depth = width * ratio
        if not math.isfinite(depth) or depth <= 0:
            raise NeuralModelError("The neural model produced an invalid depth prediction")
        distance = float(np.sqrt(np.mean(np.square(normalized))))
        confidence = math.exp(-min(self.validation_relative_mae, 4.0))
        confidence *= math.exp(-max(0.0, distance - 1.5) * 0.2)
        if self.validation_examples == 0:
            confidence *= 0.5
        confidence = min(0.95, max(0.05, confidence))
        return DepthPrediction(
            depth_mm=round(depth, 6),
            depth_ratio=round(ratio, 8),
            confidence_score=round(confidence, 6),
            measured_width_mm=width,
            checkpoint_version=CHECKPOINT_VERSION,
            trained_examples=self.trained_examples,
            validation_examples=self.validation_examples,
            validation_relative_mae=round(self.validation_relative_mae, 8),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("Neural checkpoint output must use the .npz extension")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "model_type": "numpy_mlp_depth_regressor",
                "feature_edge": FEATURE_EDGE,
                "input_features": INPUT_FEATURES,
                "hidden_one": HIDDEN_ONE,
                "hidden_two": HIDDEN_TWO,
                "min_depth_ratio": MIN_DEPTH_RATIO,
                "max_depth_ratio": MAX_DEPTH_RATIO,
                "trained_examples": self.trained_examples,
                "validation_examples": self.validation_examples,
                "validation_relative_mae": self.validation_relative_mae,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=".npz",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(
                    stream,
                    metadata=np.frombuffer(metadata, dtype=np.uint8),
                    w1=self.w1,
                    b1=self.b1,
                    w2=self.w2,
                    b2=self.b2,
                    w3=self.w3,
                    b3=self.b3,
                    feature_mean=self.feature_mean,
                    feature_scale=self.feature_scale,
                )
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise NeuralCheckpointError("Generated neural checkpoint exceeds the size limit")
            os.replace(temporary, destination)
        except OSError as error:
            raise NeuralCheckpointError("Could not publish the neural checkpoint") from error
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "NeuralDepthModel":
        source = Path(path)
        if source.suffix.lower() != ".npz" or not source.is_file():
            raise NeuralCheckpointError("Neural checkpoint must be an existing .npz file")
        try:
            if source.stat().st_size <= 0 or source.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise NeuralCheckpointError("Neural checkpoint size is invalid")
            _validate_npz_container(source)
            with np.load(source, allow_pickle=False) as archive:
                expected = {
                    "metadata", "w1", "b1", "w2", "b2", "w3", "b3",
                    "feature_mean", "feature_scale",
                }
                if set(archive.files) != expected:
                    raise NeuralCheckpointError("Neural checkpoint fields are incomplete or unexpected")
                raw_metadata = np.asarray(archive["metadata"])
                if raw_metadata.dtype != np.uint8 or raw_metadata.ndim != 1:
                    raise NeuralCheckpointError("Neural checkpoint metadata is malformed")
                metadata = json.loads(raw_metadata.tobytes().decode("utf-8"))
                arrays = {name: np.array(archive[name], dtype=np.float32) for name in expected - {"metadata"}}
        except NeuralCheckpointError:
            raise
        except (
            OSError,
            TypeError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ) as error:
            raise NeuralCheckpointError("Neural checkpoint could not be decoded safely") from error
        if not isinstance(metadata, dict) or metadata.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise NeuralCheckpointError("Neural checkpoint version is unsupported")
        if metadata.get("model_type") != "numpy_mlp_depth_regressor":
            raise NeuralCheckpointError("Neural checkpoint model type is unsupported")
        architecture = {
            "feature_edge": FEATURE_EDGE,
            "input_features": INPUT_FEATURES,
            "hidden_one": HIDDEN_ONE,
            "hidden_two": HIDDEN_TWO,
            "min_depth_ratio": MIN_DEPTH_RATIO,
            "max_depth_ratio": MAX_DEPTH_RATIO,
        }
        if any(metadata.get(name) != value for name, value in architecture.items()):
            raise NeuralCheckpointError("Neural checkpoint architecture is incompatible")
        return cls(
            **arrays,
            trained_examples=metadata.get("trained_examples", 0),
            validation_relative_mae=metadata.get("validation_relative_mae", math.inf),
            validation_examples=metadata.get("validation_examples", 0),
        )


def extract_image_features(path: str | Path) -> np.ndarray:
    """Convert a clean object image into a bounded silhouette feature vector."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Training image does not exist: {source}")
    expected_size = validated_image_size(source)
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise TrainingDataError(f"Could not decode training image: {source.name}")
    if (int(image.shape[1]), int(image.shape[0])) != expected_size:
        raise TrainingDataError(f"Training image dimensions changed while reading: {source.name}")
    try:
        silhouette = extract_silhouette(image)
    except ValueError as error:
        raise TrainingDataError(
            f"Could not extract an object silhouette from {source.name}: {error}"
        ) from error

    width, height = silhouette.source_size
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(silhouette.outer).astype(np.int32)], 255)
    for hole in silhouette.holes:
        cv2.fillPoly(mask, [np.rint(hole).astype(np.int32)], 0)
    minimum = np.floor(silhouette.outer.min(axis=0)).astype(int)
    maximum = np.ceil(silhouette.outer.max(axis=0)).astype(int)
    x0, y0 = np.maximum(minimum, 0)
    x1 = min(width - 1, int(maximum[0]))
    y1 = min(height - 1, int(maximum[1]))
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0 or not np.any(crop):
        raise TrainingDataError(f"Training image has an empty object mask: {source.name}")

    inner_edge = FEATURE_EDGE - 4
    scale = min(inner_edge / crop.shape[1], inner_edge / crop.shape[0])
    resized_width = max(1, round(crop.shape[1] * scale))
    resized_height = max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((FEATURE_EDGE, FEATURE_EDGE), dtype=np.float32)
    offset_x = (FEATURE_EDGE - resized_width) // 2
    offset_y = (FEATURE_EDGE - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = (
        resized.astype(np.float32) / 255.0
    )
    foreground_fraction = float(np.count_nonzero(mask)) / float(width * height)
    aspect_ratio = crop.shape[1] / crop.shape[0]
    horizontal_symmetry = 1.0 - float(np.mean(np.abs(canvas - np.fliplr(canvas))))
    vertical_symmetry = 1.0 - float(np.mean(np.abs(canvas - np.flipud(canvas))))
    extras = np.asarray(
        [
            np.clip(math.log(max(aspect_ratio, 1e-6)), -4.0, 4.0) / 4.0,
            foreground_fraction,
            min(len(silhouette.holes), 8) / 8.0,
            horizontal_symmetry,
            vertical_symmetry,
        ],
        dtype=np.float32,
    )
    features = np.concatenate((canvas.reshape(-1), extras)).astype(np.float32)
    if features.shape != (INPUT_FEATURES,) or not np.isfinite(features).all():
        raise TrainingDataError("Image feature extraction produced invalid values")
    return features


def load_training_manifest(path: str | Path) -> tuple[TrainingExample, ...]:
    """Load JSONL samples with an image plus explicit dimensions or an aligned STEP."""
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest}")
    examples: list[TrainingExample] = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise TrainingDataError("Training manifest must be readable UTF-8 JSONL") from error
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise TrainingDataError(f"Manifest line {line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise TrainingDataError(f"Manifest line {line_number} must be a JSON object")
        unknown = set(record) - {"image", "width_mm", "depth_mm", "step"}
        if unknown:
            raise TrainingDataError(
                f"Manifest line {line_number} has unsupported fields: {', '.join(sorted(unknown))}"
            )
        image_value = record.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            raise TrainingDataError(f"Manifest line {line_number} needs an image path")
        image_path = _manifest_path(manifest, image_value)
        has_dimensions = "width_mm" in record or "depth_mm" in record
        has_step = "step" in record
        if has_dimensions == has_step:
            raise TrainingDataError(
                f"Manifest line {line_number} needs either width_mm + depth_mm or step"
            )
        if has_step:
            step_value = record.get("step")
            if not isinstance(step_value, str) or not step_value.strip():
                raise TrainingDataError(f"Manifest line {line_number} has an invalid STEP path")
            width_mm, depth_mm = _dimensions_from_step(_manifest_path(manifest, step_value))
        else:
            if "width_mm" not in record or "depth_mm" not in record:
                raise TrainingDataError(
                    f"Manifest line {line_number} needs both width_mm and depth_mm"
                )
            width_mm = _positive_number(record["width_mm"], f"line {line_number} width_mm")
            depth_mm = _positive_number(record["depth_mm"], f"line {line_number} depth_mm")
        if not image_path.is_file():
            raise TrainingDataError(f"Manifest line {line_number} image does not exist")
        ratio = depth_mm / width_mm
        if not MIN_DEPTH_RATIO <= ratio <= MAX_DEPTH_RATIO:
            raise TrainingDataError(
                f"Manifest line {line_number} depth/width ratio must be between "
                f"{MIN_DEPTH_RATIO:g} and {MAX_DEPTH_RATIO:g}"
            )
        examples.append(TrainingExample(image_path, width_mm, depth_mm))
    if len(examples) < MIN_TRAINING_EXAMPLES:
        raise TrainingDataError(
            f"Training requires at least {MIN_TRAINING_EXAMPLES} labeled examples"
        )
    return tuple(examples)


def train_depth_model(
    examples: Sequence[TrainingExample],
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    validation_fraction: float = 0.2,
    seed: int = 17,
    progress: Callable[[int, float], None] | None = None,
) -> TrainingSummary:
    """Train the MLP with Adam and atomically publish a safe NPZ checkpoint."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    samples = tuple(examples)
    if len(samples) < MIN_TRAINING_EXAMPLES:
        raise TrainingDataError(
            f"Training requires at least {MIN_TRAINING_EXAMPLES} labeled examples"
        )
    features = np.stack([extract_image_features(item.image_path) for item in samples])
    ratios = np.asarray(
        [
            _positive_number(item.depth_mm, "depth_mm")
            / _positive_number(item.width_mm, "width_mm")
            for item in samples
        ],
        dtype=np.float32,
    )
    if not np.isfinite(ratios).all() or np.any(ratios < MIN_DEPTH_RATIO) or np.any(ratios > MAX_DEPTH_RATIO):
        raise TrainingDataError("All depth/width targets must be finite and inside model bounds")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    validation_count = 0
    if validation_fraction > 0 and len(samples) >= 5:
        validation_count = min(
            max(1, round(len(samples) * validation_fraction)),
            len(samples) - MIN_TRAINING_EXAMPLES,
        )
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    training_x = features[training_indices]
    training_y = _encode_ratios(ratios[training_indices])[:, None]
    validation_x = features[validation_indices] if validation_count else training_x
    validation_ratios = ratios[validation_indices] if validation_count else ratios[training_indices]

    feature_mean = training_x.mean(axis=0).astype(np.float32)
    feature_scale = training_x.std(axis=0).astype(np.float32)
    feature_scale[feature_scale < 1e-4] = 1.0
    training_x = np.clip((training_x - feature_mean) / feature_scale, -8.0, 8.0)
    validation_x = np.clip((validation_x - feature_mean) / feature_scale, -8.0, 8.0)
    model = _initialized_model(rng, feature_mean, feature_scale, len(training_indices))

    parameters = [model.w1, model.b1, model.w2, model.b2, model.w3, model.b3]
    first_moment = [np.zeros_like(value) for value in parameters]
    second_moment = [np.zeros_like(value) for value in parameters]
    step = 0
    final_loss = math.inf
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(len(training_x))
        epoch_losses: list[float] = []
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            batch_x = training_x[indices]
            batch_y = training_y[indices]
            prediction, cache = _forward(batch_x, model)
            error = prediction - batch_y
            loss = float(np.mean(np.square(error)))
            epoch_losses.append(loss)
            gradients = _backward(batch_x, batch_y, prediction, cache, model)
            step += 1
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                gradient = np.clip(gradient, -5.0, 5.0)
                first_moment[index] *= 0.9
                first_moment[index] += 0.1 * gradient
                second_moment[index] *= 0.999
                second_moment[index] += 0.001 * np.square(gradient)
                corrected_first = first_moment[index] / (1.0 - 0.9**step)
                corrected_second = second_moment[index] / (1.0 - 0.999**step)
                parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        final_loss = float(np.mean(epoch_losses))
        if progress is not None:
            progress(epoch, final_loss)

    validation_output = _forward(validation_x, model)[0][:, 0]
    validation_prediction = np.asarray([_decode_ratio(float(value)) for value in validation_output])
    relative_mae = float(
        np.mean(np.abs(validation_prediction - validation_ratios) / validation_ratios)
    )
    final_model = NeuralDepthModel(
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        trained_examples=len(training_indices),
        validation_relative_mae=relative_mae,
        validation_examples=validation_count,
    )
    destination = final_model.save(checkpoint)
    return TrainingSummary(
        checkpoint=destination,
        examples=len(samples),
        training_examples=len(training_indices),
        validation_examples=validation_count,
        epochs=epochs,
        final_training_loss=final_loss,
        validation_relative_mae=relative_mae,
    )


def train_from_manifest(
    manifest: str | Path,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    **kwargs: object,
) -> TrainingSummary:
    return train_depth_model(load_training_manifest(manifest), checkpoint, **kwargs)


def predict_step(
    image_path: str | Path,
    checkpoint: str | Path,
    measured_width_mm: float,
    output_path: str | Path,
) -> tuple[Path, DepthPrediction]:
    """Predict hidden depth, build the profile solid, and write a validated STEP."""
    from cad_diff.step_io import load_step
    from cadpro.reconstruct import reconstruct_single_image
    from cadpro.step import write_step

    model = NeuralDepthModel.load(checkpoint)
    prediction = model.predict(image_path, measured_width_mm)
    reconstruction = reconstruct_single_image(
        image_path,
        width_mm=prediction.measured_width_mm,
        depth_mm=prediction.depth_mm,
    )
    output = write_step(reconstruction.shape, output_path)
    if len(load_step(output)) != 1:
        raise NeuralModelError("Predicted STEP failed round-trip solid validation")
    return output, prediction


def _initialized_model(
    rng: np.random.Generator,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    trained_examples: int,
) -> NeuralDepthModel:
    def weight(inputs: int, outputs: int) -> np.ndarray:
        limit = math.sqrt(6.0 / (inputs + outputs))
        return rng.uniform(-limit, limit, size=(inputs, outputs)).astype(np.float32)

    return NeuralDepthModel(
        w1=weight(INPUT_FEATURES, HIDDEN_ONE),
        b1=np.zeros(HIDDEN_ONE, dtype=np.float32),
        w2=weight(HIDDEN_ONE, HIDDEN_TWO),
        b2=np.zeros(HIDDEN_TWO, dtype=np.float32),
        w3=weight(HIDDEN_TWO, 1),
        b3=np.zeros(1, dtype=np.float32),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        trained_examples=trained_examples,
        validation_relative_mae=0.0,
    )


def _forward(
    values: np.ndarray,
    model: NeuralDepthModel,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    z1 = values @ model.w1 + model.b1
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ model.w2 + model.b2
    a2 = np.maximum(z2, 0.0)
    z3 = np.clip(a2 @ model.w3 + model.b3, -30.0, 30.0)
    output = 1.0 / (1.0 + np.exp(-z3))
    return output.astype(np.float32), (z1, a1, z2, a2)


def _backward(
    values: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    model: NeuralDepthModel,
) -> tuple[np.ndarray, ...]:
    z1, a1, z2, a2 = cache
    batch = max(1, values.shape[0])
    dz3 = (2.0 / batch) * (prediction - target) * prediction * (1.0 - prediction)
    dw3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)
    dz2 = (dz3 @ model.w3.T) * (z2 > 0)
    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ model.w2.T) * (z1 > 0)
    dw1 = values.T @ dz1
    db1 = dz1.sum(axis=0)
    return tuple(np.asarray(value, dtype=np.float32) for value in (dw1, db1, dw2, db2, dw3, db3))


def _validate_parameter_shapes(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "w1": (INPUT_FEATURES, HIDDEN_ONE),
        "b1": (HIDDEN_ONE,),
        "w2": (HIDDEN_ONE, HIDDEN_TWO),
        "b2": (HIDDEN_TWO,),
        "w3": (HIDDEN_TWO, 1),
        "b3": (1,),
        "feature_mean": (INPUT_FEATURES,),
        "feature_scale": (INPUT_FEATURES,),
    }
    for name, shape in expected.items():
        if name not in arrays or arrays[name].shape != shape:
            raise NeuralCheckpointError(f"Checkpoint array {name} has an incompatible shape")
    if np.any(arrays["feature_scale"] <= 0):
        raise NeuralCheckpointError("Checkpoint feature scale must be positive")


def _validate_npz_container(path: Path) -> None:
    expected = {
        "metadata.npy", "w1.npy", "b1.npy", "w2.npy", "b2.npy", "w3.npy", "b3.npy",
        "feature_mean.npy", "feature_scale.npy",
    }
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(expected) or set(names) != expected:
            raise NeuralCheckpointError("Neural checkpoint archive members are unexpected")
        if any(member.flag_bits & 0x1 for member in members):
            raise NeuralCheckpointError("Encrypted neural checkpoints are unsupported")
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_CHECKPOINT_BYTES:
            raise NeuralCheckpointError("Neural checkpoint expands beyond the safe size limit")


def _encode_ratios(ratios: np.ndarray) -> np.ndarray:
    minimum = math.log(MIN_DEPTH_RATIO)
    span = math.log(MAX_DEPTH_RATIO) - minimum
    return ((np.log(ratios) - minimum) / span).astype(np.float32)


def _decode_ratio(value: float) -> float:
    bounded = min(1.0, max(0.0, value))
    return math.exp(math.log(MIN_DEPTH_RATIO) + bounded * (math.log(MAX_DEPTH_RATIO) - math.log(MIN_DEPTH_RATIO)))


def _manifest_path(manifest: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else manifest.parent / candidate


def _dimensions_from_step(path: Path) -> tuple[float, float]:
    if path.suffix.lower() not in {".step", ".stp"} or not path.is_file():
        raise TrainingDataError("Aligned training STEP must be an existing .step or .stp file")
    from cad_diff.signatures import fingerprint_solid
    from cad_diff.step_io import load_step

    try:
        solids = load_step(path)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingDataError(f"Could not load aligned training STEP: {path.name}") from error
    if len(solids) != 1:
        raise TrainingDataError("Aligned training STEP must contain exactly one solid")
    fingerprint = fingerprint_solid(path.stem, solids[0][1])
    dimensions = tuple(
        float(maximum - minimum)
        for minimum, maximum in zip(fingerprint.bbox_min, fingerprint.bbox_max, strict=True)
    )
    width = _positive_number(dimensions[0], "STEP X width")
    depth = _positive_number(dimensions[2], "STEP Z depth")
    return width, depth


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingDataError(f"{label} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise TrainingDataError(f"{label} must be a finite positive number")
    return converted


def _env_flag(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})
