from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from cad_diff.step_io import load_step
from cadpro.neural import (
    FEATURE_EDGE,
    HIDDEN_ONE,
    HIDDEN_TWO,
    INPUT_FEATURES,
    NeuralCheckpointError,
    NeuralConfig,
    NeuralDepthModel,
    TrainingDataError,
    extract_image_features,
    load_training_manifest,
    predict_step,
    train_depth_model,
)
from cadpro.step import write_step


def _image(path: Path, *, object_width: int, object_height: int, hole: bool = False) -> Path:
    canvas = np.full((112, 112, 3), 255, dtype=np.uint8)
    x0 = (112 - object_width) // 2
    y0 = (112 - object_height) // 2
    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + object_width, y0 + object_height),
        (15, 15, 15),
        -1,
    )
    if hole:
        cv2.circle(canvas, (56, 56), max(3, min(object_width, object_height) // 7), (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    path.write_bytes(encoded.tobytes())
    return path


def _constant_model(depth_ratio: float = 0.2) -> NeuralDepthModel:
    minimum = math.log(0.01)
    normalized = (math.log(depth_ratio) - minimum) / (math.log(4.0) - minimum)
    logit = math.log(normalized / (1.0 - normalized))
    return NeuralDepthModel(
        w1=np.zeros((INPUT_FEATURES, HIDDEN_ONE), dtype=np.float32),
        b1=np.zeros(HIDDEN_ONE, dtype=np.float32),
        w2=np.zeros((HIDDEN_ONE, HIDDEN_TWO), dtype=np.float32),
        b2=np.zeros(HIDDEN_TWO, dtype=np.float32),
        w3=np.zeros((HIDDEN_TWO, 1), dtype=np.float32),
        b3=np.asarray([logit], dtype=np.float32),
        feature_mean=np.zeros(INPUT_FEATURES, dtype=np.float32),
        feature_scale=np.ones(INPUT_FEATURES, dtype=np.float32),
        trained_examples=12,
        validation_relative_mae=0.08,
    )


def test_image_features_are_bounded_deterministic_and_include_shape_signals(tmp_path):
    source = _image(tmp_path / "bracket.png", object_width=72, object_height=44, hole=True)

    first = extract_image_features(source)
    second = extract_image_features(source)

    assert first.shape == (FEATURE_EDGE * FEATURE_EDGE + 5,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert 0 < first[: FEATURE_EDGE * FEATURE_EDGE].sum() < FEATURE_EDGE**2
    assert first[-3] > 0  # hole-count feature


def test_checkpoint_roundtrip_is_data_only_and_prediction_is_bounded(tmp_path):
    source = _image(tmp_path / "part.png", object_width=60, object_height=70)
    checkpoint = _constant_model(0.25).save(tmp_path / "depth.npz")

    loaded = NeuralDepthModel.load(checkpoint)
    prediction = loaded.predict(source, measured_width_mm=80)

    assert prediction.depth_ratio == pytest.approx(0.25, rel=1e-5)
    assert prediction.depth_mm == pytest.approx(20, rel=1e-5)
    assert 0.05 <= prediction.confidence_score <= 0.95
    assert prediction.to_dict()["manufacturing_verified"] is False
    with np.load(checkpoint, allow_pickle=False) as archive:
        assert all(archive[name].dtype != object for name in archive.files)


def test_checkpoint_rejects_unexpected_fields_and_object_payloads(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, metadata=np.asarray([1], dtype=np.uint8), surprise=np.asarray([1]))
    with pytest.raises(NeuralCheckpointError, match="unexpected"):
        NeuralDepthModel.load(bad)

    object_checkpoint = tmp_path / "object.npz"
    model = _constant_model()
    np.savez(
        object_checkpoint,
        metadata=np.asarray([{"unsafe": True}], dtype=object),
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
    )
    with pytest.raises(NeuralCheckpointError, match="decoded safely"):
        NeuralDepthModel.load(object_checkpoint)


def test_manifest_loads_dimension_labels_and_rejects_ambiguous_records(tmp_path):
    source = _image(tmp_path / "part.png", object_width=60, object_height=70)
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"image": source.name, "width_mm": 100, "depth_mm": 20})
            for _ in range(4)
        ),
        encoding="utf-8",
    )
    examples = load_training_manifest(manifest)
    assert len(examples) == 4
    assert examples[0].depth_ratio == pytest.approx(0.2)

    manifest.write_text(
        json.dumps(
            {"image": source.name, "step": "part.step", "width_mm": 100, "depth_mm": 20}
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrainingDataError, match="either"):
        load_training_manifest(manifest)


def test_manifest_derives_width_and_depth_from_one_aligned_step_solid(tmp_path):
    source = _image(tmp_path / "part.png", object_width=60, object_height=70)
    step = write_step(BRepPrimAPI_MakeBox(20, 12, 5).Shape(), tmp_path / "part.step")
    records = [{"image": source.name, "step": step.name}]
    records.extend(
        {"image": source.name, "width_mm": 20, "depth_mm": 5}
        for _ in range(3)
    )
    manifest = tmp_path / "paired.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    examples = load_training_manifest(manifest)

    assert examples[0].width_mm == pytest.approx(20)
    assert examples[0].depth_mm == pytest.approx(5)


def test_training_learns_a_shape_dependent_depth_signal_and_publishes_checkpoint(tmp_path):
    examples = []
    for index, object_width in enumerate((30, 36, 42, 48, 56, 64, 72, 82)):
        source = _image(
            tmp_path / f"sample-{index}.png",
            object_width=object_width,
            object_height=70,
            hole=index % 2 == 0,
        )
        ratio = 0.08 + index * 0.055
        from cadpro.neural import TrainingExample

        examples.append(TrainingExample(source, 100.0, ratio * 100.0))

    checkpoint = tmp_path / "trained.npz"
    summary = train_depth_model(
        examples,
        checkpoint,
        epochs=220,
        batch_size=4,
        learning_rate=0.004,
        validation_fraction=0,
        seed=7,
    )
    model = NeuralDepthModel.load(checkpoint)
    shallow = model.predict(examples[0].image_path, 100)
    deep = model.predict(examples[-1].image_path, 100)

    assert checkpoint.is_file()
    assert summary.final_training_loss < 0.01
    assert math.isfinite(summary.validation_relative_mae)
    assert deep.depth_mm > shallow.depth_mm
    assert deep.depth_ratio > shallow.depth_ratio


def test_prediction_builds_a_reloadable_step_with_predicted_depth(tmp_path):
    source = _image(tmp_path / "part.png", object_width=68, object_height=52, hole=True)
    checkpoint = _constant_model(0.2).save(tmp_path / "depth.npz")

    output, prediction = predict_step(
        source,
        checkpoint,
        measured_width_mm=50,
        output_path=tmp_path / "predicted.step",
    )

    assert output.is_file()
    assert prediction.depth_mm == pytest.approx(10, rel=1e-5)
    assert len(load_step(output)) == 1


def test_neural_config_is_explicit_and_does_not_expose_checkpoint_contents(tmp_path):
    checkpoint = _constant_model().save(tmp_path / "model.npz")
    disabled = NeuralConfig.from_env({"CADPRO_NEURAL_CHECKPOINT": str(checkpoint)})
    enabled = NeuralConfig.from_env(
        {
            "CADPRO_NEURAL_ENABLED": "1",
            "CADPRO_NEURAL_CHECKPOINT": str(checkpoint),
        }
    )
    assert disabled.available is False
    assert enabled.available is True
