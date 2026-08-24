from __future__ import annotations

import os
from pathlib import Path
import threading
import webbrowser

import typer

from cadpro.pipeline import convert_media, convert_turntable_video


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run truthful photo/video reconstruction, inspect dependencies, or use legacy CAD tools."""


@app.command("scan-doctor")
def scan_doctor(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Detect the exact local tools available to the real photogrammetry pipeline."""

    from cadpro.scan.capabilities import detect_toolchain

    capabilities = detect_toolchain()
    if json_output:
        typer.echo(capabilities.model_dump_json(indent=2))
        return
    typer.echo("CadPro real reconstruction dependency check")
    for capability in capabilities.tools.values():
        status = "READY" if capability.available else "MISSING"
        detail = capability.version or capability.reason or ""
        typer.echo(f"[{status:7}] {capability.name}: {detail}")
        if not capability.available and capability.install_hint:
            typer.echo(f"          {capability.install_hint}")
    typer.echo(
        "Photo SfM: "
        + ("ready" if capabilities.photo_reconstruction else "unavailable")
        + " | Video ingest: "
        + ("ready" if capabilities.video_ingest else "unavailable")
        + " | Camera texturing: "
        + ("ready" if capabilities.texture_generation else "unavailable")
    )


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Source image or video."),
    output: Path = typer.Option(Path("model.step"), "--output", "-o", help="Destination STEP file."),
    width_mm: float = typer.Option(100.0, min=0.001, help="Real-world width of the detected object."),
    depth_mm: float = typer.Option(10.0, min=0.001, help="Extrusion depth of the generated part."),
) -> None:
    """Convert an image or video silhouette into a solid STEP model."""
    try:
        result = convert_media(input_path, output, width_mm=width_mm, depth_mm=depth_mm)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    source = f"video frame {result.selected_frame}" if result.selected_frame is not None else "image"
    typer.echo(
        f"Created {result.output} from {source}: {result.outline_points} outline points, "
        f"{result.holes} hole(s), {width_mm:g} x {depth_mm:g} mm"
    )


@app.command()
def turntable(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="One full turntable video."),
    width_mm: float = typer.Option(..., min=0.001, help="Real maximum horizontal span over the full rotation."),
    output: Path = typer.Option(Path("model.step"), "--output", "-o", help="Destination STEP file."),
    views: int = typer.Option(8, min=4, max=24, help="Evenly spaced silhouettes used for reconstruction."),
    start_frame: int = typer.Option(0, min=0, help="First frame of the complete revolution."),
    end_frame: int | None = typer.Option(None, min=1, help="Frame after the complete revolution."),
    clockwise: bool = typer.Option(
        False,
        "--clockwise/--counterclockwise",
        help="Rotation direction as viewed from above.",
    ),
) -> None:
    """Reconstruct a solid visual hull from one 360-degree turntable video."""
    try:
        result = convert_turntable_video(
            input_path,
            output,
            width_mm=width_mm,
            views=views,
            start_frame=start_frame,
            end_frame=end_frame,
            clockwise=clockwise,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    frames = ", ".join(str(index) for index in result.sampled_frames)
    typer.echo(
        f"Created {result.output} from {len(result.sampled_frames)} turntable views "
        f"(frames {frames}), scaled to {width_mm:g} mm maximum width"
    )


@app.command("neural-train")
def neural_train(
    manifest: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="UTF-8 JSONL file of image/dimension or aligned image/STEP pairs.",
    ),
    checkpoint: Path = typer.Option(
        Path("cadpro-depth-model.npz"),
        "--checkpoint",
        "-c",
        help="Destination data-only neural checkpoint.",
    ),
    epochs: int = typer.Option(200, min=1, max=100_000, help="Training passes over the dataset."),
    batch_size: int = typer.Option(16, min=1, max=4096, help="Examples per Adam update."),
    learning_rate: float = typer.Option(0.001, min=0.000001, max=1.0),
    validation_fraction: float = typer.Option(0.2, min=0.0, max=0.49),
    seed: int = typer.Option(17, min=0, max=2_147_483_647),
) -> None:
    """Train the image-to-CAD depth network and write a safe NPZ checkpoint."""
    from cadpro.neural import NeuralModelError, train_from_manifest

    try:
        summary = train_from_manifest(
            manifest,
            checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    except (FileNotFoundError, ValueError, NeuralModelError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"Trained {summary.checkpoint} from {summary.examples} examples for "
        f"{summary.epochs} epochs; validation relative MAE "
        f"{summary.validation_relative_mae:.4f}"
    )


@app.command("neural-predict")
def neural_predict(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="One clean square-on object image.",
    ),
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Trained CadPro .npz checkpoint.",
    ),
    width_mm: float = typer.Option(
        ...,
        min=0.001,
        help="Measured real-world width of the visible object.",
    ),
    output: Path = typer.Option(
        Path("neural-model.step"),
        "--output",
        "-o",
        help="Destination STEP file.",
    ),
) -> None:
    """Predict hidden depth from one image and build a validated STEP solid."""
    from cadpro.neural import NeuralModelError, predict_step

    try:
        result, prediction = predict_step(
            input_path,
            checkpoint,
            measured_width_mm=width_mm,
            output_path=output,
        )
    except (FileNotFoundError, ValueError, RuntimeError, NeuralModelError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"Created {result}: predicted depth {prediction.depth_mm:g} mm "
        f"({prediction.depth_ratio:.4f} x measured width), heuristic confidence "
        f"{prediction.confidence_score:.2f}. Verify the estimate before manufacturing."
    )


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Network interface for the website."),
    port: int = typer.Option(8000, min=1, max=65535, help="Website port."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the website after launch."),
) -> None:
    """Launch the photo, photo-orbit, and turntable-video reconstruction website."""
    import uvicorn

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url_host = f"[{display_host}]" if ":" in display_host else display_host
    url = f"http://{url_host}:{port}"
    os.environ.setdefault("CADPRO_PUBLIC_ORIGIN", url)
    os.environ.setdefault("CADPRO_STORAGE_DIR", str(_default_storage_directory()))
    if open_browser:
        browser_timer = threading.Timer(1.0, lambda: webbrowser.open(url))
        browser_timer.daemon = True
        browser_timer.start()
    typer.echo(f"CadPro is running at {url}")
    uvicorn.run(
        "cadpro.web:app",
        host=host,
        port=port,
        log_level="info",
        proxy_headers=False,
    )


def _default_storage_directory() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "CadPro"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "cadpro"


if __name__ == "__main__":
    app()
