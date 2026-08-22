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
    """Create measured STEP CAD models and compare CAD geometry."""


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


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Network interface for the website."),
    port: int = typer.Option(8000, min=1, max=65535, help="Website port."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the website after launch."),
) -> None:
    """Launch the single-image profile-extrusion website."""
    import uvicorn

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url_host = f"[{display_host}]" if ":" in display_host else display_host
    url = f"http://{url_host}:{port}"
    os.environ.setdefault("CADPRO_PUBLIC_ORIGIN", url)
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


if __name__ == "__main__":
    app()
