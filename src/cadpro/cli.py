from __future__ import annotations

from pathlib import Path

import typer

from cadpro.pipeline import convert_media


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Convert pictures and videos into solid STEP CAD models."""


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


if __name__ == "__main__":
    app()
