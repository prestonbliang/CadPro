from __future__ import annotations

from rich.console import Console
from rich.table import Table

from cad_diff.diff_model import DiffReport

_STATUS_STYLE = {
    "unchanged": "dim",
    "modified": "yellow",
    "added": "green",
    "removed": "red",
}


def print_report(report: DiffReport, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"{report.base_path}  →  {report.modified_path}")
    table.add_column("Solid")
    table.add_column("Status")
    table.add_column("Volume Δ (mm³)", justify="right")
    table.add_column("Surface Δ (mm²)", justify="right")

    for diff in report.solids:
        name = (diff.modified or diff.base).name
        style = _STATUS_STYLE[diff.status]
        vol_delta = "" if diff.volume_delta is None else f"{diff.volume_delta:+.3f}"
        area_delta = "" if diff.surface_area_delta is None else f"{diff.surface_area_delta:+.3f}"
        table.add_row(name, f"[{style}]{diff.status}[/{style}]", vol_delta, area_delta)

    console.print(table)

    for solid_face_diff in report.face_diffs:
        _print_face_detail(solid_face_diff, console)


def _print_face_detail(solid_face_diff, console: Console) -> None:
    name = (solid_face_diff.solid.modified or solid_face_diff.solid.base).name
    b = solid_face_diff.boolean
    console.print(
        f"\n[bold]{name}[/bold] face detail  "
        f"[dim](boolean cross-check: +{b.added_volume:.3f} / -{b.removed_volume:.3f} mm³, "
        f"tier 5 ground truth)[/dim]"
    )

    table = Table()
    table.add_column("Face")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Tier")
    table.add_column("Area Δ (mm²)", justify="right")
    table.add_column("Param Δ")

    for face_diff in sorted(
        solid_face_diff.faces,
        key=lambda d: (d.status != "modified", d.status, (d.base or d.modified).index),
    ):
        fp = face_diff.modified or face_diff.base
        style = _STATUS_STYLE[face_diff.status]
        area_delta = "" if face_diff.area_delta is None else f"{face_diff.area_delta:+.3f}"
        param_deltas = ", ".join(f"{k}: {v:+.3f}" for k, v in face_diff.param_deltas.items())
        table.add_row(
            str(fp.index), fp.surface_type, f"[{style}]{face_diff.status}[/{style}]", face_diff.tier, area_delta, param_deltas
        )

    console.print(table)
