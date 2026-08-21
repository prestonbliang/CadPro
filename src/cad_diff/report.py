from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

from cad_diff.diff_model import DiffReport

_STATUS_STYLE = {
    "unchanged": "dim",
    "modified": "yellow",
    "added": "green",
    "removed": "red",
}

_UNICODE_REPORT_CHARS = "→Δ²³┏━┓┃┗┛"


def _can_render_unicode(console: Console) -> bool:
    try:
        _UNICODE_REPORT_CHARS.encode(console.encoding or "utf-8")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _safe_text(console: Console, value: object) -> str:
    """Replace only characters the active terminal encoding cannot emit."""
    text = str(value)
    encoding = console.encoding or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        return text


def print_report(report: DiffReport, console: Console | None = None) -> None:
    console = console or Console()
    unicode_output = _can_render_unicode(console)
    arrow = "→" if unicode_output else "->"
    delta = "Δ" if unicode_output else "delta"
    cubic_mm = "mm³" if unicode_output else "mm^3"
    square_mm = "mm²" if unicode_output else "mm^2"
    table = Table(
        title=_safe_text(console, f"{report.base_path}  {arrow}  {report.modified_path}"),
        box=box.HEAVY_HEAD if unicode_output else box.ASCII,
    )
    table.add_column("Solid")
    table.add_column("Status")
    table.add_column(f"Volume {delta} ({cubic_mm})", justify="right")
    table.add_column(f"Surface {delta} ({square_mm})", justify="right")

    for diff in report.solids:
        name = (diff.modified or diff.base).name
        style = _STATUS_STYLE[diff.status]
        vol_delta = "" if diff.volume_delta is None else f"{diff.volume_delta:+.3f}"
        area_delta = "" if diff.surface_area_delta is None else f"{diff.surface_area_delta:+.3f}"
        table.add_row(_safe_text(console, name), f"[{style}]{diff.status}[/{style}]", vol_delta, area_delta)

    console.print(table)

    for solid_face_diff in report.face_diffs:
        _print_face_detail(solid_face_diff, console)


def _print_face_detail(solid_face_diff, console: Console) -> None:
    unicode_output = _can_render_unicode(console)
    name = _safe_text(console, (solid_face_diff.solid.modified or solid_face_diff.solid.base).name)
    cubic_mm = "mm³" if unicode_output else "mm^3"
    delta = "Δ" if unicode_output else "delta"
    square_mm = "mm²" if unicode_output else "mm^2"
    b = solid_face_diff.boolean
    console.print(
        f"\n[bold]{name}[/bold] face detail  "
        f"[dim](boolean cross-check: +{b.added_volume:.3f} / -{b.removed_volume:.3f} {cubic_mm}, "
        f"tier 5 ground truth)[/dim]"
    )

    table = Table(box=box.HEAVY_HEAD if unicode_output else box.ASCII)
    table.add_column("Face")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Tier")
    table.add_column(f"Area {delta} ({square_mm})", justify="right")
    table.add_column(f"Param {delta}")

    for face_diff in sorted(
        solid_face_diff.faces,
        key=lambda d: (d.status != "modified", d.status, (d.base or d.modified).index),
    ):
        fp = face_diff.modified or face_diff.base
        style = _STATUS_STYLE[face_diff.status]
        area_delta = "" if face_diff.area_delta is None else f"{face_diff.area_delta:+.3f}"
        param_deltas = _safe_text(console, ", ".join(f"{k}: {v:+.3f}" for k, v in face_diff.param_deltas.items()))
        table.add_row(
            str(fp.index),
            _safe_text(console, fp.surface_type),
            f"[{style}]{face_diff.status}[/{style}]",
            _safe_text(console, face_diff.tier),
            area_delta,
            param_deltas,
        )

    console.print(table)
