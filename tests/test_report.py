from io import BytesIO, TextIOWrapper

from rich.console import Console

from cad_diff.diff_model import (
    BooleanCrossCheck,
    DiffReport,
    FaceDiff,
    FaceFingerprint,
    SolidDiff,
    SolidFaceDiff,
    SolidFingerprint,
)
from cad_diff.report import print_report


def _report() -> DiffReport:
    solid = SolidFingerprint(
        name="bracket",
        volume=10.0,
        surface_area=20.0,
        center_of_mass=(0.0, 0.0, 0.0),
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
    )
    solid_diff = SolidDiff(status="unchanged", base=solid, modified=solid)
    face = FaceFingerprint(
        index=1,
        surface_type="Plane",
        area=1.0,
        centroid=(0.0, 0.0, 0.0),
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 0.0),
        adjacent=frozenset(),
        params={"offset": 0.0},
    )
    return DiffReport(
        base_path="base.step",
        modified_path="modified.step",
        solids=[solid_diff],
        face_diffs=[
            SolidFaceDiff(
                solid=solid_diff,
                faces=[FaceDiff(status="unchanged", tier="T1", base=face, modified=face)],
                boolean=BooleanCrossCheck(added_volume=0.0, removed_volume=0.0, common_volume=10.0),
            )
        ],
    )


def test_cp1252_console_uses_readable_ascii_without_encoding_errors():
    output = BytesIO()
    stream = TextIOWrapper(output, encoding="cp1252", errors="strict")
    console = Console(file=stream, color_system=None, force_terminal=False, width=100)

    print_report(_report(), console)
    stream.flush()
    rendered = output.getvalue().decode("cp1252")

    assert "base.step  ->  modified.step" in rendered
    assert "Volume delta (mm^3)" in rendered
    assert "Surface delta (mm^2)" in rendered
    assert "boolean cross-check: +0.000 / -0.000 mm^3" in rendered
    assert "Area delta (mm^2)" in rendered
