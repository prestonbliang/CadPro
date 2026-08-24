"""Native and Python dependency detection with actionable, truthful status."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
from typing import Final

from cadpro.scan.models import ToolCapability, ToolchainCapabilities


_INSTALL_HINTS: Final[dict[str, str]] = {
    "ffmpeg": (
        "Install FFmpeg and FFprobe from https://ffmpeg.org/download.html, then ensure both "
        "executables are on PATH or set CADPRO_FFMPEG_PATH and CADPRO_FFPROBE_PATH."
    ),
    "ffprobe": (
        "FFprobe ships with FFmpeg: https://ffmpeg.org/download.html. Put it on PATH or set "
        "CADPRO_FFPROBE_PATH."
    ),
    "colmap": (
        "Install an official COLMAP release from https://colmap.github.io/install.html and put "
        "colmap on PATH, or set CADPRO_COLMAP_PATH."
    ),
    "openmvs": (
        "Install OpenMVS from https://github.com/cdcseacave/openMVS/wiki/Building and expose "
        "InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, and RefineMesh on PATH; add "
        "TextureMesh for camera-derived textures."
    ),
    "blender": (
        "Optional: install Blender from https://www.blender.org/download/ and put blender on PATH."
    ),
    "trimesh": "Install the project dependencies with: python -m pip install -e .",
    "ocp": "Install the project dependencies with: python -m pip install -e .",
}


def _resolved_executable(
    name: str,
    environment_variable: str,
    conventional_candidates: tuple[Path, ...] = (),
) -> str | None:
    configured = os.environ.get(environment_variable, "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve())
        return None
    discovered = shutil.which(name)
    if discovered:
        return discovered
    for candidate in conventional_candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _probe_executable(
    *,
    key: str,
    display_name: str,
    executable_name: str,
    environment_variable: str,
    version_arguments: tuple[str, ...],
    conventional_candidates: tuple[Path, ...] = (),
) -> ToolCapability:
    executable = _resolved_executable(
        executable_name, environment_variable, conventional_candidates
    )
    if executable and key == "colmap" and Path(executable).suffix.lower() in {".bat", ".cmd"}:
        direct = Path(executable).parent / "bin" / "colmap.exe"
        if direct.is_file():
            executable = str(direct.resolve())
    configured = os.environ.get(environment_variable, "").strip()
    if executable is None:
        reason = (
            f"{environment_variable} does not point to a file."
            if configured
            else f"{display_name} was not found on PATH."
        )
        return ToolCapability(
            name=display_name,
            available=False,
            reason=reason,
            install_hint=_INSTALL_HINTS[key],
        )
    try:
        completed = subprocess.run(
            [executable, *version_arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return ToolCapability(
            name=display_name,
            available=False,
            executable=executable,
            reason=f"The executable could not be started: {type(error).__name__}.",
            install_hint=_INSTALL_HINTS[key],
        )
    output = " ".join((completed.stdout or "").split())
    version = output[:300] or f"exit code {completed.returncode}"
    # Some OpenMVS programs return a non-zero status after printing usage with --help.
    probe_text = output.lower()
    recognizable_help = bool(output) and any(
        token in probe_text
        for token in ("usage", "options", "openmvs")
    )
    available = completed.returncode == 0 or recognizable_help
    return ToolCapability(
        name=display_name,
        available=available,
        executable=executable,
        version=version if available else None,
        reason=None if available else f"Version probe exited with code {completed.returncode}.",
        install_hint=None if available else _INSTALL_HINTS[key],
    )


def _python_capability(key: str, display_name: str, module_name: str) -> ToolCapability:
    available = importlib.util.find_spec(module_name) is not None
    version: str | None = None
    if available:
        try:
            module = __import__(module_name)
            version = str(getattr(module, "__version__", "installed"))
        except Exception:
            version = "installed"
    return ToolCapability(
        name=display_name,
        available=available,
        version=version,
        reason=None if available else f"Python module {module_name!r} is not installed.",
        install_hint=None if available else _INSTALL_HINTS[key],
    )


def detect_toolchain() -> ToolchainCapabilities:
    """Probe all optional tools without mutating the machine or running reconstruction."""

    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    blender_candidates = tuple(
        sorted(
            (program_files / "Blender Foundation").glob("Blender */blender.exe"),
            reverse=True,
        )
    )
    tools: dict[str, ToolCapability] = {
        "ffmpeg": _probe_executable(
            key="ffmpeg",
            display_name="FFmpeg",
            executable_name="ffmpeg",
            environment_variable="CADPRO_FFMPEG_PATH",
            version_arguments=("-version",),
        ),
        "ffprobe": _probe_executable(
            key="ffprobe",
            display_name="FFprobe",
            executable_name="ffprobe",
            environment_variable="CADPRO_FFPROBE_PATH",
            version_arguments=("-version",),
        ),
        "colmap": _probe_executable(
            key="colmap",
            display_name="COLMAP",
            executable_name="colmap",
            environment_variable="CADPRO_COLMAP_PATH",
            version_arguments=("-h",),
        ),
        "interface_colmap": _probe_executable(
            key="openmvs",
            display_name="OpenMVS InterfaceCOLMAP",
            executable_name="InterfaceCOLMAP",
            environment_variable="CADPRO_OPENMVS_INTERFACE_PATH",
            version_arguments=("--help",),
        ),
        "densify_point_cloud": _probe_executable(
            key="openmvs",
            display_name="OpenMVS DensifyPointCloud",
            executable_name="DensifyPointCloud",
            environment_variable="CADPRO_OPENMVS_DENSIFY_PATH",
            version_arguments=("--help",),
        ),
        "reconstruct_mesh": _probe_executable(
            key="openmvs",
            display_name="OpenMVS ReconstructMesh",
            executable_name="ReconstructMesh",
            environment_variable="CADPRO_OPENMVS_RECONSTRUCT_PATH",
            version_arguments=("--help",),
        ),
        "refine_mesh": _probe_executable(
            key="openmvs",
            display_name="OpenMVS RefineMesh",
            executable_name="RefineMesh",
            environment_variable="CADPRO_OPENMVS_REFINE_PATH",
            version_arguments=("--help",),
        ),
        "texture_mesh": _probe_executable(
            key="openmvs",
            display_name="OpenMVS TextureMesh",
            executable_name="TextureMesh",
            environment_variable="CADPRO_OPENMVS_TEXTURE_PATH",
            version_arguments=("--help",),
        ),
        "blender": _probe_executable(
            key="blender",
            display_name="Blender",
            executable_name="blender",
            environment_variable="CADPRO_BLENDER_PATH",
            version_arguments=("--version",),
            conventional_candidates=blender_candidates,
        ),
        "trimesh": _python_capability("trimesh", "trimesh", "trimesh"),
        "ocp": _python_capability("ocp", "OpenCascade Python bindings", "OCP"),
    }
    openmvs_ready = all(
        tools[name].available
        for name in (
            "interface_colmap",
            "densify_point_cloud",
            "reconstruct_mesh",
            "refine_mesh",
        )
    )
    return ToolchainCapabilities(
        tools=tools,
        photo_reconstruction=tools["colmap"].available,
        video_ingest=tools["ffmpeg"].available and tools["ffprobe"].available,
        dense_reconstruction=tools["colmap"].available or openmvs_ready,
        texture_generation=openmvs_ready and tools["texture_mesh"].available,
        mesh_processing=tools["trimesh"].available,
        analytic_cad=tools["ocp"].available,
    )


@lru_cache(maxsize=1)
def default_toolchain() -> ToolchainCapabilities:
    """Cache startup probes while leaving detect_toolchain directly testable."""

    return detect_toolchain()


def executable(capabilities: ToolchainCapabilities, key: str) -> str:
    """Return a previously probed executable or raise an actionable error."""

    capability = capabilities.tools[key]
    if not capability.available or not capability.executable:
        detail = capability.install_hint or capability.reason or "Dependency unavailable."
        raise RuntimeError(f"{capability.name} is unavailable. {detail}")
    return capability.executable


def redacted_toolchain(capabilities: ToolchainCapabilities) -> ToolchainCapabilities:
    """Remove local filesystem paths before capabilities leave the server boundary."""

    return capabilities.model_copy(
        update={
            "tools": {
                key: capability.model_copy(update={"executable": None})
                for key, capability in capabilities.tools.items()
            }
        }
    )
