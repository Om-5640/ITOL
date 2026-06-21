"""
§12 / Step 16 CLI and Helm invariants.

test_serve_help_works          — `itol serve --help` succeeds
test_new_adapter_generates_file — `itol new-adapter --name X` writes a valid file
test_new_adapter_class_name    — generated class has correct name
test_entry_point_group_in_pyproject — itol.adapters entry-point group declared
test_helm_lint                 — helm lint passes (skipped if helm absent)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# CLI: serve --help
# ---------------------------------------------------------------------------

def test_serve_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "itol.cli", "serve", "--help"],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert result.returncode == 0, f"serve --help failed:\n{result.stderr}"
    assert "port" in result.stdout.lower() or "serve" in result.stdout.lower()


# ---------------------------------------------------------------------------
# CLI: new-adapter scaffold
# ---------------------------------------------------------------------------

def test_new_adapter_generates_file(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "itol.cli", "new-adapter",
         "--name", "TestProvider", "--output", str(tmp_path)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert result.returncode == 0, f"new-adapter failed:\n{result.stderr}"
    out_file = tmp_path / "testprovider.py"
    assert out_file.exists(), f"Expected {out_file} to be created"


def test_new_adapter_class_name(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "itol.cli", "new-adapter",
         "--name", "my-cool-provider", "--output", str(tmp_path)],
        capture_output=True, cwd=str(_REPO),
    )
    out_file = tmp_path / "my_cool_provider.py"
    assert out_file.exists(), f"Expected slug-named file at {out_file}"
    content = out_file.read_text(encoding="utf-8")
    assert "MyCoolProviderAdapter" in content
    assert "OpenAICompatibleAdapter" in content


def test_new_adapter_does_not_overwrite_without_force(tmp_path):
    args = [sys.executable, "-m", "itol.cli", "new-adapter",
            "--name", "Duplicate", "--output", str(tmp_path)]
    subprocess.run(args, capture_output=True, cwd=str(_REPO))
    result = subprocess.run(args, capture_output=True, text=True, cwd=str(_REPO))
    assert result.returncode != 0 or "already exists" in result.stderr


# ---------------------------------------------------------------------------
# Entry-point group
# ---------------------------------------------------------------------------

def test_entry_point_group_in_pyproject():
    pyproject = _REPO / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    assert 'itol.adapters' in content, (
        "pyproject.toml must declare [project.entry-points.\"itol.adapters\"] "
        "for third-party adapter discovery (§8.3)"
    )


def test_all_builtin_adapters_in_entry_points():
    pyproject = _REPO / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    for adapter in ("openai", "anthropic", "mistral", "groq", "ollama", "cohere"):
        assert adapter in content, f"Adapter '{adapter}' missing from entry-points"


# ---------------------------------------------------------------------------
# Helm lint (skipped if helm not available)
# ---------------------------------------------------------------------------

@pytest.mark.helm
def test_helm_lint():
    import shutil
    if not shutil.which("helm"):
        pytest.skip("helm not installed")

    result = subprocess.run(
        ["helm", "lint", str(_REPO / "deploy" / "helm")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"helm lint failed:\n{result.stdout}\n{result.stderr}"
    )
