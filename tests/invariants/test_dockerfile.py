"""
§14.3 Deploy file syntax tests.

Docker-dependent tests are SKIPPED BY DEFAULT.
Set DOCKER_AVAILABLE=1 in the environment to run them.

Syntax-only tests (YAML parse, Dockerfile token check) always run.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEPLOY_DIR = _REPO_ROOT / "deploy"
_DOCKERFILE = _DEPLOY_DIR / "Dockerfile"
_COMPOSE_FILE = _DEPLOY_DIR / "docker-compose.yml"
_HELM_DIR = _DEPLOY_DIR / "helm"

_DOCKER_AVAILABLE = os.environ.get("DOCKER_AVAILABLE", "").strip() in ("1", "true", "yes")


# ===========================================================================
# Dockerfile syntax (no Docker required)
# ===========================================================================

class TestDockerfileSyntax:

    def test_dockerfile_exists(self):
        assert _DOCKERFILE.exists(), f"Dockerfile not found at {_DOCKERFILE}"

    def test_dockerfile_has_from_directive(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        from_lines = [l for l in content.splitlines() if l.strip().startswith("FROM")]
        assert len(from_lines) >= 2, (
            "Dockerfile must be multi-stage (at least 2 FROM directives)"
        )

    def test_dockerfile_runtime_stage_uses_slim_base(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "python:3.11-slim" in content, (
            "Runtime stage must use python:3.11-slim base image"
        )

    def test_dockerfile_has_healthcheck(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content, (
            "Dockerfile must include a HEALTHCHECK directive"
        )

    def test_dockerfile_exposes_port_8080(self):
        content = _DOCKERFILE.read_text(encoding="utf-8")
        assert "EXPOSE 8080" in content, (
            "Dockerfile must EXPOSE port 8080"
        )


# ===========================================================================
# Docker Compose YAML parse (no Docker required)
# ===========================================================================

class TestDockerComposeSyntax:

    def test_compose_file_exists(self):
        assert _COMPOSE_FILE.exists(), (
            f"docker-compose.yml not found at {_COMPOSE_FILE}"
        )

    def test_compose_file_is_valid_yaml(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed — skipping YAML parse test")

        content = _COMPOSE_FILE.read_text(encoding="utf-8")
        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            pytest.fail(f"docker-compose.yml is not valid YAML: {exc}")

        assert parsed is not None
        assert "services" in parsed, "docker-compose.yml must have a 'services' key"
        assert "itol" in parsed["services"], (
            "docker-compose.yml must define an 'itol' service"
        )


# ===========================================================================
# Helm chart YAML parse (no Docker/Helm required)
# ===========================================================================

class TestHelmChartSyntax:

    def test_chart_yaml_exists(self):
        assert (_HELM_DIR / "Chart.yaml").exists()

    def test_values_yaml_exists(self):
        assert (_HELM_DIR / "values.yaml").exists()

    def test_deployment_template_exists(self):
        assert (_HELM_DIR / "templates" / "deployment.yaml").exists()

    def test_service_template_exists(self):
        assert (_HELM_DIR / "templates" / "service.yaml").exists()

    def test_chart_yaml_is_valid_yaml(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        content = (_HELM_DIR / "Chart.yaml").read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert parsed["name"] == "itol"
        assert "version" in parsed

    def test_values_yaml_is_valid_yaml(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        content = (_HELM_DIR / "values.yaml").read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert "replicaCount" in parsed
        assert "resources" in parsed


# ===========================================================================
# Docker build test (skipped unless DOCKER_AVAILABLE=1)
# ===========================================================================

@pytest.mark.docker
@pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="DOCKER_AVAILABLE not set — skipping Docker build test",
)
class TestDockerBuild:

    def test_dockerfile_builds_successfully(self):
        """
        Build the ITOL Docker image and assert exit code 0.
        Run with: DOCKER_AVAILABLE=1 pytest -m docker
        """
        result = subprocess.run(
            [
                "docker", "build",
                "-t", "itol-test:ci",
                "-f", str(_DOCKERFILE),
                str(_REPO_ROOT),
                "--no-cache",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"Docker build failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_dockerfile_hadolint_passes(self):
        """
        Run hadolint (if available) to check for common Dockerfile issues.
        Requires: docker pull hadolint/hadolint
        """
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "-i", "hadolint/hadolint"],
                input=_DOCKERFILE.read_text(encoding="utf-8"),
                capture_output=True,
                text=True,
                timeout=60,
            )
            # DL3008, DL3009 are common warnings for apt-get; allow them
            lines = [l for l in result.stdout.splitlines()
                     if l and not any(code in l for code in ("DL3008", "DL3009", "DL3013"))]
            assert not lines, f"hadolint found issues:\n" + "\n".join(lines)
        except FileNotFoundError:
            pytest.skip("Docker not available for hadolint check")
