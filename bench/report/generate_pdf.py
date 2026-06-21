"""
Optional PDF export via weasyprint.
Skipped gracefully if weasyprint is not installed.
"""
from __future__ import annotations
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_pdf(html_path: Path, pdf_path: Path) -> bool:
    """
    Convert HTML report to PDF using weasyprint.
    Returns True on success, False if weasyprint is unavailable.
    """
    try:
        from weasyprint import HTML
    except ImportError:
        logger.info("weasyprint not installed — skipping PDF export. "
                    "Install with: pip install weasyprint")
        return False

    try:
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        logger.info("PDF report written to %s", pdf_path)
        return True
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)
        return False
