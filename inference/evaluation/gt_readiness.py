"""Phase 2E-R11 — Ground Truth Dataset Readiness Tooling.

Provides a deterministic, human-readable readiness report for a
GroundTruthDataset before R10 calibration can execute.

Functions:
    load_gt_dataset(source, source_name)
        Load a GroundTruthDataset from a JSON string, a file path, or
        an already-parsed dict.  Returns (dataset_or_None, issues_list).

    assess_gt_dataset_readiness(dataset)
        Run all existing schema checks and produce a GTDatasetReadinessReport.
        Pure function — no file I/O, no fabricated data.

    validate_for_r10(dataset)
        Strict gate: returns a dict with "ok" and "blockers" keys.
        R10 is blocked if any schema error is present, even when counts
        look fine.  This is the function to call before launching R10.

The report contains only counts and error strings derived from the input
dataset.  No operating-point values, recall, precision, or any other
derived metric is fabricated.  If the dataset is empty or invalid the
report says so plainly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from inference.evaluation.ground_truth import (
    DatasetManifest,
    GroundTruthDataset,
    validate_dataset,
)

__all__ = [
    "GTDatasetReadinessReport",
    "load_gt_dataset",
    "assess_gt_dataset_readiness",
    "validate_for_r10",
]


# ---------------------------------------------------------------------------
# Readiness report
# ---------------------------------------------------------------------------


@dataclass
class GTDatasetReadinessReport:
    """Deterministic, JSON-serializable readiness assessment.

    Fields
    ------
    usable_for_r10 : bool
        True only when no blockers remain.
    video_count, incident_count, normal_period_count, ambiguous_period_count : int
        Raw counts from the input dataset.
    schema_error_count : int
        Number of validation issues returned by validate_dataset().
    schema_errors : List[str]
        Human-readable validation issue strings (same as validate_dataset()).
    r10_blockers : List[str]
        High-level blocker strings (e.g. "no GT incidents",
        "schema errors present").
    unblock_requirements : List[str]
        Human-readable steps required to make the dataset usable for R10.
    """
    usable_for_r10: bool
    video_count: int
    incident_count: int
    normal_period_count: int
    ambiguous_period_count: int
    schema_error_count: int
    schema_errors: List[str]
    r10_blockers: List[str]
    unblock_requirements: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "usable_for_r10": self.usable_for_r10,
            "video_count": self.video_count,
            "incident_count": self.incident_count,
            "normal_period_count": self.normal_period_count,
            "ambiguous_period_count": self.ambiguous_period_count,
            "schema_error_count": self.schema_error_count,
            "schema_errors": list(self.schema_errors),
            "r10_blockers": list(self.r10_blockers),
            "unblock_requirements": list(self.unblock_requirements),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Load a dataset from various sources
# ---------------------------------------------------------------------------


def load_gt_dataset(
    source: Any,
    source_name: str = "",
) -> Tuple[Optional[GroundTruthDataset], List[str]]:
    """Load a GroundTruthDataset from ``source``.

    ``source`` may be:
      - a JSON string (parsed via DatasetManifest.from_json)
      - a filesystem path to a JSON file
      - a dict (parsed via DatasetManifest.from_dict)

    Returns (dataset_or_None, issues).  ``issues`` is empty on success.
    """
    issues: List[str] = []

    # If source is already a dict
    if isinstance(source, dict):
        try:
            manifest = DatasetManifest.from_dict(source)
        except Exception as exc:
            issues.append(f"Manifest parse error from dict source '{source_name}': {exc}")
            return None, issues
        return GroundTruthDataset(manifests=[manifest]), []

    if not isinstance(source, str):
        issues.append(f"Unsupported source type: {type(source).__name__} (expected str or dict)")
        return None, issues

    # Try to parse as JSON string first
    stripped = source.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(source)
        except json.JSONDecodeError as exc:
            issues.append(f"Invalid JSON in source '{source_name}': {exc}")
            return None, issues
        if isinstance(parsed, dict):
            try:
                manifest = DatasetManifest.from_dict(parsed)
            except Exception as exc:
                issues.append(f"Manifest schema error in JSON source '{source_name}': {exc}")
                return None, issues
            return GroundTruthDataset(manifests=[manifest]), []
        issues.append(
            f"JSON source '{source_name}' parsed to a non-dict value; "
            "expected a DatasetManifest object"
        )
        return None, issues

    # Otherwise treat as a file path
    if not os.path.isfile(source):
        issues.append(
            f"Source file not found: '{source}' "
            f"(source_name='{source_name}')"
        )
        return None, issues

    try:
        with open(source, "r", encoding="utf-8") as fh:
            manifest = DatasetManifest.from_json(fh.read())
    except json.JSONDecodeError as exc:
        issues.append(f"Invalid JSON in file '{source}': {exc}")
        return None, issues
    except FileNotFoundError as exc:
        issues.append(f"FileNotFoundError for '{source}': {exc}")
        return None, issues
    except Exception as exc:
        issues.append(f"Failed to load dataset from '{source}': {exc}")
        return None, issues

    return GroundTruthDataset(manifests=[manifest]), []


# ---------------------------------------------------------------------------
# Assess readiness
# ---------------------------------------------------------------------------


def assess_gt_dataset_readiness(dataset: GroundTruthDataset) -> GTDatasetReadinessReport:
    """Run all schema checks on ``dataset`` and return a readiness report.

    Pure function: no file I/O, no fabricated values.  All content in the
    report is derived from ``dataset`` and the existing ``validate_dataset``
    implementation.
    """
    all_videos = dataset.all_videos()
    all_incidents = dataset.all_incidents()
    all_normal: List[Any] = []
    all_ambiguous: List[Any] = []
    for m in dataset.manifests:
        all_normal.extend(m.normal_periods)
        all_ambiguous.extend(m.ambiguous_periods)

    schema_errors = validate_dataset(dataset)
    schema_error_count = len(schema_errors)

    blockers: List[str] = []
    unblock: List[str] = []

    if not all_videos:
        blockers.append("no source videos in dataset")
    if not all_incidents:
        blockers.append("no GT incidents in dataset")
    if schema_error_count > 0:
        blockers.append(f"schema errors present ({schema_error_count} issue(s))")

    unblock.append(
        "Provide a valid GroundTruthDataset manifest file with at least one video."
    )
    if not all_incidents:
        unblock.append(
            "Annotate at least one GT incident (SUSPICIOUS, NORMAL, or AMBIGUOUS)."
        )
    if schema_error_count > 0:
        unblock.append(
            "Resolve all schema errors reported by validate_dataset() "
            "before running R10 calibration."
        )

    usable = len(all_incidents) > 0 and schema_error_count == 0

    return GTDatasetReadinessReport(
        usable_for_r10=usable,
        video_count=len(all_videos),
        incident_count=len(all_incidents),
        normal_period_count=len(all_normal),
        ambiguous_period_count=len(all_ambiguous),
        schema_error_count=schema_error_count,
        schema_errors=schema_errors,
        r10_blockers=blockers,
        unblock_requirements=unblock,
    )


# ---------------------------------------------------------------------------
# R10 gate
# ---------------------------------------------------------------------------


def validate_for_r10(dataset: GroundTruthDataset) -> Dict[str, Any]:
    """Strict gate: R10 may only run when this returns ok=True.

    Stricter than counts alone: a dataset with valid counts but schema
    errors (duplicate IDs, out-of-range intervals, missing video references)
    is still blocked.

    Returns
    -------
    dict with keys:
        ok        : bool
        blockers  : List[str]  (empty when ok=True)
        report    : Dict[str, Any]  (full GTDatasetReadinessReport.to_dict())
    """
    report = assess_gt_dataset_readiness(dataset)
    return {
        "ok": report.usable_for_r10,
        "blockers": list(report.r10_blockers),
        "report": report.to_dict(),
    }
