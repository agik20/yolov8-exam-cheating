"""Phase 2E-R11 — Ground Truth Dataset Readiness Tooling.

Covers:
- GTDatasetReadinessReport structure and field correctness
- load_gt_dataset() — success, file-not-found, invalid JSON, non-JSON manifest
- assess_gt_dataset_readiness() — usable, schema errors block, empty dataset
- validate_for_r10() — strict gate: schema errors block even with counts present
- Determinism: same input → same output, JSON serializable
- No file I/O in assess_gt_dataset_readiness / validate_for_r10
- No fabricated data: empty dataset reports usable=False, no synthetic values
- R10 unblock criteria documented in the report
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from inference.evaluation.ground_truth import (
    GTLabel,
    GroundTruthDataset,
    GroundTruthIncident,
    NormalPeriod,
    AmbiguousPeriod,
    SourceVideoMetadata,
    DatasetManifest,
)
from inference.evaluation.gt_readiness import (
    GTDatasetReadinessReport,
    assess_gt_dataset_readiness,
    load_gt_dataset,
    validate_for_r10,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_manifest_valid(
    n_videos: int = 3,
    n_incidents_per_video: int = 2,
    include_normal: bool = True,
    include_ambiguous: bool = True,
) -> DatasetManifest:
    videos = [
        SourceVideoMetadata(
            video_id=f"vid_{i}",
            reference=f"ref_{i}",
            duration_seconds=60.0,
            source_fps=30.0,
        )
        for i in range(n_videos)
    ]
    incidents = [
        GroundTruthIncident(
            video_id=f"vid_{i}",
            incident_id=f"inc_{i}_{j}",
            label=GTLabel.SUSPICIOUS,
            start_video_time=j * 10.0,
            end_video_time=j * 10.0 + 5.0,
            behavior_class="provide-object",
        )
        for i in range(n_videos)
        for j in range(n_incidents_per_video)
    ]
    normal = []
    ambiguous = []
    if include_normal:
        normal = [
            NormalPeriod(
                video_id=f"vid_{i}",
                period_id=f"norm_{i}",
                start_video_time=50.0,
                end_video_time=55.0,
            )
            for i in range(n_videos)
        ]
    if include_ambiguous:
        ambiguous = [
            AmbiguousPeriod(
                video_id=f"vid_{i}",
                period_id=f"ambig_{i}",
                start_video_time=0.0,
                end_video_time=2.0,
                reason="occlusion",
            )
            for i in range(n_videos)
        ]
    return DatasetManifest(
        version="test-v1",
        created_at="2026-01-01T00:00:00Z",
        videos=videos,
        incidents=incidents,
        normal_periods=normal,
        ambiguous_periods=ambiguous,
    )


def _mk_gt_dataset(valid: bool = True, **kw) -> GroundTruthDataset:
    return GroundTruthDataset(manifests=[_mk_manifest_valid(**kw)] if valid else [])


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


class TestR11ReportStructure(unittest.TestCase):

    def test_valid_dataset_report_fields(self):
        gt = _mk_gt_dataset()
        report = assess_gt_dataset_readiness(gt)
        self.assertIsInstance(report, GTDatasetReadinessReport)
        self.assertTrue(report.usable_for_r10)
        self.assertEqual(report.video_count, 3)
        self.assertEqual(report.incident_count, 6)
        self.assertEqual(report.normal_period_count, 3)
        self.assertEqual(report.ambiguous_period_count, 3)
        self.assertEqual(report.schema_error_count, 0)
        self.assertEqual(report.r10_blockers, [])

    def test_empty_dataset_report_fields(self):
        gt = GroundTruthDataset()
        report = assess_gt_dataset_readiness(gt)
        self.assertFalse(report.usable_for_r10)
        self.assertEqual(report.video_count, 0)
        self.assertEqual(report.incident_count, 0)
        self.assertIn("no GT incidents", " ".join(report.r10_blockers))
        self.assertIsInstance(report.unblock_requirements, list)

    def test_report_to_dict_json_serializable(self):
        gt = _mk_gt_dataset()
        report = assess_gt_dataset_readiness(gt)
        json.dumps(report.to_dict())

    def test_report_deterministic(self):
        gt = _mk_gt_dataset()
        r1 = assess_gt_dataset_readiness(gt)
        r2 = assess_gt_dataset_readiness(gt)
        self.assertEqual(r1.to_dict(), r2.to_dict())


# ---------------------------------------------------------------------------
# Schema errors block R10
# ---------------------------------------------------------------------------


class TestR11SchemaErrorsBlock(unittest.TestCase):

    def test_incident_exceeding_video_duration_is_flagged(self):
        """An incident with end_video_time > video duration is a schema error."""
        manifest = DatasetManifest(
            version="t",
            created_at="2026-01-01T00:00:00Z",
            videos=[
                SourceVideoMetadata(
                    video_id="v1",
                    reference="v1",
                    duration_seconds=10.0,
                    source_fps=30.0,
                )
            ],
            incidents=[
                GroundTruthIncident(
                    video_id="v1",
                    incident_id="i1",
                    label=GTLabel.SUSPICIOUS,
                    start_video_time=5.0,
                    end_video_time=15.0,
                    behavior_class="provide-object",
                )
            ],
        )
        gt = GroundTruthDataset(manifests=[manifest])
        report = assess_gt_dataset_readiness(gt)
        self.assertFalse(report.usable_for_r10)
        self.assertGreater(report.schema_error_count, 0)
        self.assertIn("schema errors present", " ".join(report.r10_blockers))

    def test_duplicate_incident_id_is_flagged(self):
        manifest = DatasetManifest(
            version="t",
            created_at="2026-01-01T00:00:00Z",
            videos=[
                SourceVideoMetadata(
                    video_id="v1",
                    reference="v1",
                    duration_seconds=60.0,
                    source_fps=30.0,
                )
            ],
            incidents=[
                GroundTruthIncident(
                    video_id="v1",
                    incident_id="dup",
                    label=GTLabel.SUSPICIOUS,
                    start_video_time=0.0,
                    end_video_time=5.0,
                    behavior_class="provide-object",
                ),
                GroundTruthIncident(
                    video_id="v1",
                    incident_id="dup",
                    label=GTLabel.SUSPICIOUS,
                    start_video_time=10.0,
                    end_video_time=15.0,
                    behavior_class="provide-object",
                ),
            ],
        )
        gt = GroundTruthDataset(manifests=[manifest])
        report = assess_gt_dataset_readiness(gt)
        self.assertFalse(report.usable_for_r10)
        self.assertGreater(report.schema_error_count, 0)

    def test_negative_interval_is_flagged(self):
        manifest = DatasetManifest(
            version="t",
            created_at="2026-01-01T00:00:00Z",
            videos=[
                SourceVideoMetadata(
                    video_id="v1",
                    reference="v1",
                    duration_seconds=60.0,
                    source_fps=30.0,
                )
            ],
            incidents=[
                GroundTruthIncident(
                    video_id="v1",
                    incident_id="i1",
                    label=GTLabel.SUSPICIOUS,
                    start_video_time=-1.0,
                    end_video_time=5.0,
                    behavior_class="provide-object",
                )
            ],
        )
        gt = GroundTruthDataset(manifests=[manifest])
        report = assess_gt_dataset_readiness(gt)
        self.assertFalse(report.usable_for_r10)
        self.assertGreater(report.schema_error_count, 0)

    def test_incident_referring_missing_video_is_flagged(self):
        manifest = DatasetManifest(
            version="t",
            created_at="2026-01-01T00:00:00Z",
            videos=[
                SourceVideoMetadata(
                    video_id="v1",
                    reference="v1",
                    duration_seconds=60.0,
                    source_fps=30.0,
                )
            ],
            incidents=[
                GroundTruthIncident(
                    video_id="v_missing",
                    incident_id="i1",
                    label=GTLabel.SUSPICIOUS,
                    start_video_time=0.0,
                    end_video_time=5.0,
                    behavior_class="provide-object",
                )
            ],
        )
        gt = GroundTruthDataset(manifests=[manifest])
        report = assess_gt_dataset_readiness(gt)
        self.assertFalse(report.usable_for_r10)
        self.assertGreater(report.schema_error_count, 0)


# ---------------------------------------------------------------------------
# validate_for_r10 — strict gate
# ---------------------------------------------------------------------------


class TestR11ValidateForR10(unittest.TestCase):

    def test_valid_dataset_passes(self):
        gt = _mk_gt_dataset()
        result = validate_for_r10(gt)
        self.assertTrue(result["ok"])
        self.assertEqual(result["blockers"], [])

    def test_empty_dataset_fails(self):
        gt = GroundTruthDataset()
        result = validate_for_r10(gt)
        self.assertFalse(result["ok"])
        self.assertGreater(len(result["blockers"]), 0)

    def test_schema_errors_fail_even_with_counts_present(self):
        """A dataset with valid counts but schema errors must NOT pass."""
        manifest = DatasetManifest(
            version="t",
            created_at="2026-01-01T00:00:00Z",
            videos=[
                SourceVideoMetadata(
                    video_id=f"v{i}",
                    reference=f"r{i}",
                    duration_seconds=60.0,
                    source_fps=30.0,
                )
                for i in range(3)
            ],
            incidents=[
                GroundTruthIncident(
                    video_id="v_missing",
                    incident_id="i0",
                    label=GTLabel.SUSPICIOUS,
                    start_video_time=0.0,
                    end_video_time=5.0,
                    behavior_class="provide-object",
                )
            ],
        )
        gt = GroundTruthDataset(manifests=[manifest])
        result = validate_for_r10(gt)
        self.assertFalse(
            result["ok"],
            "validate_for_r10 must be stricter than counts-only",
        )
        self.assertIn("schema errors present", " ".join(result["blockers"]))

    def test_result_is_json_serializable(self):
        gt = _mk_gt_dataset()
        serialized = json.dumps(validate_for_r10(gt))
        self.assertIsInstance(serialized, str)


# ---------------------------------------------------------------------------
# load_gt_dataset
# ---------------------------------------------------------------------------


class TestR11LoadGtDataset(unittest.TestCase):

    def test_load_valid_json_string(self):
        manifest = _mk_manifest_valid()
        json_str = manifest.to_json()
        gt, issues = load_gt_dataset(json_str, source_name="inline_json")
        self.assertIsNotNone(gt)
        self.assertEqual(issues, [])

    def test_load_invalid_json_returns_none(self):
        gt, issues = load_gt_dataset('{"bad json {{{', source_name="bad")
        self.assertIsNone(gt)
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("Invalid JSON" in i or "invalid JSON" in i.lower() for i in issues))

    def test_load_csv_string_fails(self):
        gt, issues = load_gt_dataset("col1,col2\nval1,val2", source_name="csv_file")
        self.assertIsNone(gt)
        self.assertGreater(len(issues), 0)

    def test_load_file_not_found(self):
        gt, issues = load_gt_dataset("/nonexistent/path.json", source_name="missing_file")
        self.assertIsNone(gt)
        self.assertGreater(len(issues), 0)
        self.assertTrue(
            any("not found" in i.lower() or "FileNotFoundError" in i for i in issues)
        )

    def test_load_valid_file(self):
        manifest = _mk_manifest_valid()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(manifest.to_dict(), fh)
            tmp_path = fh.name
        try:
            gt, issues = load_gt_dataset(tmp_path, source_name="tmp_file")
            self.assertIsNotNone(gt)
            self.assertEqual(issues, [])
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# No fabrication: empty dataset, no file I/O
# ---------------------------------------------------------------------------


class TestR11NoFabrication(unittest.TestCase):

    def test_empty_dataset_produces_no_synthetic_values(self):
        gt = GroundTruthDataset()
        report = assess_gt_dataset_readiness(gt)
        self.assertFalse(report.usable_for_r10)
        self.assertEqual(report.video_count, 0)
        self.assertEqual(report.incident_count, 0)
        self.assertFalse(hasattr(report, "operating_point"))
        self.assertFalse(hasattr(report, "operating_point_result"))
        self.assertFalse(hasattr(report, "recall"))
        self.assertFalse(hasattr(report, "precision"))

    def test_assess_has_no_file_io(self):
        gt = _mk_gt_dataset()
        with patch("builtins.open", side_effect=OSError("unexpected file open")):
            report = assess_gt_dataset_readiness(gt)
        self.assertIsInstance(report, GTDatasetReadinessReport)

    def test_validate_for_r10_has_no_file_io(self):
        gt = _mk_gt_dataset()
        with patch("builtins.open", side_effect=OSError("unexpected file open")):
            result = validate_for_r10(gt)
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# R10 unblock criteria in report
# ---------------------------------------------------------------------------


class TestR11UnblockCriteria(unittest.TestCase):

    def test_unblock_list_present(self):
        gt = _mk_gt_dataset()
        report = assess_gt_dataset_readiness(gt)
        self.assertIsInstance(report.unblock_requirements, list)
        self.assertGreater(len(report.unblock_requirements), 0)

    def test_unblock_mentions_gt_dataset(self):
        gt = _mk_gt_dataset()
        report = assess_gt_dataset_readiness(gt)
        joined = " ".join(report.unblock_requirements).lower()
        self.assertIn("groundtruthdataset", joined)

    def test_blocked_report_lists_remaining_blockers(self):
        gt = GroundTruthDataset()
        report = assess_gt_dataset_readiness(gt)
        self.assertGreater(len(report.r10_blockers), 0)


if __name__ == "__main__":
    unittest.main()
