"""Phase 2C.4 calibration tests.

Covers:
- CalibrationParameterGrid (validation + determinism)
- CalibrationMetrics (to_dict/from_dict)
- CalibrationSplit / split_videos (determinism, no leakage)
- compute_calibration_metrics (edge cases: None denominators, p95 threshold)
- OperatingPointPolicy / select_operating_point (precision-first, Pareto)
- sweep_matching_parameters (structure)
- sweep_confirmation_parameters (confidence gate)
- compute_subgroup_metrics (per-tag aggregation)
- build_calibration_report (JSON round-trip, provenance)
"""
from __future__ import annotations

import json
import statistics
import pytest

from inference.evaluation.calibration import (
    CALIBRATION_VERSION,
    CandidateEvaluationConfig,
    CalibrationMetrics,
    CalibrationParameterGrid,
    CalibrationReport,
    CalibrationSplit,
    OperatingPointPolicy,
    OperatingPointResult,
    build_calibration_report,
    compute_calibration_metrics,
    compute_subgroup_metrics,
    select_operating_point,
    split_videos,
    sweep_confirmation_parameters,
    sweep_matching_parameters,
)
from inference.evaluation.matching import (
    ClassMatchStatus,
    EvaluationOutcome,
    IdentityMatchStatus,
    MatchResult,
    MatchingPolicy,
    NormalizedPrediction,
    PredictionBoundary,
    TemporalMatchStatus,
)
from inference.evaluation.runner import PredictionArtifact, PredictionRecord
from inference.evaluation.ground_truth import (
    AmbiguousPeriod,
    AnnotationProvenance,
    DatasetManifest,
    GroundTruthDataset,
    GroundTruthIncident,
    GTLabel,
    NormalPeriod,
    SceneTag,
    SourceVideoMetadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mk_video(
    video_id: str = "vid_1",
    scene_tags: tuple = (),
    duration: float = 3600.0,
) -> SourceVideoMetadata:
    return SourceVideoMetadata(
        video_id=video_id,
        reference=f"ref_{video_id}",
        duration_seconds=duration,
        scene_tags=tuple(SceneTag(t) for t in scene_tags),
    )


def _mk_incident(
    incident_id: str,
    video_id: str = "vid_1",
    start: float = 10.0,
    end: float = 20.0,
    label: GTLabel = GTLabel.SUSPICIOUS,
    behavior_class: str = "provide-object",
) -> GroundTruthIncident:
    return GroundTruthIncident(
        incident_id=incident_id,
        video_id=video_id,
        label=label,
        start_video_time=start,
        end_video_time=end,
        behavior_class=behavior_class,
    )


def _mk_manifest(
    version: str = "v1",
    videos: list = None,
    incidents: list = None,
    normal_periods: list = None,
    ambiguous_periods: list = None,
) -> DatasetManifest:
    return DatasetManifest(
        version=version,
        created_at="2025-01-01T00:00:00Z",
        videos=videos or [],
        incidents=incidents or [],
        normal_periods=normal_periods or [],
        ambiguous_periods=ambiguous_periods or [],
    )


def _mk_dataset(manifests: list = None) -> GroundTruthDataset:
    return GroundTruthDataset(manifests=manifests or [])


def _mk_prediction(
    pred_id: str,
    video_id: str = "vid_1",
    start: float = 10.0,
    end: float = 20.0,
    confidence: float = None,
    confirmation_time: float = None,
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=pred_id,
        video_id=video_id,
        start_video_time=start,
        end_video_time=end,
        event_confidence=confidence,
        confirmation_video_time=confirmation_time,
        boundary_type="exact_interval",
    )


def _mk_artifact(records: list = None) -> PredictionArtifact:
    return PredictionArtifact(
        source_identifier="test_artifact",
        records=records or [],
    )


def _mk_match_result(
    pred_id: str,
    outcome: EvaluationOutcome,
    gt_id: str = None,
) -> MatchResult:
    return MatchResult(
        prediction_id=pred_id,
        ground_truth_id=gt_id,
        outcome=outcome,
        class_match=ClassMatchStatus.CLASS_UNAVAILABLE,
        identity_match=IdentityMatchStatus.IDENTITY_UNAVAILABLE,
        temporal_status=TemporalMatchStatus.UNMATCHED,
        overlap_duration=0.0,
        overlap_ratio_prediction=None,
        overlap_ratio_ground_truth=None,
        reason="test",
    )


# ---------------------------------------------------------------------------
# CalibrationParameterGrid
# ---------------------------------------------------------------------------

class TestCalibrationParameterGrid:

    def test_empty_parameters_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CalibrationParameterGrid(parameters={})

    def test_empty_values_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            CalibrationParameterGrid(parameters={"min_overlap_ratio": []})

    def test_non_list_values_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            CalibrationParameterGrid(parameters={"x": "not_a_list"})

    def test_combinations_deterministic(self):
        grid = CalibrationParameterGrid(
            parameters={
                "min_overlap_ratio": [0.5, 0.7],
                "overlap_metric": ["gt_ratio", "pred_ratio"],
            }
        )
        combos = grid.combinations()
        assert len(combos) == 4
        # Sorted lexicographically over sorted param names
        names = sorted(grid.parameters.keys())
        assert combos[0] == dict(zip(names, [0.5, "gt_ratio"]))
        # Re-running gives same result
        assert grid.combinations() == combos

    def test_combinations_single_param(self):
        grid = CalibrationParameterGrid(
            parameters={"min_confidence_mean": [0.5, 0.8]}
        )
        combos = grid.combinations()
        assert len(combos) == 2
        assert combos[0]["min_confidence_mean"] == 0.5
        assert combos[1]["min_confidence_mean"] == 0.8


# ---------------------------------------------------------------------------
# CalibrationMetrics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:

    def _empty(self) -> CalibrationMetrics:
        return CalibrationMetrics(
            tp=0, fp=0, fn=0, duplicates=0, ambiguous=0,
            precision=None, recall=None, f1=None,
            duplicate_rate=None, fp_per_hour=None, fn_per_hour=None,
            latency_count=0, latency_mean=None, latency_median=None,
            latency_p95=None, total_videos=0,
            total_gt_incidents=0, total_predictions=0,
        )

    def test_to_dict_round_trip(self):
        m = CalibrationMetrics(
            tp=5, fp=2, fn=3, duplicates=1, ambiguous=0,
            precision=5/7, recall=5/8, f1=2*(5/7)*(5/8)/(5/7+5/8),
            duplicate_rate=1/8,
            fp_per_hour=2/3.0, fn_per_hour=3/3.0,
            latency_count=10, latency_mean=5.0, latency_median=4.0,
            latency_p95=8.0, total_videos=3,
            total_gt_incidents=10, total_predictions=8,
        )
        d = m.to_dict()
        m2 = CalibrationMetrics.from_dict(d)
        assert m2.tp == 5
        assert m2.precision == m.precision
        assert m2.latency_p95 == 8.0

    def test_default_empty(self):
        m = self._empty()
        d = m.to_dict()
        assert d["precision"] is None
        assert d["latency_p95"] is None
        m2 = CalibrationMetrics.from_dict(d)
        assert m2.tp == 0


# ---------------------------------------------------------------------------
# split_videos
# ---------------------------------------------------------------------------

class TestSplitVideos:

    def test_deterministic_same_seed(self):
        vids = [f"vid_{i}" for i in range(10)]
        s1 = split_videos(vids, seed=42)
        s2 = split_videos(vids, seed=42)
        assert s1 == s2

    def test_different_seed_different_split(self):
        vids = [f"vid_{i}" for i in range(50)]
        s1 = split_videos(vids, seed=42)
        s2 = split_videos(vids, seed=99)
        # Both should partition the same set
        assert set(s1["train"] + s1["validation"] + s1["holdout"]) == set(vids)
        assert set(s2["train"] + s2["validation"] + s2["holdout"]) == set(vids)
        # Different seeds should (almost certainly) produce different splits
        assert s1 != s2

    def test_no_leakage_disjoint(self):
        vids = [f"vid_{i}" for i in range(20)]
        s = split_videos(vids, seed=42)
        train_set = set(s["train"])
        val_set = set(s["validation"])
        hold_set = set(s["holdout"])
        assert not (train_set & val_set)
        assert not (train_set & hold_set)
        assert not (val_set & hold_set)

    def test_all_videos_included(self):
        vids = [f"vid_{i}" for i in range(10)]
        s = split_videos(vids, seed=42)
        assert set(s["train"] + s["validation"] + s["holdout"]) == set(vids)

    def test_fewer_than_3_videos(self):
        s = split_videos(["vid_a", "vid_b"], seed=42)
        assert s["validation"] == []
        assert s["holdout"] == []
        assert set(s["train"]) == {"vid_a", "vid_b"}

    def test_single_video(self):
        s = split_videos(["vid_a"], seed=42)
        assert s["train"] == ["vid_a"]
        assert s["validation"] == []
        assert s["holdout"] == []

    def test_empty_list(self):
        s = split_videos([], seed=42)
        assert s["train"] == []
        assert s["validation"] == []
        assert s["holdout"] == []


# ---------------------------------------------------------------------------
# compute_calibration_metrics
# ---------------------------------------------------------------------------

class TestComputeCalibrationMetrics:

    def _results(self, outcomes: list) -> list:
        return [_mk_match_result(f"p{i}", o) for i, o in enumerate(outcomes)]

    def test_no_results(self):
        m = compute_calibration_metrics([])
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None
        assert m.duplicate_rate is None

    def test_precision_none_when_no_predictions(self):
        # Only FN results — no TP/FP, so precision denominator is 0
        results = self._results([EvaluationOutcome.FN])
        m = compute_calibration_metrics(results)
        assert m.precision is None
        assert m.recall is not None  # 0/(0+1) = 0.0

    def test_recall_none_when_no_gt(self):
        results = self._results([EvaluationOutcome.FP])
        m = compute_calibration_metrics(results)
        assert m.recall is None

    def test_f1_none_when_precision_or_recall_none(self):
        results = self._results([EvaluationOutcome.FP])
        m = compute_calibration_metrics(results)
        assert m.f1 is None  # recall is None

    def test_fp_per_hour_none_when_hours_zero(self):
        results = self._results([EvaluationOutcome.FP])
        m = compute_calibration_metrics(results, total_video_hours=0.0)
        assert m.fp_per_hour is None

    def test_fp_per_hour_when_hours_positive(self):
        results = self._results([EvaluationOutcome.FP, EvaluationOutcome.FP])
        m = compute_calibration_metrics(results, total_video_hours=2.0)
        assert m.fp_per_hour == pytest.approx(1.0)

    def test_p95_none_when_fewer_than_10(self):
        results = self._results([EvaluationOutcome.TP])
        lats = [1.0, 2.0, 3.0]
        m = compute_calibration_metrics(results, latencies=lats)
        assert m.latency_p95 is None
        assert m.latency_count == 3

    def test_p95_present_when_10_or_more(self):
        results = self._results([EvaluationOutcome.TP] * 10)
        lats = [float(i) for i in range(1, 11)]
        m = compute_calibration_metrics(results, latencies=lats)
        assert m.latency_p95 is not None
        assert m.latency_p95 == pytest.approx(10.0)

    def test_none_latencies_excluded(self):
        results = self._results([EvaluationOutcome.TP])
        lats = [1.0, None, 3.0, None]
        m = compute_calibration_metrics(results, latencies=lats)
        assert m.latency_count == 2
        assert m.latency_mean == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# OperatingPointPolicy / select_operating_point
# ---------------------------------------------------------------------------

def _row(combo, precision=None, fp_per_hour=None, recall=None, duplicate_rate=None):
    m = CalibrationMetrics(
        tp=1, fp=1, fn=1, duplicates=0, ambiguous=0,
        precision=precision, recall=recall,
        f1=(2*precision*recall/(precision+recall))
            if (precision is not None and recall is not None
                and (precision+recall) > 0) else None,
        duplicate_rate=duplicate_rate,
        fp_per_hour=fp_per_hour,
        fn_per_hour=None,
        latency_count=0, latency_mean=None, latency_median=None,
        latency_p95=None,
        total_videos=1, total_gt_incidents=1, total_predictions=1,
    )
    return {**combo, "metrics": m, "split": "train"}


class TestSelectOperatingPoint:

    def test_pareto_when_no_targets(self):
        rows = [
            _row({"min_overlap_ratio": 0.5}, precision=0.5, fp_per_hour=2.0, recall=0.9),
            _row({"min_overlap_ratio": 0.7}, precision=0.8, fp_per_hour=0.5, recall=0.7),
        ]
        result = select_operating_point(rows, OperatingPointPolicy())
        assert result.is_pareto_candidate is True
        assert len(result.pareto_set) >= 1

    def test_recall_first_with_fp_target(self):
        # Row A: high precision but FP rate too high
        # Row B: slightly lower precision but satisfies FP rate constraint and higher recall
        rows = [
            _row({"min_overlap_ratio": 0.5}, precision=0.95, fp_per_hour=10.0, recall=0.5),
            _row({"min_overlap_ratio": 0.7}, precision=0.80, fp_per_hour=1.0,  recall=0.8),
        ]
        policy = OperatingPointPolicy(fp_per_hour_target=2.0, min_recall=0.5)
        result = select_operating_point(rows, policy)
        # Row B qualifies (fp=1.0<=2.0, recall=0.8>=0.5); Row A fails FP constraint
        assert result.selected_combination["min_overlap_ratio"] == 0.7
        assert result.is_pareto_candidate is False

    def test_empty_rows_raises_value_error(self):
        with pytest.raises(ValueError, match="no valid sweep rows"):
            select_operating_point([], OperatingPointPolicy())

    def test_no_candidates_with_constraints_raises(self):
        rows = [
            _row({"min_overlap_ratio": 0.5}, precision=0.5, fp_per_hour=10.0, recall=0.5),
        ]
        policy = OperatingPointPolicy(fp_per_hour_target=1.0, min_recall=0.5)
        # fp_per_hour=10.0 > target 1.0 → excluded → no candidates → ValueError
        with pytest.raises(ValueError, match="no sweep row satisfies"):
            select_operating_point(rows, policy)

    def test_tie_break_by_fp_per_hour(self):
        rows = [
            _row({"a": 1}, precision=0.9, fp_per_hour=5.0, recall=0.8),
            _row({"a": 2}, precision=0.9, fp_per_hour=1.0, recall=0.8),
        ]
        policy = OperatingPointPolicy(fp_per_hour_target=10.0, min_recall=0.5)
        result = select_operating_point(rows, policy)
        # Both qualify (fp<=10, recall>=0.5); same recall and precision; tie-break by lower fp_per_hour
        assert result.selected_combination["a"] == 2

    def test_fp_constraint_oustrates_higher_recall(self):
        """Operating-point regression: candidate A has higher recall but FP/hour
        above the allowed limit; candidate B has lower recall but FP/hour within
        limit.  With an FP/hour constraint, B must be selected."""
        rows = [
            _row({"label": "A_high_recall_over_fp"},
                 precision=0.95, fp_per_hour=10.0, recall=0.95),
            _row({"label": "B_lower_recall_in_fp"},
                 precision=0.80, fp_per_hour=1.0,  recall=0.70),
        ]
        policy = OperatingPointPolicy(fp_per_hour_target=2.0)
        result = select_operating_point(rows, policy)
        assert result.selected_combination["label"] == "B_lower_recall_in_fp"
        assert result.metrics.recall == pytest.approx(0.70)
        assert result.metrics.fp_per_hour == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sweep_matching_parameters — structure
# ---------------------------------------------------------------------------

class TestSweepMatchingParameters:

    def _setup(self):
        videos = [_mk_video("vid_1", scene_tags=("normal_exam",), duration=3600.0)]
        incidents = [_mk_incident("inc_1", "vid_1", 10.0, 20.0)]
        manifest = _mk_manifest(videos=videos, incidents=incidents)
        dataset = _mk_dataset([manifest])
        preds = [_mk_prediction("p_1", "vid_1", 10.0, 20.0, confirmation_time=15.0)]
        artifact = _mk_artifact(preds)
        split = CalibrationSplit(split_name="all", videos=["vid_1"])
        return dataset, artifact, split

    def test_returns_one_row_per_combination(self):
        dataset, artifact, split = self._setup()
        grid = CalibrationParameterGrid(
            parameters={
                "min_overlap_ratio": [0.3, 0.5],
                "overlap_metric": ["gt_ratio"],
            }
        )
        rows = sweep_matching_parameters(dataset, artifact, grid, split)
        assert len(rows) == 2
        assert all("metrics" in r for r in rows)
        assert all(r["split"] == "all" for r in rows)

    def test_rows_are_deterministic(self):
        dataset, artifact, split = self._setup()
        grid = CalibrationParameterGrid(
            parameters={"min_overlap_ratio": [0.3, 0.5]}
        )
        r1 = sweep_matching_parameters(dataset, artifact, grid, split)
        r2 = sweep_matching_parameters(dataset, artifact, grid, split)
        # Compare by JSON (CalibrationMetrics objects may differ by identity)
        assert json.dumps([r.get("min_overlap_ratio") for r in r1]) == \
               json.dumps([r.get("min_overlap_ratio") for r in r2])

    def test_metrics_are_calibration_metrics(self):
        dataset, artifact, split = self._setup()
        grid = CalibrationParameterGrid(parameters={"min_overlap_ratio": [0.5]})
        rows = sweep_matching_parameters(dataset, artifact, grid, split)
        assert isinstance(rows[0]["metrics"], CalibrationMetrics)


# ---------------------------------------------------------------------------
# sweep_confirmation_parameters — confidence gate
# ---------------------------------------------------------------------------

class TestSweepConfirmationParameters:

    def _setup(self):
        videos = [_mk_video("vid_1", scene_tags=("normal_exam",), duration=3600.0)]
        incidents = [_mk_incident("inc_1", "vid_1", 10.0, 20.0)]
        manifest = _mk_manifest(videos=videos, incidents=incidents)
        dataset = _mk_dataset([manifest])
        preds = [
            _mk_prediction("p_1", "vid_1", 10.0, 20.0,
                           confidence=0.8, confirmation_time=15.0),
            _mk_prediction("p_2", "vid_1", 10.0, 20.0,
                           confidence=0.4, confirmation_time=18.0),
        ]
        artifact = _mk_artifact(preds)
        split = CalibrationSplit(split_name="all", videos=["vid_1"])
        return dataset, artifact, split

    def test_confidence_gate_filters_low_confidence(self):
        dataset, artifact, split = self._setup()
        grid = CalibrationParameterGrid(
            parameters={"min_confidence_mean": [0.5]}
        )
        rows = sweep_confirmation_parameters(dataset, artifact, grid, split)
        m = rows[0]["metrics"]
        # Only p_1 (confidence=0.8) passes the 0.5 gate
        assert m.total_predictions == 1

    def test_no_gate_when_min_confidence_absent(self):
        dataset, artifact, split = self._setup()
        grid = CalibrationParameterGrid(
            parameters={"other_param": [1]}
        )
        rows = sweep_confirmation_parameters(dataset, artifact, grid, split)
        m = rows[0]["metrics"]
        assert m.total_predictions == 2

    def test_none_confidence_always_excluded_with_gate(self):
        preds = [
            _mk_prediction("p_x", "vid_1", 10.0, 20.0,
                           confidence=None, confirmation_time=15.0),
        ]
        manifest = _mk_manifest(
            videos=[_mk_video("vid_1", duration=3600.0)],
            incidents=[_mk_incident("inc_1", "vid_1")],
        )
        dataset = _mk_dataset([manifest])
        artifact = _mk_artifact(preds)
        split = CalibrationSplit(split_name="all", videos=["vid_1"])
        grid = CalibrationParameterGrid(parameters={"min_confidence_mean": [0.1]})
        rows = sweep_confirmation_parameters(dataset, artifact, grid, split)
        m = rows[0]["metrics"]
        # p_x has confidence=None → excluded
        assert m.total_predictions == 0


# ---------------------------------------------------------------------------
# compute_subgroup_metrics
# ---------------------------------------------------------------------------

class TestComputeSubgroupMetrics:

    def _setup_two_videos(self):
        manifest = _mk_manifest(
            videos=[
                _mk_video("vid_1", scene_tags=("normal_exam", "multi_person")),
                _mk_video("vid_2", scene_tags=("normal_exam",)),
            ],
            incidents=[
                _mk_incident("i1", "vid_1", behavior_class="provide-object"),
                _mk_incident("i2", "vid_2", behavior_class="provide-object"),
            ],
        )
        dataset = _mk_dataset([manifest])
        preds = [
            _mk_prediction("p1", "vid_1", 10.0, 20.0, confirmation_time=15.0),
            _mk_prediction("p2", "vid_2", 10.0, 20.0, confirmation_time=15.0),
        ]
        artifact = _mk_artifact(preds)
        results = [
            _mk_match_result("p1", EvaluationOutcome.TP, gt_id="i1"),
            _mk_match_result("p2", EvaluationOutcome.TP, gt_id="i2"),
        ]
        return dataset, artifact, results

    def test_returns_only_present_tags(self):
        dataset, artifact, results = self._setup_two_videos()
        result = compute_subgroup_metrics(dataset, artifact, results)
        assert "normal_exam" in result
        assert "multi_person" in result
        assert result["normal_exam"]["video_count"] == 2
        assert result["multi_person"]["video_count"] == 1

    def test_video_with_multiple_tags_counts_in_each(self):
        dataset, artifact, results = self._setup_two_videos()
        result = compute_subgroup_metrics(dataset, artifact, results)
        assert result["normal_exam"]["tp"] == 2
        assert result["multi_person"]["tp"] == 1

    def test_no_tags_in_dataset_returns_empty(self):
        manifest = _mk_manifest(
            videos=[_mk_video("vid_1", scene_tags=())],
            incidents=[_mk_incident("i1", "vid_1", behavior_class="provide-object")],
        )
        dataset = _mk_dataset([manifest])
        artifact = _mk_artifact([])
        result = compute_subgroup_metrics(dataset, artifact, [])
        assert result == {}


# ---------------------------------------------------------------------------
# build_calibration_report
# ---------------------------------------------------------------------------

class TestBuildCalibrationReport:

    def _setup(self):
        manifest = _mk_manifest(
            version="v1",
            videos=[_mk_video("vid_1", scene_tags=("normal_exam",), duration=3600.0)],
            incidents=[_mk_incident("inc_1", "vid_1")],
        )
        dataset = _mk_dataset([manifest])
        preds = [_mk_prediction("p_1", "vid_1", 10.0, 20.0,
                                confidence=0.9, confirmation_time=15.0)]
        artifact = _mk_artifact(preds)
        split = CalibrationSplit(split_name="all", videos=["vid_1"])
        grid = CalibrationParameterGrid(
            parameters={"min_overlap_ratio": [0.5]}
        )
        matching_sweep = sweep_matching_parameters(dataset, artifact, grid, split)
        # matching_selected is a plain row dict (not OperatingPointResult)
        matching_selected = matching_sweep[0] if matching_sweep else None
        return dataset, artifact, split, grid, matching_sweep, matching_selected

    def test_report_contains_all_fields(self):
        ds, art, split, grid, mres, msel = self._setup()
        report = build_calibration_report(
            gt_dataset=ds,
            artifact=art,
            matching_sweep=mres,
            matching_selected=msel,
        )
        d = report.to_dict()
        assert d["calibration_version"] == CALIBRATION_VERSION
        assert d["dataset_version"] == "v1"
        assert d["artifact_version"] == "1.0"
        assert d["artifact_identifier"] == "test_artifact"
        assert "matching_results" in d
        assert len(d["matching_results"]) == 1

    def test_report_is_json_serializable(self):
        ds, art, split, grid, mres, msel = self._setup()
        report = build_calibration_report(
            gt_dataset=ds,
            artifact=art,
            matching_sweep=mres,
            matching_selected=msel,
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["calibration_version"] == CALIBRATION_VERSION

    def test_from_dict_round_trip(self):
        ds, art, split, grid, mres, msel = self._setup()
        report = build_calibration_report(
            gt_dataset=ds,
            artifact=art,
            matching_sweep=mres,
            matching_selected=msel,
        )
        restored = CalibrationReport.from_dict(report.to_dict())
        assert restored.calibration_version == report.calibration_version
        assert restored.dataset_version == report.dataset_version
        assert restored.artifact_identifier == report.artifact_identifier

    def test_report_with_confirmation_sweep(self):
        ds, art, split, _, _, _ = self._setup()
        conf_grid = CalibrationParameterGrid(
            parameters={"min_confidence_mean": [0.5, 0.7]}
        )
        conf_sweep = sweep_confirmation_parameters(ds, art, conf_grid, split)
        conf_selected = conf_sweep[0] if conf_sweep else None
        report = build_calibration_report(
            gt_dataset=ds,
            artifact=art,
            matching_sweep=[],
            confirmation_sweep=conf_sweep,
            confirmation_selected=conf_selected,
        )
        d = report.to_dict()
        assert d["confirmation_grid"] is not None
        assert d["confirmation_results"] is not None
        assert len(d["confirmation_results"]) == 2

    def test_timestamp_is_iso_format(self):
        ds, art, split, grid, mres, msel = self._setup()
        report = build_calibration_report(
            gt_dataset=ds,
            artifact=art,
            matching_sweep=mres,
            matching_selected=msel,
        )
        ts = report.timestamp_utc
        assert "T" in ts  # ISO-8601 separator

    def test_parameter_status_marks_unswept(self):
        ds, art, split, grid, mres, msel = self._setup()
        report = build_calibration_report(
            gt_dataset=ds,
            artifact=art,
            matching_sweep=mres,
            matching_selected=msel,
            # No confirmation_sweep → confirmation params marked not_calibrated
        )
        d = report.to_dict()
        # matching.min_overlap_ratio should be "calibrated"
        assert d["parameter_status"].get("matching.min_overlap_ratio") == "calibrated"
        # confirmation.min_confidence_mean should be "not_calibrated"
        assert d["parameter_status"].get("confirmation.min_confidence_mean") == "not_calibrated"
