"""Phase 2E-R8 — Shadow-to-Calibration bridge.

Converts ShadowEventRecord objects to PredictionRecord / PredictionArtifact
objects for use in Phase 2C.4 calibration, and produces a
CalibrationReadinessReport that honestly documents which parameters
are sweepable today and which are blocked.

This module is read-only: no file I/O, no mutation of input records,
no fabricated confidence scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference.core.shadow_lifecycle import ShadowEventRecord
from inference.evaluation.runner import PredictionRecord, PredictionArtifact


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def shadow_events_to_prediction_records(
    shadow_events: List[ShadowEventRecord],
    video_id: str,
) -> List[PredictionRecord]:
    """Convert a list of ShadowEventRecord to PredictionRecord.

    Each ShadowEventRecord is converted via its EventRecord.to_dict() using
    the existing PredictionRecord.from_event_record_dict static method.
    The caller supplies video_id (not embedded in EventRecord).

    - event_confidence passes through verbatim (None stays None)
    - association_score is NOT a PredictionRecord field; when None it is
      recorded as None in source_metadata (never fabricated)
    - No I/O, no mutation of input records
    """
    records: List[PredictionRecord] = []
    for sec in shadow_events:
        er_dict = sec.event.to_dict()
        pr = PredictionRecord.from_event_record_dict(er_dict, video_id=video_id)
        sm: Dict[str, Any] = dict(pr.source_metadata or {})
        sm["association_score"] = sec.event.association_score
        pr.source_metadata = sm
        records.append(pr)
    return records


def shadow_events_to_prediction_artifact(
    shadow_events: List[ShadowEventRecord],
    video_id: str,
    source_identifier: str = "shadow_bridge",
) -> PredictionArtifact:
    """Convert ShadowEventRecords to a PredictionArtifact.

    Empty input produces an artifact with an empty records list.
    """
    records = shadow_events_to_prediction_records(shadow_events, video_id)
    return PredictionArtifact(
        source_identifier=source_identifier,
        records=records,
    )


# ---------------------------------------------------------------------------
# CalibrationReadinessReport
# ---------------------------------------------------------------------------

_MATCHING_PARAMS = [
    "min_overlap_ratio",
    "overlap_metric",
    "require_class_match",
    "identity_mode",
    "ambiguous_handling",
]


@dataclass
class CalibrationReadinessReport:
    """Honest report on which calibration parameters are sweepable.

    - matching_parameters_sweepable: always True (no event_confidence needed)
    - sweepable_parameters: parameters that CAN be swept with the given records
    - blocked_parameters: parameters that CANNOT be swept
    - blocked_reasons: explanation for each blocked parameter
    - association_evidence_status: "available" | "unavailable"
    - total_predictions: number of PredictionRecord in the input
    """
    total_predictions: int
    matching_parameters_sweepable: bool
    sweepable_parameters: List[str]
    blocked_parameters: List[str]
    blocked_reasons: Dict[str, str]
    association_evidence_status: str

    @classmethod
    def from_prediction_records(
        cls,
        records: List[PredictionRecord],
    ) -> "CalibrationReadinessReport":
        """Build a readiness report from a list of PredictionRecord.

        Matching parameters never require event_confidence and are always
        sweepable.  min_confidence_mean is sweepable only when at least one
        record has a non-None event_confidence; when ALL records have
        event_confidence=None it is blocked.
        """
        total = len(records)
        any_conf = any(r.event_confidence is not None for r in records)

        sweepable: List[str] = list(_MATCHING_PARAMS)
        blocked: List[str] = []
        reasons: Dict[str, str] = {}

        if any_conf:
            sweepable.append("min_confidence_mean")
        else:
            blocked.append("min_confidence_mean")
            reasons["min_confidence_mean"] = (
                "event_confidence is None on all shadow records; "
                "a calibrated event-confidence scoring policy is required"
            )

        has_assoc = any(
            (r.source_metadata or {}).get("association_score") is not None
            for r in records
        )
        assoc_status = "available" if has_assoc else "unavailable"

        return cls(
            total_predictions=total,
            matching_parameters_sweepable=True,
            sweepable_parameters=sorted(sweepable),
            blocked_parameters=blocked,
            blocked_reasons=reasons,
            association_evidence_status=assoc_status,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "matching_parameters_sweepable": self.matching_parameters_sweepable,
            "sweepable_parameters": self.sweepable_parameters,
            "blocked_parameters": self.blocked_parameters,
            "blocked_reasons": self.blocked_reasons,
            "association_evidence_status": self.association_evidence_status,
        }


__all__ = [
    "shadow_events_to_prediction_records",
    "shadow_events_to_prediction_artifact",
    "CalibrationReadinessReport",
]
