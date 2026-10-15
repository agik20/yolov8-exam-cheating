"""Core package: detection pipeline, NMS algorithm, Phase 2B.1 event
structures, Phase 2B.2 shadow candidate observation layer, and
Phase 2B.3 shadow confirmation decision layer."""
from .pipeline import CheatingDetectionPipeline
from .events import (
    ModelSource,
    EntityKey,
    Observation,
    BoundedObservationBuffer,
    CandidateState,
    ConfidenceStats,
    Candidate,
    EventRecord,
    EvidenceBundle,
)
from .candidate_layer import (
    InvalidSourceFps,
    VideoTimeSource,
    CandidateKey,
    CandidateLayer,
)
from .confirmation import (
    ConfirmationGateResult,
    ConfirmationPolicy,
    UnconfiguredConfirmationPolicy,
    CalibratedConfirmationPolicy,
)
from .cooldown import (
    CooldownGateResult,
    CooldownPolicy,
    UnconfiguredCooldownPolicy,
    CalibratedCooldownPolicy,
)
from .shadow_lifecycle import (
    ShadowEventRecord,
    ShadowComparisonRecord,
    ShadowLifecycleManager,
)
