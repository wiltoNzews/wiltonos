# WiltonOS Sensor Modules
# Real-time coherence inputs from body signals

from .keystroke_rhythm import KeystrokeRhythm, KeystrokeMetrics

# Breath mic requires PortAudio - optional import
try:
    from .breath_mic import BreathMic, BreathMetrics, BreathPhase
    BREATH_MIC_AVAILABLE = True
except (ImportError, OSError) as e:
    BreathMic = None
    BreathMetrics = None
    BreathPhase = None
    BREATH_MIC_AVAILABLE = False

# Breath camera uses OpenCV - optional import
try:
    from .breath_camera import BreathCamera, BreathCameraMetrics
    from .breath_camera import BreathPhase as CameraBreathPhase
    BREATH_CAMERA_AVAILABLE = True
except (ImportError, OSError) as e:
    BreathCamera = None
    BreathCameraMetrics = None
    BREATH_CAMERA_AVAILABLE = False

# Coherence hub requires breath_mic
try:
    from .coherence_hub import CoherenceHub, CoherenceState, BreathMode
    COHERENCE_HUB_AVAILABLE = True
except (ImportError, OSError) as e:
    CoherenceHub = None
    CoherenceState = None
    BreathMode = None
    COHERENCE_HUB_AVAILABLE = False

# SharedBreathField from parent core module (re-export for convenience)
try:
    from ..shared_breath import SharedBreathField, AlignmentState, ResponseGuidance
    SHARED_BREATH_AVAILABLE = True
except (ImportError, OSError) as e:
    SharedBreathField = None
    AlignmentState = None
    ResponseGuidance = None
    SHARED_BREATH_AVAILABLE = False

__all__ = [
    'KeystrokeRhythm', 'KeystrokeMetrics',
    'BreathMic', 'BreathMetrics', 'BreathPhase',
    'BreathCamera', 'BreathCameraMetrics',
    'CoherenceHub', 'CoherenceState', 'BreathMode',
    'SharedBreathField', 'AlignmentState', 'ResponseGuidance',
    'BREATH_MIC_AVAILABLE', 'BREATH_CAMERA_AVAILABLE',
    'COHERENCE_HUB_AVAILABLE', 'SHARED_BREATH_AVAILABLE'
]
