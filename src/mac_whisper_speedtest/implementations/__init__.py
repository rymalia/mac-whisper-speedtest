"""Implementation registry for Whisper implementations."""

import importlib.util
import logging
from typing import List, Type

from mac_whisper_speedtest.implementations.base import WhisperImplementation
from mac_whisper_speedtest.implementations.faster import FasterWhisperImplementation
from mac_whisper_speedtest.implementations.mlx import MLXWhisperImplementation

logger = logging.getLogger(__name__)

# Conditionally import implementations based on available packages
available_implementations = [
    FasterWhisperImplementation,
    MLXWhisperImplementation,
]

# Try to import WhisperCppCoreMLImplementation
try:
    if importlib.util.find_spec("pywhispercpp"):
        from mac_whisper_speedtest.implementations.coreml import WhisperCppCoreMLImplementation
        available_implementations.append(WhisperCppCoreMLImplementation)
    else:
        logger.warning("pywhispercpp not found, WhisperCppCoreMLImplementation will not be available")
except ImportError:
    logger.warning("Failed to import WhisperCppCoreMLImplementation")

# Try to import InsanelyFastWhisperImplementation
try:
    if importlib.util.find_spec("insanely_fast_whisper"):
        from mac_whisper_speedtest.implementations.insanely import InsanelyFastWhisperImplementation
        available_implementations.append(InsanelyFastWhisperImplementation)
    else:
        logger.warning("insanely-fast-whisper not found, InsanelyFastWhisperImplementation will not be available")
except ImportError:
    logger.warning("Failed to import InsanelyFastWhisperImplementation")

# Try to import LightningWhisperMLXImplementation
try:
    if importlib.util.find_spec("lightning_whisper_mlx"):
        from mac_whisper_speedtest.implementations.lightning import LightningWhisperMLXImplementation
        available_implementations.append(LightningWhisperMLXImplementation)
    else:
        logger.warning("lightning-whisper-mlx not found, LightningWhisperMLXImplementation will not be available")
except ImportError:
    logger.warning("Failed to import LightningWhisperMLXImplementation")

# Try to import ParakeetMLXImplementation
try:
    if importlib.util.find_spec("parakeet_mlx"):
        from mac_whisper_speedtest.implementations.parakeet_mlx import ParakeetMLXImplementation
        available_implementations.append(ParakeetMLXImplementation)
        logger.info("ParakeetMLXImplementation loaded successfully")
    else:
        logger.warning("parakeet-mlx not found, ParakeetMLXImplementation will not be available")
except ImportError as e:
    logger.warning(f"Failed to import ParakeetMLXImplementation: {e}")

# Try to import FluidAudioCoreMLImplementation
try:
    if importlib.util.find_spec("coremltools"):
        from mac_whisper_speedtest.implementations.fluidaudio_coreml import FluidAudioCoreMLImplementation
        available_implementations.append(FluidAudioCoreMLImplementation)
        logger.info("FluidAudioCoreMLImplementation loaded successfully")
    else:
        logger.warning("coremltools not found, FluidAudioCoreMLImplementation will not be available")
except ImportError as e:
    logger.warning(f"Failed to import FluidAudioCoreMLImplementation: {e}")

# Try to import WhisperKitImplementation
try:
    import platform
    if platform.system() == "Darwin":  # macOS only
        from mac_whisper_speedtest.implementations.whisperkit import WhisperKitImplementation
        available_implementations.append(WhisperKitImplementation)
        logger.info("WhisperKitImplementation loaded successfully")
    else:
        logger.warning("WhisperKit is only supported on macOS, WhisperKitImplementation will not be available")
except ImportError as e:
    logger.warning(f"Failed to import WhisperKitImplementation: {e}")
except RuntimeError as e:
    logger.warning(f"WhisperKitImplementation not available: {e}")

# Try to import WhisperMPSImplementation
try:
    if importlib.util.find_spec("whisper_mps"):
        import platform
        if platform.system() == "Darwin":  # macOS only
            from mac_whisper_speedtest.implementations.whisper_mps import WhisperMPSImplementation
            available_implementations.append(WhisperMPSImplementation)
            logger.info("WhisperMPSImplementation loaded successfully")
        else:
            logger.warning("whisper-mps is only supported on macOS, WhisperMPSImplementation will not be available")
    else:
        logger.warning("whisper-mps not found, WhisperMPSImplementation will not be available")
except ImportError as e:
    logger.warning(f"Failed to import WhisperMPSImplementation: {e}")
except RuntimeError as e:
    logger.warning(f"WhisperMPSImplementation not available: {e}")



def get_all_implementations() -> List[Type[WhisperImplementation]]:
    """Get all available Whisper implementations.

    Returns:
        List of WhisperImplementation classes
    """
    return available_implementations


def get_implementation_by_name(name: str) -> Type[WhisperImplementation]:
    """Get a Whisper implementation by name.

    Args:
        name: The name of the implementation

    Returns:
        The WhisperImplementation class

    Raises:
        ValueError: If the implementation is not found
    """
    for impl in get_all_implementations():
        if impl.__name__ == name:
            return impl

    raise ValueError(f"Implementation not found: {name}")
