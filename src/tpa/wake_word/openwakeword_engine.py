"""OpenWakeWordEngine — implements WakeWordPort via `openwakeword`.

★ New. Open-source, no-API-key replacement for PorcupineWakeWord — see
docs/voice/open-source-wake-speaker-design.md §4.1 for the full design.
openwakeword is frame-driven like Porcupine (not windowed like
Resemblyzer), so this adapter buffers raw frames up to the engine's
preferred chunk size (1280 samples = 80ms @ 16kHz, openwakeword's own
recommendation) and calls `Model.predict()` once per full chunk,
carrying any remainder over to the next `process()` call — same slicing
idiom PorcupineWakeWord's caller already uses in
service/voice/voice_session.py, just performed inside the adapter here
since openwakeword's native chunk size doesn't match frame_length's
reported value the way Porcupine's does.

The configured wake phrase ("Hey Veda") is a custom-trained model, not
one of openwakeword's bundled pretrained words — model_path is required
in practice (see config_schemas.py's AudioConfig.wake_oww_model_path).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# The two feature extractors every openwakeword model runs through. The pip package does not ship them.
FEATURE_MODEL_FILES = ("melspectrogram.onnx", "embedding_model.onnx")


def require_feature_models(package_dir: Path | None = None) -> None:
    """Fail with the fix, not an onnxruntime NO_SUCHFILE trace, when openwakeword's helper models are missing."""
    if package_dir is None:
        spec = importlib.util.find_spec("openwakeword")   # does not import the package
        if spec is None or not spec.origin:
            return   # not installed: the import below reports that
        package_dir = Path(spec.origin).parent
    models = package_dir / "resources" / "models"
    missing = [name for name in FEATURE_MODEL_FILES if not (models / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"openwakeword's helper model(s) {', '.join(missing)} are missing from {models}. "
            "Run scripts/setup_pi.sh (it downloads them), or: "
            'python -c "import openwakeword; openwakeword.utils.download_models([\'none\'])"'
        )


class OpenWakeWordEngine:
    """Implements WakeWordPort via openwakeword."""

    def __init__(
        self,
        model_name: str = "hey_veda",
        model_path: str | None = None,
        threshold: float = 0.5,
        chunk_samples: int = 1280,
        sample_rate: int = 16000,
    ):
        if model_path:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"openwakeword model not found at {path}")
        require_feature_models()
        from openwakeword.model import Model

        # None -> openwakeword's own bundled pretrained set; only used if a
        # pretrained word is ever selected instead of the custom "Hey Veda" model.
        paths = [model_path] if model_path else None
        self._model = Model(wakeword_models=paths, inference_framework="onnx")
        self._model_name = model_name
        self._threshold = threshold
        self._chunk_samples = chunk_samples
        self._sample_rate = sample_rate
        self._buf = bytearray()
        self.last_score = 0.0

    @property
    def frame_length(self) -> int:
        return self._chunk_samples

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def process(self, frame: bytes) -> bool:
        import numpy as np

        self._buf += frame
        chunk_bytes = self._chunk_samples * 2  # int16 PCM = 2 bytes/sample
        triggered = False
        while len(self._buf) >= chunk_bytes:
            chunk, self._buf = bytes(self._buf[:chunk_bytes]), self._buf[chunk_bytes:]
            pcm = np.frombuffer(chunk, dtype=np.int16)
            scores = self._model.predict(pcm)
            self.last_score = float(scores.get(self._model_name, 0.0))
            if self.last_score >= self._threshold:
                triggered = True
        if triggered:
            # Reset the model's prediction buffer and any partial chunk. Without this the same detection
            # keeps re-firing on the following frames (observed as repeated wake acknowledgements in silence).
            self._buf.clear()
            reset = getattr(self._model, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass
        return triggered

    def start(self) -> None:
        # Frame-driven; no background thread. Present so the voice loop can
        # drive every wake engine through the same start/stop pair.
        pass

    def stop(self) -> None:
        # Model has no explicit teardown.
        pass
