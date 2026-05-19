from __future__ import annotations

import asyncio
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from orion_stt.backends.base import STTBackend


class WhisperBackend(STTBackend):
    def __init__(self, config: dict) -> None:
        self._model_size: str = config.get('model', 'small')
        self._device: str = config.get('device', 'cpu')
        self._compute_type: str = 'float32' if self._device == 'cpu' else 'float16'
        self._no_speech_threshold: float = config.get('no_speech_threshold', 0.6)
        self._log_prob_threshold: float = config.get('log_prob_threshold', -1.0)
        self._beam_size: int = config.get('beam_size', 5)
        self._model = None  # lazy-loaded on first transcription

    def _load_model(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        self._load_model()
        audio = self._resample(audio, orig_sr=sample_rate)
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=self._no_speech_threshold,
            log_prob_threshold=self._log_prob_threshold,
        )
        texts = [
            s.text.strip()
            for s in segments
            if s.no_speech_prob < self._no_speech_threshold
        ]
        return ' '.join(texts).strip()

    async def transcribe(
        self, audio: np.ndarray, sample_rate: int, language: str
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio, sample_rate, language)

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False
