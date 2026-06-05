from __future__ import annotations

import asyncio
import io
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from orion_tts.backends.base import TTSBackend


class EdgeTTSBackend(TTSBackend):
    def __init__(self, config: dict) -> None:
        self._default_voice: str = config.get('voice', 'es-VE-SebastianNeural')
        self._target_sr: int = config.get('sample_rate', 48000)

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self._target_sr:
            return audio
        g = gcd(orig_sr, self._target_sr)
        return resample_poly(audio, self._target_sr // g, orig_sr // g).astype(np.float32)

    async def _stream_mp3(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                chunks.extend(chunk['data'])
        return bytes(chunks)

    def _decode(self, mp3: bytes) -> tuple[np.ndarray, int]:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(io.BytesIO(mp3), format='mp3')
        seg = seg.set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        # int16/int32 PCM -> [-1.0, 1.0]
        max_val = float(1 << (8 * seg.sample_width - 1))
        samples /= max_val
        return self._resample(samples, seg.frame_rate), self._target_sr

    async def synthesize(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        voice = voice or self._default_voice
        mp3 = await self._stream_mp3(text, voice)
        if not mp3:
            return np.zeros(0, dtype=np.float32), self._target_sr
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._decode, mp3)

    def is_available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            import pydub  # noqa: F401
            return True
        except ImportError:
            return False
