from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class SLRN_TTS:
    """Simple TTS/vocoder neuron stub.

    Methods:
    - set_voice: choose a pseudo-voice id
    - synthesize: generate a fake waveform from token ids
    - save_wav: write the waveform to a .wav-like bytes buffer
    """

    voice_id: int = 0
    sample_rate: int = 16000

    def set_voice(self, voice_id: int) -> None:
        if voice_id < 0:
            raise ValueError("voice_id must be >= 0")
        self.voice_id = voice_id

    def synthesize(self, token_ids: Iterable[int], duration_sec: float = 1.0) -> List[int]:
        length = max(1, int(self.sample_rate * max(0.05, duration_sec)))
        waveform: List[int] = []
        base = (self.voice_id * 123) % 251
        ids = list(token_ids)
        for i in range(length):
            t = ids[i % max(1, len(ids))] if ids else base
            # bounded int16-like value
            sample = ((t + base + i) * 73) % 65536 - 32768
            waveform.append(int(sample))
        return waveform

    def save_wav(self, samples: List[int], path: Optional[str] = None) -> bytes:
        # not a real wav header; just a simple bytes format for demo/testing
        header = b"WV" + int(self.sample_rate).to_bytes(4, "little", signed=False)
        body = b"".join(int(s & 0xFFFF).to_bytes(2, "little", signed=False) for s in samples)
        data = header + body
        return data

    # Optional adapters
    def mel_spectrogram(self, samples: List[int]) -> Optional[object]:
        try:
            import numpy as np  # type: ignore
            import librosa  # type: ignore
        except Exception:
            return None
        y = np.array(samples, dtype=np.float32) / 32768.0
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=80)
        return mel

    def try_vits_coqui(self, text: str) -> Optional[List[int]]:
        try:
            import TTS  # type: ignore  # Coqui TTS package root
        except Exception:
            return None
        # Placeholder: assume external pipeline handles the audio
        # Return None to indicate not executed within this environment
        return None

    def try_hifigan_vocoder(self, mel: object) -> Optional[List[int]]:
        try:
            import torch  # type: ignore
        except Exception:
            return None
        # Placeholder without model weights
        return None
