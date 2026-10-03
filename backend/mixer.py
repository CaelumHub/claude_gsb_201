"""
mixer.py — Multi-track mixdown.

Mixes any number of tracks (each an audio file with a gain, pan and mute flag)
into a single stereo WAV.  Tracks may have different sample rates and lengths;
the mixer resamples on the fly with a seamless streaming resampler and pads
shorter tracks with silence.  Memory stays bounded because every track is read
a fixed-size chunk at a time.

Channel handling
----------------
* Every *channel* of a track gets its own ``StreamingResampler``.  The
  resampler carries a fractional source position and buffered samples, so it
  must never be shared between channels — doing so interleaves one channel's
  carry-over samples with the other channel's block and produces audible
  crosstalk and clicks.
* Each track is rendered onto one shared output timeline, block by block.
  A track that yields fewer than ``chunk`` frames for a block is simply mixed
  over the frames it produced and contributes silence afterwards; subsequent
  blocks resume at the correct absolute position.
* Mono tracks are panned with a constant-power law; stereo tracks use a
  balance control and keep their own L/R content and imaging.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

from . import audio_io, dsp


def _pan_gains(pan: float) -> tuple:
    """Constant-power pan gains for a mono source (pan in [-1, 1])."""
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def _balance_gains(pan: float, gain: float) -> tuple:
    """Stereo balance gains: pan < 0 attenuates right, pan > 0 attenuates left."""
    pan = max(-1.0, min(1.0, pan))
    if pan <= 0:
        return gain, gain * (1.0 + pan)
    return gain * (1.0 - pan), gain


class _TrackStream:
    """Streaming state for one track: reader plus one resampler per channel."""

    def __init__(self, reader: "audio_io.WavReader", track: Dict):
        self.reader = reader
        self.track = track
        self.src_sr = reader.sr
        self.channels = reader.channels
        self.target_sr = 0
        self.resamplers: List[Optional[dsp.StreamingResampler]] = []
        # Pending rendered samples (already at the target rate) per channel.
        self.pending: List[List[float]] = [[] for _ in range(reader.channels)]
        self.source_eof = False

    def prepare(self, target_sr: int) -> None:
        self.target_sr = target_sr
        self.resamplers = [
            dsp.StreamingResampler(self.src_sr, target_sr)
            if self.src_sr != target_sr else None
            for _ in range(self.channels)
        ]

    @property
    def done(self) -> bool:
        return self.source_eof and not any(self.pending)

    def _fill(self, need: int) -> None:
        """Ensure each channel has at least ``need`` pending samples, if possible."""
        if self.source_eof:
            return
        while True:
            avail = min(len(p) for p in self.pending)
            if avail >= need:
                return
            n_src = (max(1, int((need - avail) * self.src_sr / self.target_sr))
                     if any(rs is not None for rs in self.resamplers) else need - avail)
            raw = self.reader.read_chunk(n_src)
            if raw is None:
                self.source_eof = True
                for c, rs in enumerate(self.resamplers):
                    if rs is not None:
                        self.pending[c].extend(rs.flush(1 << 28))
                return
            for c, ch_data in enumerate(raw):
                rs = self.resamplers[c]
                if rs is None:
                    self.pending[c].extend(ch_data)
                else:
                    rs.push(ch_data)
                    self.pending[c].extend(rs.pull(len(ch_data)))

    def take(self, n: int) -> Optional[List[List[float]]]:
        """Return up to ``n`` rendered frames per channel, or None when ended."""
        if self.done:
            return None
        self._fill(n)
        if self.done:
            return None
        k = min(n, min(len(p) for p in self.pending))
        block = [p[:k] for p in self.pending]
        for p in self.pending:
            del p[:k]
        return block


def mixdown(tracks: Sequence[Dict], out_path: str, target_sr: Optional[int] = None,
            master_gain: float = 1.0, chunk: int = 1 << 15) -> Dict:
    """Mix ``tracks`` into ``out_path``.

    Each track is a dict: ``{"path", "gain", "pan", "muted"}``.
    """
    active = [t for t in tracks if t.get("path") and not t.get("muted")
              and os.path.isfile(t["path"])]
    if not active:
        raise ValueError("no active tracks to mix")

    streams: List[_TrackStream] = []
    try:
        for t in active:
            streams.append(_TrackStream(audio_io.WavReader(t["path"]), t))
        sr = target_sr or max(s.src_sr for s in streams)
        for s in streams:
            s.prepare(sr)

        total_frames = 0
        with audio_io.WavWriter(out_path, sr, 2, 2) as w:
            while True:
                out_l = [0.0] * chunk
                out_r = [0.0] * chunk
                n_frames = 0
                for st in streams:
                    block = st.take(chunk)
                    if block is None:
                        continue
                    n = min(len(ch_data) for ch_data in block)
                    n_frames = max(n_frames, n)
                    gain = st.track.get("gain", 1.0)
                    pan = st.track.get("pan", 0.0)
                    if st.channels == 1:
                        lg, rg = _pan_gains(pan)
                        mono = block[0]
                        gl, gr = gain * lg, gain * rg
                        for j in range(n):
                            v = mono[j]
                            out_l[j] += v * gl
                            out_r[j] += v * gr
                    else:
                        lg, rg = _balance_gains(pan, gain)
                        left, right = block[0], block[1]
                        for j in range(n):
                            out_l[j] += left[j] * lg
                            out_r[j] += right[j] * rg
                if n_frames == 0:
                    break
                # Master gain + soft clipping to guard against overload.
                total_frames += n_frames
                w.write_chunk([
                    [math.tanh(x * master_gain) for x in out_l[:n_frames]],
                    [math.tanh(x * master_gain) for x in out_r[:n_frames]],
                ])
    finally:
        for st in streams:
            st.reader.close()

    return {
        "tracks": len(active),
        "sr": sr,
        "duration": total_frames / sr if sr else 0.0,
        "frames": total_frames,
    }
