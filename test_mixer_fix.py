"""
Reproduction / regression tests for the multi-track mixdown.

Scenario from the bug report: a stereo track and a mono track with *different*
sample rates and different lengths must be mixed without changing either
track's channel content or stereo image.

The fixed mixer must be bit-for-bit equal (interior samples) to independently
resampling each channel of each track on a whole-file basis and summing.

Run: python3 test_mixer_fix.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import audio_io, dsp, mixer


def sine(freq, n, sr, amp=0.2, phase0=0.0):
    return [amp * math.sin(2 * math.pi * freq * i / sr + phase0) for i in range(n)]


def write_wav(path, channels, sr):
    with audio_io.WavWriter(path, sr, len(channels), 2) as w:
        w.write_chunk(channels)


def read_stereo(path):
    with audio_io.WavReader(path) as r:
        assert r.channels == 2
        left, right = [], []
        for chunk in r.iter_chunks():
            left.extend(chunk[0])
            right.extend(chunk[1])
        return r.sr, left, right


def ideal_stream(samples, src_sr, dst_sr):
    """Whole-file version of the mixer's streaming linear resampler."""
    rs = dsp.StreamingResampler(src_sr, dst_sr)
    rs.push(samples)
    return rs.flush(1 << 28)


def error_db(got, expected):
    """Total error-power / signal-power in dB over the overlapping region."""
    n = min(len(got), len(expected))
    err = sum((got[i] - expected[i]) ** 2 for i in range(n))
    ref = sum(x * x for x in expected[:n])
    return 10 * math.log10((err + 1e-30) / (ref + 1e-30))


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db_ = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db_)


def test_downsample_stereo(tmp, chunk):
    """Stereo 48 kHz downmixed to a 44.1 kHz project (mono track silent):
    L/R must each equal the ideal per-channel resample, no crosstalk."""
    n_s = int(48000 * 2.5)
    sl = sine(300, n_s, 48000, phase0=0.3)
    sr_ = sine(700, n_s, 48000, phase0=1.1)
    stereo = os.path.join(tmp, f"s48_{chunk}.wav")
    write_wav(stereo, [sl, sr_], 48000)
    mono = os.path.join(tmp, f"m441_{chunk}.wav")
    write_wav(mono, [sine(500, int(44100 * 1.3), 44100, amp=0.0)], 44100)

    out = os.path.join(tmp, f"o1_{chunk}.wav")
    mixer.mixdown(
        [{"path": stereo, "gain": 1.0, "pan": 0.0},
         {"path": mono, "gain": 1.0, "pan": 0.0}],
        out, target_sr=44100, master_gain=0.05, chunk=chunk,
    )
    osr, left, right = read_stereo(out)
    assert osr == 44100
    il = [math.tanh(0.05 * x) for x in ideal_stream(sl, 48000, 44100)]
    ir = [math.tanh(0.05 * x) for x in ideal_stream(sr_, 48000, 44100)]
    assert len(left) == len(il), f"length drift: {len(left)} != {len(il)}"
    el, er = error_db(left[100:-100], il[100:-100]), error_db(right[100:-100], ir[100:-100])
    print(f"[chunk={chunk}] downsample stereo: err L={el:6.1f} dB R={er:6.1f} dB")
    assert el < -50.0, f"left channel corrupted: {el:.1f} dB error"
    assert er < -50.0, f"right channel corrupted: {er:.1f} dB error"


def test_upsample_stereo_with_mono(tmp, chunk):
    """Stereo 44.1 kHz upsampled into a 48 kHz project together with a shorter
    mono 48 kHz track: channel content, alignment and total length exact."""
    n_s = int(44100 * 2.5)
    sl = sine(300, n_s, 44100, phase0=0.3)
    sr_ = sine(700, n_s, 44100, phase0=1.1)
    stereo = os.path.join(tmp, f"s441_{chunk}.wav")
    write_wav(stereo, [sl, sr_], 44100)
    mono_src = sine(440, int(48000 * 1.3), 48000, amp=0.2)
    mono = os.path.join(tmp, f"m48_{chunk}.wav")
    write_wav(mono, [mono_src], 48000)

    out = os.path.join(tmp, f"o2_{chunk}.wav")
    mixer.mixdown(
        [{"path": stereo, "gain": 1.0, "pan": 0.0},
         {"path": mono, "gain": 0.5, "pan": 0.0}],
        out, target_sr=48000, master_gain=0.05, chunk=chunk,
    )
    osr, left, right = read_stereo(out)
    assert osr == 48000
    il = ideal_stream(sl, 44100, 48000)
    ir = ideal_stream(sr_, 44100, 48000)
    assert len(left) == len(il), f"length drift: {len(left)} != {len(il)}"

    g = 0.5 * math.sqrt(0.5)
    exp_l = [math.tanh(0.05 * (il[i] + (g * mono_src[i] if i < len(mono_src) else 0.0)))
             for i in range(40000, 60000)]
    exp_r = [math.tanh(0.05 * (ir[i] + (g * mono_src[i] if i < len(mono_src) else 0.0)))
             for i in range(40000, 60000)]
    cl, cr = corr(left[40000:60000], exp_l), corr(right[40000:60000], exp_r)
    print(f"[chunk={chunk}] upsample stereo+mono: corr L={cl:.5f} R={cr:.5f}")
    assert cl > 0.9999, f"left alignment broken: corr={cl:.5f}"
    assert cr > 0.9999, f"right alignment broken: corr={cr:.5f}"


def test_pan_identity(tmp, chunk):
    """A centered stereo track with no other resampling must survive a
    same-rate mix unchanged; mono pan law stays constant power."""
    n = 48000
    stereo = os.path.join(tmp, f"sreg_{chunk}.wav")
    write_wav(stereo, [sine(300, n, 48000), sine(700, n, 48000)], 48000)
    out = os.path.join(tmp, f"o3_{chunk}.wav")
    mixer.mixdown([{"path": stereo, "gain": 1.0, "pan": 0.0}],
                  out, target_sr=48000, master_gain=0.05, chunk=chunk)
    _, left, right = read_stereo(out)
    assert len(left) == n
    el = error_db(left, [math.tanh(0.05 * v) for v in sine(300, n, 48000)])
    er = error_db(right, [math.tanh(0.05 * v) for v in sine(700, n, 48000)])
    assert el < -50.0 and er < -50.0
    print(f"[chunk={chunk}] same-rate stereo identity OK ({el:.0f}/{er:.0f} dB)")


def test_many_tracks_variable_lengths(tmp):
    """Many tracks with different rates (8k..48k), different channels and very
    different lengths: every stereo channel must match its ideal independent
    resample; shorter tracks pad with silence and must not shift the rest."""
    rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
    tracks = []
    stereo_contrib = []
    mono_contrib = []
    sr_p = 48000
    total_len = 0
    for k, rate in enumerate(rates):
        dur = 0.4 + 0.35 * k  # strongly differing lengths
        n = int(rate * dur)
        if k % 2 == 0:  # mono track, various pan positions
            pan = (k % 3 - 1) * 0.5
            sig = sine(220 + 70 * k, n, rate, amp=0.05)
            path = os.path.join(tmp, f"mono_{rate}.wav")
            write_wav(path, [sig], rate)
            tracks.append({"path": path, "gain": 0.7, "pan": pan})
            up = ideal_stream(sig, rate, sr_p)
            lg, rg = mixer._pan_gains(pan)
            mono_contrib.append((up, 0.7 * lg, 0.7 * rg))
        else:  # stereo track with distinct L/R content
            fl_ = sine(300 + 30 * k, n, rate, amp=0.07, phase0=0.2 * k)
            fr_ = sine(800 + 40 * k, n, rate, amp=0.07, phase0=0.5 * k + 0.7)
            path = os.path.join(tmp, f"st_{rate}.wav")
            write_wav(path, [fl_, fr_], rate)
            tracks.append({"path": path, "gain": 1.0, "pan": 0.0})
            stereo_contrib.append((ideal_stream(fl_, rate, sr_p),
                                   ideal_stream(fr_, rate, sr_p)))
        total_len = max(total_len, int(round(n * sr_p / rate)))

    ideal_l = [0.0] * total_len
    ideal_r = [0.0] * total_len
    for rl, rr in stereo_contrib:
        for i, v in enumerate(rl):
            ideal_l[i] += v
        for i, v in enumerate(rr):
            ideal_r[i] += v
    for sig, lg, rg in mono_contrib:
        for i, v in enumerate(sig):
            ideal_l[i] += v * lg
            ideal_r[i] += v * rg

    out = os.path.join(tmp, "many.wav")
    mixer.mixdown(tracks, out, target_sr=sr_p, master_gain=0.05, chunk=8192)
    osr, left, right = read_stereo(out)
    assert osr == sr_p
    assert len(left) == len(ideal_l), f"{len(left)} != {len(ideal_l)}"
    el = error_db(left[200:-200], [math.tanh(0.05 * x) for x in ideal_l[200:-200]])
    er = error_db(right[200:-200], [math.tanh(0.05 * x) for x in ideal_r[200:-200]])
    print(f"many-tracks stress: err L={el:.1f} dB R={er:.1f} dB, frames={len(left)}")
    assert el < -45.0 and er < -45.0


def test_mono_pan_position(tmp):
    """A panned mono track stays at the exact constant-power L/R ratio after a
    rate conversion (pan must not drift along the timeline)."""
    rate, n = 32000, 32000
    src = os.path.join(tmp, "panmono.wav")
    write_wav(src, [sine(440, n, rate, amp=0.3)], rate)
    out = os.path.join(tmp, "panout.wav")
    mixer.mixdown([{"path": src, "gain": 1.0, "pan": -0.6}],
                  out, target_sr=48000, master_gain=0.1, chunk=3000)
    _, left, right = read_stereo(out)
    lg, rg = mixer._pan_gains(-0.6)
    ratio = rg / lg
    # Measure the R/L *energy* ratio in short windows across the timeline;
    # resampling block boundaries must not make the pan wander.
    import statistics
    win = 2400
    seg_ratios = []
    for start in range(20000, 40000, win):
        el_ = sum(v * v for v in left[start:start + win])
        er_ = sum(v * v for v in right[start:start + win])
        seg_ratios.append(math.sqrt(er_ / el_))
    mean = statistics.mean(seg_ratios)
    spread = max(seg_ratios) - min(seg_ratios)
    print(f"pan: mean R/L={mean:.4f} expected={ratio:.4f} window spread={spread:.2e}")
    assert abs(mean - ratio) < 0.01
    assert spread < 0.01, "pan position drifts along the timeline"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for chunk in (4096, 1 << 15):
            test_downsample_stereo(tmp, chunk)
            test_upsample_stereo_with_mono(tmp, chunk)
            test_pan_identity(tmp, chunk)
        test_many_tracks_variable_lengths(tmp)
        test_mono_pan_position(tmp)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
