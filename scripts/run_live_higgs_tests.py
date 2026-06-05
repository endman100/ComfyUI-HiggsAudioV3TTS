from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import wave
from pathlib import Path
from typing import Any

import requests


def validate_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        frames = wf.readframes(frame_count)

    if sample_width != 2:
        raise AssertionError(
            f"{path.name}: expected 16-bit WAV, got sample width {sample_width}"
        )
    if len(frames) < 2:
        raise AssertionError(f"{path.name}: no samples")

    usable = len(frames) // 2 * 2
    samples = struct.unpack("<" + "h" * (usable // 2), frames[:usable])
    rms = math.sqrt(sum(float(x) * x for x in samples) / len(samples)) / 32768.0
    peak = max(abs(x) for x in samples) / 32768.0
    seconds = frame_count / float(sample_rate)

    if sample_rate != 24000:
        raise AssertionError(f"{path.name}: expected 24000 Hz, got {sample_rate}")
    if seconds < 0.35:
        raise AssertionError(f"{path.name}: too short {seconds:.3f}s")
    if rms < 0.001 or peak < 0.01:
        raise AssertionError(
            f"{path.name}: likely silence rms={rms:.6f} peak={peak:.6f}"
        )

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "seconds": round(seconds, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "bytes": path.stat().st_size,
    }


def validate_pcm_to_wav(
    pcm_path: Path, wav_path: Path, sample_rate: int = 24000, channels: int = 1
) -> dict[str, Any]:
    data = pcm_path.read_bytes()
    if len(data) < sample_rate * 2 * 0.25:
        raise AssertionError(f"{pcm_path.name}: too few PCM bytes {len(data)}")

    usable = len(data) // 2 * 2
    samples = struct.unpack("<" + "h" * (usable // 2), data[:usable])
    rms = math.sqrt(sum(float(x) * x for x in samples) / len(samples)) / 32768.0
    peak = max(abs(x) for x in samples) / 32768.0
    if rms < 0.001 or peak < 0.01:
        raise AssertionError(
            f"{pcm_path.name}: likely silence rms={rms:.6f} peak={peak:.6f}"
        )

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data[:usable])

    return validate_wav(wav_path)


def post_wav(
    base_url: str, out_dir: Path, name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    response = requests.post(base_url, json=payload, timeout=180)
    response.raise_for_status()
    path = out_dir / f"{name}.wav"
    path.write_bytes(response.content)
    return validate_wav(path)


def run_live_tests(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/") + "/v1/audio/speech"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_text = (
        "Hey, Adam here. Let's create something that feels real, sounds human, "
        "and connects every time."
    )
    common_sampling = {"temperature": 0.8, "top_k": 50}
    results: dict[str, Any] = {}

    results["zero_shot"] = post_wav(
        base_url,
        out_dir,
        "zero_shot",
        {
            "input": "Hello, how are you? This is a short Higgs Audio test.",
            "max_new_tokens": 512,
            **common_sampling,
        },
    )

    results["inline_control"] = post_wav(
        base_url,
        out_dir,
        "inline_control",
        {
            "input": (
                "<|emotion:amusement|><|prosody:expressive_high|>"
                "Wait, wait, that was kind of hilarious. "
                "<|sfx:laughter|>Hehe, I was not ready for that."
            ),
            "max_new_tokens": 768,
            **common_sampling,
        },
    )

    reference = [{"audio_path": args.reference_audio, "text": reference_text}]
    results["voice_clone"] = post_wav(
        base_url,
        out_dir,
        "voice_clone",
        {
            "input": "Have a nice day and enjoy southern California sunshine.",
            "references": reference,
            "max_new_tokens": 768,
            **common_sampling,
        },
    )

    sse_path = out_dir / "stream_sse.wav"
    chunk_count = 0
    with requests.post(
        base_url,
        json={
            "input": "Get the trust fund to the bank early.",
            "references": reference,
            "stream": True,
            "max_new_tokens": 768,
        },
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()
        with sse_path.open("wb") as f:
            for line in response.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[len(b"data: ") :])
                if event.get("finish_reason") == "stop":
                    break
                audio = event.get("audio") or {}
                if audio.get("data"):
                    f.write(base64.b64decode(audio["data"]))
                    chunk_count += 1
    if chunk_count < 1:
        raise AssertionError("stream_sse: no audio chunks")
    results["stream_sse"] = validate_wav(sse_path) | {"chunks": chunk_count}

    pcm_path = out_dir / "stream_pcm.pcm"
    response = requests.post(
        base_url,
        json={
            "input": "This raw PCM stream should become a normal wave file.",
            "references": reference,
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
            "initial_codec_chunk_frames": 1,
            "max_new_tokens": 768,
        },
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    with pcm_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    sample_rate = int(
        response.headers.get("x-sample-rate")
        or response.headers.get("X-Sample-Rate")
        or 24000
    )
    results["stream_pcm"] = validate_pcm_to_wav(
        pcm_path, out_dir / "stream_pcm.wav", sample_rate=sample_rate
    ) | {
        "pcm_bytes": pcm_path.stat().st_size,
        "content_type": response.headers.get("content-type"),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--reference-audio", default="docs/_static/audio/male-voice.wav"
    )
    parser.add_argument("--out-dir", default="live_test_outputs")
    args = parser.parse_args()
    print(json.dumps(run_live_tests(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
