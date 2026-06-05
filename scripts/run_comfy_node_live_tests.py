from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _import_node(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from nodes import HiggsAudioV3TTS

    return HiggsAudioV3TTS


def _validate_and_save(audio: dict[str, Any], path: Path) -> dict[str, Any]:
    waveform = audio["waveform"].detach().cpu().float().numpy()[0].T
    sample_rate = int(audio["sample_rate"])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate, format="WAV", subtype="PCM_16")

    rms = float(np.sqrt(np.mean(np.square(waveform))))
    peak = float(np.max(np.abs(waveform)))
    seconds = waveform.shape[0] / sample_rate
    if sample_rate != 24000:
        raise AssertionError(f"{path.name}: expected 24000 Hz, got {sample_rate}")
    if seconds < 0.35:
        raise AssertionError(f"{path.name}: too short {seconds:.3f}s")
    if rms < 0.001 or peak < 0.01:
        raise AssertionError(
            f"{path.name}: likely silence rms={rms:.6f} peak={peak:.6f}"
        )
    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "seconds": round(seconds, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to the ComfyUI-HiggsAudioV3TTS custom node repo.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--reference-audio",
        default="docs/_static/audio/male-voice.wav",
    )
    parser.add_argument(
        "--reference-text",
        default=(
            "Hey, Adam here. Let's create something that feels real, "
            "sounds human, and connects every time."
        ),
    )
    parser.add_argument("--out-dir", default="comfy_node_live_outputs")
    args = parser.parse_args()

    node_cls = _import_node(Path(args.repo_root))
    node = node_cls()
    out_dir = Path(args.out_dir)

    cases = {
        "stream_sse_wav": "Get the trust fund to the bank early.",
        "stream_pcm": (
            "This raw PCM streaming mode should produce normal speech through "
            "the ComfyUI custom node."
        ),
    }
    results: dict[str, Any] = {}
    for mode, text in cases.items():
        audio, request_json = node.generate(
            server_url=args.base_url,
            text=text,
            response_mode=mode,
            temperature=0.8,
            top_k=50,
            max_new_tokens=768,
            timeout_seconds=300,
            reference_audio=None,
            reference_audio_path=args.reference_audio,
            reference_text=args.reference_text,
        )
        results[mode] = _validate_and_save(audio, out_dir / f"{mode}.wav")
        results[mode]["request"] = json.loads(request_json)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
