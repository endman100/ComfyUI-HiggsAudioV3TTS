from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import traceback
from typing import Any

MARKER = "__HIGGS_AUDIO_V3_JSON__"


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(MARKER + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _maybe_add_python_path(path: str) -> None:
    path = (path or "").strip()
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _apply_runtime_overrides(
    config: Any,
    *,
    device: str,
    attention_backend: str,
    disable_cuda_graph: bool,
) -> Any:
    updates: dict[str, Any] = {}
    stages = list(getattr(config, "stages", []) or [])
    for idx, stage in enumerate(stages):
        name = getattr(stage, "name", "")
        factory_args = dict(getattr(stage, "factory_args", None) or {})
        if name in {"audio_encoder", "tts_engine", "vocoder"} and "device" in factory_args:
            updates[f"stages.{idx}.factory_args.device"] = device
        if name == "tts_engine":
            overrides = dict(factory_args.get("server_args_overrides") or {})
            if attention_backend != "default":
                overrides["attention_backend"] = attention_backend
            overrides["disable_cuda_graph"] = bool(disable_cuda_graph)
            factory_args["server_args_overrides"] = overrides
            updates[f"stages.{idx}.factory_args"] = factory_args

    if not updates:
        return config
    from sglang_omni.config.manager import ConfigManager

    return ConfigManager(config).merge_config(updates)


async def _start_client(args):
    from sglang_omni.client import Client
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    config = ConfigManager.from_model_path(args.model_path).merge_config({})
    config = _apply_runtime_overrides(
        config,
        device=args.device,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    runner = MultiProcessPipelineRunner(config)
    await runner.start(timeout=float(args.startup_timeout_seconds))
    return runner, Client(runner.coordinator)


async def _speech(client, model_path: str, payload: dict[str, Any]) -> bytes:
    from sglang_omni.serve.openai_api import build_speech_generate_request
    from sglang_omni.serve.protocol import CreateSpeechRequest

    local_payload = dict(payload)
    local_payload.pop("stream", None)
    local_payload.pop("stream_format", None)
    local_payload.pop("initial_codec_chunk_frames", None)
    local_payload["response_format"] = "wav"

    req = CreateSpeechRequest(**local_payload)
    gen_req = build_speech_generate_request(req, model_path)
    result = await client.speech(
        gen_req,
        request_id=f"comfy-worker-speech-{id(payload)}",
        response_format=req.response_format,
        speed=req.speed,
    )
    return result.audio_bytes


async def _run(args) -> int:
    _maybe_add_python_path(args.python_path)
    runner = None
    try:
        runner, client = await _start_client(args)
        _emit({"type": "ready"})
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            try:
                msg = json.loads(line)
                msg_type = msg.get("type")
                if msg_type == "stop":
                    break
                if msg_type != "speech":
                    continue
                audio_bytes = await _speech(client, args.model_path, msg["payload"])
                _emit(
                    {
                        "type": "speech_result",
                        "id": msg.get("id"),
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                    }
                )
            except Exception as exc:
                _emit(
                    {
                        "type": "error",
                        "id": msg.get("id") if isinstance(msg, dict) else None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as exc:
        _emit(
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    finally:
        if runner is not None:
            try:
                await runner.stop()
            except Exception:
                pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--python-path", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--attention-backend",
        default="triton",
        choices=["triton", "flashinfer", "default"],
    )
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())

