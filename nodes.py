from __future__ import annotations

import base64
import asyncio
import io
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

import numpy as np
import requests
import soundfile as sf
import torch

try:
    import folder_paths
except Exception:  # pragma: no cover - lets unit tests import outside ComfyUI
    folder_paths = None

try:
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover
    ProgressBar = None


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v3-tts-4b"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_RUNTIME_MODE = "python_worker"
DEFAULT_PYTHON_EXECUTABLE = sys.executable
DEFAULT_SGLANG_OMNI_PYTHON_PATH = ""
DEFAULT_ATTENTION_BACKEND = "triton"
DEFAULT_DISABLE_CUDA_GRAPH = True
DEFAULT_STARTUP_TIMEOUT_SECONDS = 600
RESPONSE_MODES = ["standard_wav", "stream_sse_wav", "stream_pcm"]
LOCAL_MODEL_TYPE = "HIGGS_AUDIO_V3_MODEL"
HIGGS_MODEL_FOLDER = "higgs_audio"
HIGGS_MODEL_FOLDER_NAMES = (HIGGS_MODEL_FOLDER, "llm", "LLM")
HIGGS_MODEL_MARKER_FILES = ("config.json", "model_index.json", "generation_config.json")
_LOCAL_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_WORKER_MARKER = "__HIGGS_AUDIO_V3_JSON__"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _higgs_runtime_config() -> dict[str, Any]:
    return {
        "runtime_mode": os.getenv("HIGGS_AUDIO_V3_RUNTIME_MODE", DEFAULT_RUNTIME_MODE).strip() or DEFAULT_RUNTIME_MODE,
        "python_executable": os.getenv("HIGGS_AUDIO_V3_PYTHON_EXECUTABLE", DEFAULT_PYTHON_EXECUTABLE).strip()
        or DEFAULT_PYTHON_EXECUTABLE,
        "sglang_omni_python_path": os.getenv(
            "HIGGS_AUDIO_V3_SGLANG_OMNI_PYTHON_PATH",
            DEFAULT_SGLANG_OMNI_PYTHON_PATH,
        ).strip(),
        "attention_backend": os.getenv("HIGGS_AUDIO_V3_ATTENTION_BACKEND", DEFAULT_ATTENTION_BACKEND).strip()
        or DEFAULT_ATTENTION_BACKEND,
        "disable_cuda_graph": _env_bool("HIGGS_AUDIO_V3_DISABLE_CUDA_GRAPH", DEFAULT_DISABLE_CUDA_GRAPH),
        "startup_timeout_seconds": int(
            os.getenv("HIGGS_AUDIO_V3_STARTUP_TIMEOUT_SECONDS", str(DEFAULT_STARTUP_TIMEOUT_SECONDS))
        ),
    }


def _register_higgs_model_folder() -> None:
    if folder_paths is None:
        return
    try:
        default_dir = os.path.join(folder_paths.models_dir, HIGGS_MODEL_FOLDER)
        folder_paths.add_model_folder_path(HIGGS_MODEL_FOLDER, default_dir, is_default=True)
    except Exception:
        pass


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen = set()
    out = []
    for path in paths:
        if not path:
            continue
        normalized = os.path.normcase(os.path.abspath(os.path.expanduser(path)))
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(path)
    return out


def _folder_paths_for(folder_name: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return folder_paths.get_folder_paths(folder_name)
    except Exception:
        return []


def _higgs_model_roots() -> list[str]:
    roots: list[str] = []
    if folder_paths is not None:
        for folder_name in HIGGS_MODEL_FOLDER_NAMES:
            roots.extend(_folder_paths_for(folder_name))
        for folder_name in HIGGS_MODEL_FOLDER_NAMES:
            try:
                roots.append(os.path.join(folder_paths.models_dir, folder_name))
            except Exception:
                pass
    return _dedupe_paths(roots)


def _is_higgs_named_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "higgs" in normalized or "bosonai" in normalized or "audio-v3" in normalized


def _is_model_dir(path: str) -> bool:
    return os.path.isdir(path) and any(os.path.isfile(os.path.join(path, marker)) for marker in HIGGS_MODEL_MARKER_FILES)


def _is_higgs_model_dir(path: str, *, allow_generic: bool = False) -> bool:
    if not _is_model_dir(path):
        return False
    return allow_generic or _is_higgs_named_path(path)


def _path_depth(root: str, path: str) -> int:
    rel = os.path.relpath(path, root)
    if rel == ".":
        return 0
    return len(rel.split(os.sep))


def _display_model_name(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    if rel == ".":
        return os.path.basename(os.path.normpath(path))
    return rel.replace(os.sep, "/")


def _iter_higgs_model_dirs() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for root in _higgs_model_roots():
        if not os.path.isdir(root):
            continue
        root_is_higgs_bucket = os.path.basename(os.path.normpath(root)).lower() == HIGGS_MODEL_FOLDER
        for dirpath, dirnames, _filenames in os.walk(root, followlinks=True):
            dirnames[:] = [d for d in dirnames if d not in {".cache", ".git", "__pycache__"}]
            depth = _path_depth(root, dirpath)
            if depth > 2:
                dirnames[:] = []
                continue
            if _is_higgs_model_dir(dirpath, allow_generic=root_is_higgs_bucket):
                found.append((_display_model_name(root, dirpath), dirpath))
                dirnames[:] = []
    return found


def _higgs_model_choices() -> list[str]:
    choices: list[str] = []
    for display_name, _path in _iter_higgs_model_dirs():
        if display_name not in choices:
            choices.append(display_name)
    if DEFAULT_MODEL_PATH not in choices:
        choices.append(DEFAULT_MODEL_PATH)
    return choices


def _resolve_higgs_model_path(model_path: str) -> str:
    selected = (model_path or DEFAULT_MODEL_PATH).strip() or DEFAULT_MODEL_PATH
    expanded = os.path.expanduser(selected)
    if os.path.isabs(expanded) and os.path.isdir(expanded):
        return expanded

    normalized = selected.replace("\\", "/").strip("/")
    aliases = [normalized]
    if normalized == DEFAULT_MODEL_PATH:
        aliases.append(os.path.basename(DEFAULT_MODEL_PATH))

    for root in _higgs_model_roots():
        if not os.path.isdir(root):
            continue
        root_is_higgs_bucket = os.path.basename(os.path.normpath(root)).lower() == HIGGS_MODEL_FOLDER
        for alias in aliases:
            candidate = os.path.join(root, *alias.split("/"))
            if _is_higgs_model_dir(candidate, allow_generic=root_is_higgs_bucket):
                return candidate

    for display_name, path in _iter_higgs_model_dirs():
        if display_name == normalized or os.path.basename(os.path.normpath(path)) == normalized:
            return path
        if normalized == DEFAULT_MODEL_PATH and os.path.basename(os.path.normpath(path)) == os.path.basename(DEFAULT_MODEL_PATH):
            return path
    return selected


_register_higgs_model_folder()


def _clean_server_url(server_url: str) -> str:
    server_url = (server_url or DEFAULT_SERVER_URL).strip().rstrip("/")
    if not server_url:
        return DEFAULT_SERVER_URL
    return server_url


def _maybe_add_python_path(path: str) -> None:
    path = (path or "").strip()
    if not path:
        return
    import sys

    if path not in sys.path:
        sys.path.insert(0, path)


def _audio_to_wav_bytes(audio: dict[str, Any]) -> bytes:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Reference audio must be a ComfyUI AUDIO object.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if waveform.ndim != 3:
        raise ValueError(f"Expected reference waveform shape [B, C, T], got {tuple(waveform.shape)}.")

    wav = waveform[0].detach().cpu().float().numpy().T
    if wav.ndim == 2 and wav.shape[1] == 1:
        wav = wav[:, 0]

    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _write_reference_audio(audio: dict[str, Any]) -> str:
    if folder_paths is not None:
        temp_dir = folder_paths.get_temp_directory()
    else:
        temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"higgs_audio_v3_ref_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        f.write(_audio_to_wav_bytes(audio))
    return path


def _build_higgs_payload(
    *,
    text: str,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    reference_audio: dict[str, Any] | None = None,
    reference_audio_path: str = "",
    reference_text: str = "",
) -> tuple[dict[str, Any], str | None]:
    payload: dict[str, Any] = {
        "input": text,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
    }

    temp_ref_path = None
    ref_path = (reference_audio_path or "").strip()
    if reference_audio is not None:
        temp_ref_path = _write_reference_audio(reference_audio)
        ref_path = temp_ref_path
    if ref_path:
        reference: dict[str, Any] = {"audio_path": ref_path}
        if reference_text and reference_text.strip():
            reference["text"] = reference_text.strip()
        payload["references"] = [reference]

    return payload, temp_ref_path


def _apply_higgs_runtime_overrides(
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

    manager = ConfigManager(config)
    return manager.merge_config(updates)


class HiggsAudioV3LocalPipeline:
    def __init__(
        self,
        *,
        model_path: str,
        sglang_omni_python_path: str,
        device: str,
        attention_backend: str,
        disable_cuda_graph: bool,
        startup_timeout_seconds: int,
    ) -> None:
        self.model_path = model_path
        self.sglang_omni_python_path = sglang_omni_python_path
        self.device = device
        self.attention_backend = attention_backend
        self.disable_cuda_graph = disable_cuda_graph
        self.startup_timeout_seconds = int(startup_timeout_seconds)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner = None
        self._client = None
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            _maybe_add_python_path(self.sglang_omni_python_path)
            self._start_loop()
            self._run_coro(self._start_runner(), self.startup_timeout_seconds)
            self._started = True

    def _start_loop(self) -> None:
        ready = threading.Event()

        def _thread_main() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(
            target=_thread_main,
            name="HiggsAudioV3LocalPipeline",
            daemon=True,
        )
        self._thread.start()
        ready.wait(timeout=10)
        if self._loop is None:
            raise RuntimeError("Failed to start Higgs local pipeline event loop.")

    def _run_coro(self, coro, timeout_seconds: int | float):
        if self._loop is None:
            raise RuntimeError("Higgs local pipeline event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=float(timeout_seconds))
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError("Timed out waiting for Higgs local pipeline.") from exc

    async def _start_runner(self) -> None:
        from sglang_omni.client import Client
        from sglang_omni.config.manager import ConfigManager
        from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

        config = ConfigManager.from_model_path(self.model_path).merge_config({})
        config = _apply_higgs_runtime_overrides(
            config,
            device=self.device,
            attention_backend=self.attention_backend,
            disable_cuda_graph=self.disable_cuda_graph,
        )
        runner = MultiProcessPipelineRunner(config)
        await runner.start(timeout=float(self.startup_timeout_seconds))
        self._runner = runner
        self._client = Client(runner.coordinator)

    async def _speech(self, payload: dict[str, Any]) -> bytes:
        if self._client is None:
            raise RuntimeError("Higgs local pipeline is not started.")
        from sglang_omni.serve.openai_api import build_speech_generate_request
        from sglang_omni.serve.protocol import CreateSpeechRequest

        req = CreateSpeechRequest(**payload)
        gen_req = build_speech_generate_request(req, self.model_path)
        result = await self._client.speech(
            gen_req,
            request_id=f"comfy-local-speech-{uuid.uuid4()}",
            response_format=req.response_format,
            speed=req.speed,
        )
        return result.audio_bytes

    def speech(self, payload: dict[str, Any], timeout_seconds: int) -> bytes:
        self.start()
        local_payload = dict(payload)
        local_payload.pop("stream", None)
        local_payload.pop("stream_format", None)
        local_payload.pop("initial_codec_chunk_frames", None)
        local_payload["response_format"] = "wav"
        return self._run_coro(self._speech(local_payload), timeout_seconds)

    def stop(self) -> None:
        with self._lock:
            if self._loop is None:
                return
            if self._runner is not None:
                try:
                    self._run_coro(self._runner.stop(), 60)
                except Exception:
                    pass
            loop = self._loop
            loop.call_soon_threadsafe(loop.stop)
            self._loop = None
            self._runner = None
            self._client = None
            self._started = False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


class HiggsAudioV3WorkerPipeline:
    def __init__(
        self,
        *,
        python_executable: str,
        model_path: str,
        sglang_omni_python_path: str,
        device: str,
        attention_backend: str,
        disable_cuda_graph: bool,
        startup_timeout_seconds: int,
    ) -> None:
        self.python_executable = (python_executable or "python").strip()
        self.model_path = model_path
        self.sglang_omni_python_path = sglang_omni_python_path
        self.device = device
        self.attention_backend = attention_backend
        self.disable_cuda_graph = disable_cuda_graph
        self.startup_timeout_seconds = int(startup_timeout_seconds)
        self._proc: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return

            script_path = os.path.join(os.path.dirname(__file__), "local_worker.py")
            python_cmd = shlex.split(self.python_executable, posix=(os.name != "nt"))
            if not python_cmd:
                python_cmd = [sys.executable]
            args = [
                *python_cmd,
                "-u",
                script_path,
                "--model-path",
                self.model_path,
                "--device",
                self.device,
                "--attention-backend",
                self.attention_backend,
                "--startup-timeout-seconds",
                str(self.startup_timeout_seconds),
            ]
            if self.disable_cuda_graph:
                args.append("--disable-cuda-graph")
            if (self.sglang_omni_python_path or "").strip():
                args.extend(["--python-path", self.sglang_omni_python_path.strip()])

            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            msg = self._wait_for_message(None, self.startup_timeout_seconds)
            if msg.get("type") == "ready":
                return
            if msg.get("type") == "error":
                raise RuntimeError(f"Higgs worker failed to start: {msg.get('error')}")
            raise RuntimeError(f"Unexpected Higgs worker startup message: {msg}")

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line.startswith(_WORKER_MARKER):
                continue
            try:
                self._messages.put(json.loads(line[len(_WORKER_MARKER) :]))
            except json.JSONDecodeError:
                continue

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())
            self._stderr_tail = self._stderr_tail[-80:]

    def _wait_for_message(self, request_id: str | None, timeout_seconds: int | float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stderr = "\n".join(self._stderr_tail[-20:])
                raise TimeoutError(f"Timed out waiting for Higgs worker. Recent stderr:\n{stderr}")
            try:
                msg = self._messages.get(timeout=min(1.0, remaining))
            except queue.Empty:
                proc = self._proc
                if proc is not None and proc.poll() is not None:
                    stderr = "\n".join(self._stderr_tail[-20:])
                    raise RuntimeError(f"Higgs worker exited with code {proc.returncode}. Recent stderr:\n{stderr}")
                continue
            if request_id is None or msg.get("id") == request_id or msg.get("type") in {"ready", "error"}:
                return msg

    def speech(self, payload: dict[str, Any], timeout_seconds: int) -> bytes:
        self.start()
        with self._lock:
            proc = self._proc
            if proc is None or proc.stdin is None or proc.poll() is not None:
                raise RuntimeError("Higgs worker is not running.")
            request_id = uuid.uuid4().hex
            proc.stdin.write(json.dumps({"type": "speech", "id": request_id, "payload": payload}, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            msg = self._wait_for_message(request_id, timeout_seconds)
            if msg.get("type") == "speech_result":
                return base64.b64decode(msg["audio_b64"])
            if msg.get("type") == "error":
                raise RuntimeError(f"Higgs worker request failed: {msg.get('error')}")
            raise RuntimeError(f"Unexpected Higgs worker response: {msg}")

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and proc.poll() is None:
                proc.stdin.write(json.dumps({"type": "stop"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        self._proc = None

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


def _read_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    if not data:
        raise ValueError("Higgs server returned an empty audio response.")

    try:
        wav, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        prefix = data[:80]
        raise ValueError(f"Could not decode Higgs audio response as WAV/FLAC/OGG bytes. Prefix={prefix!r}") from exc

    return wav, int(sample_rate)


def _array_to_comfy_audio(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if wav.size == 0:
        raise ValueError("Decoded audio response contains zero samples.")

    waveform = torch.from_numpy(wav.T.copy()).float().unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _wav_bytes_to_comfy_audio(data: bytes) -> dict[str, Any]:
    wav, sample_rate = _read_audio_bytes(data)
    return _array_to_comfy_audio(wav, sample_rate)


def _wav_chunks_to_comfy_audio(chunks: list[bytes]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("Higgs server returned an empty SSE audio stream.")

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None
    for chunk in chunks:
        wav, chunk_sample_rate = _read_audio_bytes(chunk)
        chunk_channels = wav.shape[1]
        if sample_rate is None:
            sample_rate = chunk_sample_rate
            channels = chunk_channels
        elif chunk_sample_rate != sample_rate:
            raise ValueError(f"SSE WAV chunks use mixed sample rates: {sample_rate} and {chunk_sample_rate}.")
        elif chunk_channels != channels:
            raise ValueError(f"SSE WAV chunks use mixed channel counts: {channels} and {chunk_channels}.")
        arrays.append(wav)

    return _array_to_comfy_audio(np.concatenate(arrays, axis=0), int(sample_rate or DEFAULT_SAMPLE_RATE))


def _pcm16_bytes_to_comfy_audio(data: bytes, sample_rate: int) -> dict[str, Any]:
    if not data:
        raise ValueError("Higgs server returned an empty PCM stream.")
    if len(data) % 2:
        data = data[:-1]
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples.copy()).float().view(1, 1, -1)
    return {"waveform": waveform, "sample_rate": int(sample_rate or DEFAULT_SAMPLE_RATE)}


def _sample_rate_from_headers(headers: requests.structures.CaseInsensitiveDict[str]) -> int:
    for key in (
        "x-sample-rate",
        "x-audio-sample-rate",
        "sample-rate",
        "x-sglang-sample-rate",
    ):
        value = headers.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    content_type = headers.get("content-type", "")
    for part in content_type.split(";"):
        part = part.strip().lower()
        if part.startswith("rate="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                pass
    return DEFAULT_SAMPLE_RATE


def _raise_for_bad_response(resp: requests.Response) -> None:
    if resp.ok:
        return
    text = resp.text[:1200] if resp.text else ""
    raise RuntimeError(f"Higgs server request failed: HTTP {resp.status_code} {resp.reason}. {text}")


def _collect_sse_wav_chunks(resp: requests.Response) -> list[bytes]:
    chunks: list[bytes] = []
    for line in resp.iter_lines():
        if not line or line == b"data: [DONE]":
            continue
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[len(b"data: "):])
        if event.get("finish_reason") == "stop":
            break
        audio = event.get("audio") or {}
        encoded = audio.get("data")
        if encoded:
            chunks.append(base64.b64decode(encoded))
    return chunks


class HiggsAudioV3TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, how are you?",
                        "tooltip": "Text sent to /v1/audio/speech. Inline Higgs control tokens are supported.",
                    },
                ),
                "response_mode": (RESPONSE_MODES, {"default": "standard_wav"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 64}),
                "timeout_seconds": ("INT", {"default": 300, "min": 5, "max": 3600, "step": 5}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Optional ComfyUI audio used for zero-shot voice cloning."}),
                "reference_audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional path or URL visible to the Higgs/SGLang server. Ignored when reference_audio is connected.",
                    },
                ),
                "reference_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Transcript for the reference audio. Hugging Face recommends providing it for better cloning.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "request_json")
    FUNCTION = "generate"
    CATEGORY = "audio/Higgs Audio V3"
    DESCRIPTION = "Generate 24 kHz speech with bosonai/higgs-audio-v3-tts-4b through an SGLang-Omni /v1/audio/speech server."

    def generate(
        self,
        server_url: str,
        text: str,
        response_mode: str,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        timeout_seconds: int,
        reference_audio: dict[str, Any] | None = None,
        reference_audio_path: str = "",
        reference_text: str = "",
    ):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        if response_mode not in RESPONSE_MODES:
            raise ValueError(f"Unsupported response_mode: {response_mode}")

        pbar = ProgressBar(3) if ProgressBar is not None else None
        if pbar:
            pbar.update_absolute(1, 3)

        payload, temp_ref_path = _build_higgs_payload(
            text=text,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            reference_audio=reference_audio,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )

        if response_mode == "stream_sse_wav":
            payload["stream"] = True
        elif response_mode == "stream_pcm":
            payload.update(
                {
                    "stream": True,
                    "stream_format": "audio",
                    "response_format": "pcm",
                    "initial_codec_chunk_frames": 1,
                }
            )

        url = f"{_clean_server_url(server_url)}/v1/audio/speech"
        request_json = json.dumps(payload, ensure_ascii=False, indent=2)

        try:
            if response_mode == "stream_sse_wav":
                with requests.post(url, json=payload, timeout=int(timeout_seconds), stream=True) as resp:
                    _raise_for_bad_response(resp)
                    chunks = _collect_sse_wav_chunks(resp)
                audio = _wav_chunks_to_comfy_audio(chunks)
            elif response_mode == "stream_pcm":
                with requests.post(url, json=payload, timeout=int(timeout_seconds), stream=True) as resp:
                    _raise_for_bad_response(resp)
                    sample_rate = _sample_rate_from_headers(resp.headers)
                    data = b"".join(resp.iter_content(chunk_size=1024 * 64))
                audio = _pcm16_bytes_to_comfy_audio(data, sample_rate)
            else:
                resp = requests.post(url, json=payload, timeout=int(timeout_seconds))
                _raise_for_bad_response(resp)
                audio = _wav_bytes_to_comfy_audio(resp.content)
        finally:
            if temp_ref_path:
                try:
                    os.remove(temp_ref_path)
                except OSError:
                    pass

        if pbar:
            pbar.update_absolute(3, 3)

        return (audio, request_json)


class HiggsAudioV3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _higgs_model_choices()
        return {
            "required": {
                "model_path": (
                    model_choices,
                    {
                        "default": model_choices[0],
                        "tooltip": "Local ComfyUI model folder is preferred. Falls back to Hugging Face model id if no local copy is found.",
                    },
                ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = (LOCAL_MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "audio/Higgs Audio V3"
    DESCRIPTION = "Load a Higgs Audio V3 TTS pipeline inside ComfyUI without exposing an HTTP server."

    def load_model(
        self,
        model_path: str,
        device: str,
    ):
        runtime_config = _higgs_runtime_config()
        runtime_mode = runtime_config["runtime_mode"]
        python_executable = runtime_config["python_executable"]
        sglang_omni_python_path = runtime_config["sglang_omni_python_path"]
        attention_backend = runtime_config["attention_backend"]
        disable_cuda_graph = runtime_config["disable_cuda_graph"]
        startup_timeout_seconds = runtime_config["startup_timeout_seconds"]

        selected_model_path = (model_path or DEFAULT_MODEL_PATH).strip() or DEFAULT_MODEL_PATH
        resolved_model_path = _resolve_higgs_model_path(selected_model_path)
        key = (
            runtime_mode,
            (python_executable or "python").strip(),
            resolved_model_path,
            (sglang_omni_python_path or "").strip(),
            device,
            attention_backend,
            bool(disable_cuda_graph),
            int(startup_timeout_seconds),
        )
        model = _LOCAL_MODEL_CACHE.get(key)
        if model is None:
            if runtime_mode == "python_worker":
                model = HiggsAudioV3WorkerPipeline(
                    python_executable=python_executable,
                    model_path=resolved_model_path,
                    sglang_omni_python_path=sglang_omni_python_path,
                    device=device,
                    attention_backend=attention_backend,
                    disable_cuda_graph=bool(disable_cuda_graph),
                    startup_timeout_seconds=int(startup_timeout_seconds),
                )
            elif runtime_mode == "in_process":
                model = HiggsAudioV3LocalPipeline(
                    model_path=resolved_model_path,
                    sglang_omni_python_path=sglang_omni_python_path,
                    device=device,
                    attention_backend=attention_backend,
                    disable_cuda_graph=bool(disable_cuda_graph),
                    startup_timeout_seconds=int(startup_timeout_seconds),
                )
            else:
                raise ValueError(f"Unsupported runtime_mode: {runtime_mode}")
            model.start()
            _LOCAL_MODEL_CACHE[key] = model

        info = {
            "model_path": selected_model_path,
            "resolved_model_path": resolved_model_path,
            "model_source": "local_comfy_model_path" if os.path.isdir(resolved_model_path) else "huggingface_or_custom_path",
            "runtime_mode": runtime_mode,
            "device": device,
            "mode": "local_pipeline",
        }
        return (model, json.dumps(info, ensure_ascii=False, indent=2))


class HiggsAudioV3LocalTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (LOCAL_MODEL_TYPE,),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, how are you?",
                        "tooltip": "Text sent to the local Higgs Audio V3 pipeline. Inline Higgs control tokens are supported.",
                    },
                ),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 64}),
                "timeout_seconds": ("INT", {"default": 300, "min": 5, "max": 3600, "step": 5}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Optional ComfyUI audio used for zero-shot voice cloning."}),
                "reference_audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional path visible to the local Higgs pipeline. Ignored when reference_audio is connected.",
                    },
                ),
                "reference_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Transcript for the reference audio.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "request_json")
    FUNCTION = "generate"
    CATEGORY = "audio/Higgs Audio V3"
    DESCRIPTION = "Generate speech through a loaded local Higgs Audio V3 pipeline without a separate HTTP server."

    def generate(
        self,
        model: HiggsAudioV3LocalPipeline,
        text: str,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
        timeout_seconds: int,
        reference_audio: dict[str, Any] | None = None,
        reference_audio_path: str = "",
        reference_text: str = "",
    ):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        if not hasattr(model, "speech"):
            raise ValueError("model must be loaded by Higgs Audio V3 Model Loader.")

        pbar = ProgressBar(3) if ProgressBar is not None else None
        if pbar:
            pbar.update_absolute(1, 3)

        payload, temp_ref_path = _build_higgs_payload(
            text=text,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            reference_audio=reference_audio,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )
        request_json = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            data = model.speech(payload, timeout_seconds=int(timeout_seconds))
            audio = _wav_bytes_to_comfy_audio(data)
        finally:
            if temp_ref_path:
                try:
                    os.remove(temp_ref_path)
                except OSError:
                    pass

        if pbar:
            pbar.update_absolute(3, 3)

        return (audio, request_json)
