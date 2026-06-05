from .nodes import HiggsAudioV3TTS

NODE_CLASS_MAPPINGS = {
    "HiggsAudioV3TTS": HiggsAudioV3TTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsAudioV3TTS": "Higgs Audio V3 TTS",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.1.0"

print(f"[ComfyUI-HiggsAudioV3TTS] loaded v{__version__}")
