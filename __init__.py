from .nodes import HiggsAudioV3LocalTTS, HiggsAudioV3ModelLoader, HiggsAudioV3TTS

NODE_CLASS_MAPPINGS = {
    "HiggsAudioV3TTS": HiggsAudioV3TTS,
    "HiggsAudioV3ModelLoader": HiggsAudioV3ModelLoader,
    "HiggsAudioV3LocalTTS": HiggsAudioV3LocalTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsAudioV3TTS": "Higgs Audio V3 TTS",
    "HiggsAudioV3ModelLoader": "Higgs Audio V3 Model Loader",
    "HiggsAudioV3LocalTTS": "Higgs Audio V3 Local TTS",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.1.0"

print(f"[ComfyUI-HiggsAudioV3TTS] loaded v{__version__}")
