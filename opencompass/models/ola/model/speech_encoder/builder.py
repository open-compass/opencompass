from .speech_encoder import WhisperWrappedEncoder, DualWrappedEncoder
import torch.nn as nn

def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)
    elif "dual" in speech_encoder_type.lower():
        return DualWrappedEncoder(config)
    elif "none" in speech_encoder_type.lower():
        return None
    
    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')
