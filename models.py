import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

import pickle

"""reconocimiento automatico de voz"""
def getASRModel(language: str) -> nn.Module:
    if language == 'en':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language='en', device=torch.device('cpu'))
    return (model, decoder)

"""sintesis de voz"""
def getTTSModel(language: str) -> nn.Module:
    if language == 'en':
        speaker = 'lj_16khz'  # 16 kHz
        model = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=language, speaker=speaker)
    else:
        raise ValueError('Language not implemented')
    return model


