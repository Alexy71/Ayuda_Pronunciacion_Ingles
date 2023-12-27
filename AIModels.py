import torch
import numpy as np

import ModelInterfaces
"""reconocimiento automatico del habla usando redes neuronales"""

class NeuralASR(ModelInterfaces.IASRModel):
    word_locations_in_samples = None
    audio_transcript = None

    def __init__(self, model: torch.nn.Module, decoder) -> None:
        super().__init__()
        self.model = model
        self.decoder = decoder  # Decoder from CTC-outputs to transcripts

    #devuelve el audio procesado
    def getTranscript(self) -> str:
        """Get the transcripts of the process audio"""
        assert(self.audio_transcript != None,
               'Can get audio transcripts without having processed the audio')
        return self.audio_transcript

    #devuelve la lista de ubicaciones de palabras en el audio
    def getWordLocations(self) -> list:
        """Get the pair of words location from audio"""
        assert(self.word_locations_in_samples != None,
               'Can get word locations without having processed the audio')

        return self.word_locations_in_samples

    #procesa y genera la transcripcion del audio
    def processAudio(self, audio: torch.Tensor):
        """Process the audio and generate tanscription"""
        audio_length_in_samples = audio.shape[1]
        with torch.inference_mode():
            nn_output = self.model(audio)

            self.audio_transcript, self.word_locations_in_samples = self.decoder(
                nn_output[0, :, :].detach(), audio_length_in_samples, word_align=True)

"""sintesis de voz """
class NeuralTTS(ModelInterfaces.ITextToSpeechModel):
    def __init__(self, model: torch.nn.Module, sampling_rate: int) -> None:
        super().__init__()
        self.model = model
        self.sampling_rate = sampling_rate
    """devuelve el audio generado a partir de una oracion"""
    def getAudioFromSentence(self, sentence: str) -> np.array:
        with torch.inference_mode():
            audio_transcript = self.model.apply_tts(texts=[sentence],
                                                    sample_rate=self.sampling_rate)[0]

        return audio_transcript

