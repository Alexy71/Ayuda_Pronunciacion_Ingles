import abc
import numpy as np

class IASRModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'getTranscript') and
                callable(subclass.getTranscript) and
                hasattr(subclass, 'getWordLocations') and
                callable(subclass.getWordLocations) and
                hasattr(subclass, 'processAudio') and
                callable(subclass.processAudio))

    @abc.abstractmethod
    def getTranscript(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def getWordLocations(self) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def processAudio(self, audio):
        raise NotImplementedError


class ITranslationModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'translateSentence') and
                callable(subclass.translateSentence))

    """modelo de traduccion de texto"""
    @abc.abstractmethod
    def translateSentence(self, str) -> str:
        raise NotImplementedError


class ITextToSpeechModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'getAudioFromSentence') and
                callable(subclass.getAudioFromSentence))

    """genera un archivo de audio a partir de una oracion"""
    @abc.abstractmethod
    def getAudioFromSentence(self, str) -> np.array:
        raise NotImplementedError


class ITextToPhonemModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'convertToPhonem') and
                callable(subclass.convertToPhonem))
    
    @abc.abstractmethod
    def convertToPhonem(self, str) -> str:
        raise NotImplementedError
