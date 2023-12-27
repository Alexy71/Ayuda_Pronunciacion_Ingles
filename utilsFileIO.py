import string 
import random 

"""Genera una cadena aleatoria de 20 caracteres"""
def generateRandomString(str_length: int = 20):

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(str_length))