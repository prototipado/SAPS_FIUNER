# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Módulo que contiene funciones para el cálculo de la FFT, pensado para 
ejemplificar el uso de funciones en Python.

Autor: Albano Peñalva
Fecha: Abril 2020

"""

# Librerías
import numpy as np
from scipy import fft


def fft_mag(x, fs):
    """
    ------------------------
    INPUT:
    --------
    x: array de una dimensión conteniendo la señal cuya fft se busca calcular
    fs: frecuncia a la que está muestreada la señal
    ------------------------
    OUTPUT:
    --------
    f: array de una dimension con con los valores correspondientes al eje de 
    frecuencias de la fft.
    mag: array de una dimensión conteniendo los valores en magnitud de la fft
    de la señal.    
    """
    freq = fft.fftfreq(len(x), d=1/fs)   # se genera el vector de frecuencias
    senial_fft = fft.fft(x)    # se calcula la transformada rápida de Fourier

    # El espectro es simétrico, nos quedamos solo con el semieje positivo
    f = freq[np.where(freq >= 0)]      
    senial_fft = senial_fft[np.where(freq >= 0)]

    # Se calcula la magnitud del espectro
    mag = np.abs(senial_fft) / len(x)    # Respetando la relación de Parceval
    # Al haberse descartado la mitad del espectro, para conservar la energía 
    # original de la señal, se debe multiplicar la mitad restante por dos (excepto
    # en 0 y fm/2)
    mag[1:len(mag)-1] = 2 * mag[1:len(mag)-1]
    
    return f, mag

def fft_pot(x, fs):
    """
    ------------------------
    INPUT:
    --------
    x: array de una dimensión conteniendo la señal cuya fft se busca calcular
    fs: frecuncia a la que está muestreada la señal
    ------------------------
    OUTPUT:
    --------
    f: array de una dimension con con los valores correspondientes al eje de 
    frecuencias de la fft.
    pot: array de una dimensión conteniendo los valores en potencia de la fft
    de la señal.    
    """
    
    # Se calcula la magnitud de la fft de la señal.
    f, senial_fft_mod = fft_mag(x, fs)
    
    # Se calcula el espectro en potencia
    pot = senial_fft_mod ** 2
    
    return f, pot

