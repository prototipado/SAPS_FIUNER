# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Funciones para la importación de los resultados del diseño realizado utilizando 
la herramienta Analog Filter Wizard (https://tools.analog.com/en/filterwizard/).

Autor: Albano Peñalva
Fecha: Febrero 2025

"""

# %% Librerías
from scipy import signal
import numpy as np

# %% 
def import_AnalogFilterWizard(filename):
    """
    ------------------------
    INPUT:
    --------
    filename: string conteniendo el nombre del archivo exportado de LTSpice.
    ------------------------
    OUTPUT:
    --------
    freq: array de una dimensión conteniendo los valores del eje de frecuencia.
    dB: array de una dimensión conteniendo la magnitud de la respuesta en 
    frecuencia en dB.
    deg: array de una dimensión conteniendo la fase de la respuesta en 
    frecuencia en grados.
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data[:, 0], data[:, 1]