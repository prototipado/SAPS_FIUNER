# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Funciones para la importación de los resultados de los análisis realizados 
utilizando el software LTSpice.

Autor: Albano Peñalva
Fecha: Mayo 2020

"""

# %% Librerías
from scipy import signal
import numpy as np

# %% 
def import_AC_LTSpice(filename):
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
    
    freq = np.array([])
    dB = np.array([])
    deg = np.array([])
    
    with open(filename, 'r') as fp:
        campos = fp.readline()
        
        STOP = False
        
        while not STOP:
            linea = fp.readline()
            
            if linea:
                linea = linea.replace(',', ' ')
                linea = linea.replace('\t', ' ')
                linea = linea.replace('(', '')
                linea = linea.replace(')', '')
                linea = linea.replace('dB', '')
                linea = linea.replace('°', '')
                
                valores = linea.split()
                
                freq = np.append(freq, valores[0]).astype(float)
                dB = np.append(dB, valores[1]).astype(float)
                deg = np.append(deg, valores[2]).astype(float)
                
            else:
                STOP = True
    
    return freq, dB, deg

# %% 
def import_Transient_LTSpice(filename):
    """
    ------------------------
    INPUT:
    --------
    filename: string conteniendo el nombre del archivo exportado de LTSpice. 
    ------------------------
    OUTPUT:
    --------
    t: array de una dimensión conteniendo los valores del eje temporal.
    V: array de una dimensión conteniendo los valores de tensión de la señal.
    """
    
    t = np.array([])
    V = np.array([])
        
    with open(filename, 'r') as fp:
        campos = fp.readline()
        
        STOP = False
        
        while not STOP:
            linea = fp.readline()
            
            if linea:
                linea = linea.replace('\t', ' ')
                
                valores = linea.split()
                
                t = np.append(t, valores[0]).astype(float)
                V = np.append(V, valores[1]).astype(float)
                
            else:
                STOP = True
                
    return t, V

