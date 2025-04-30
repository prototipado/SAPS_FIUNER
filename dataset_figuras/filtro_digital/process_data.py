# -*- coding: utf-8 -*-
"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Módulo que contiene funciones para el cálculo de la FFT, pensado para 
ejemplificar el uso de funciones en Python.

Autor: Juani Cerrudo
Fecha: Marzo 2022

"""


import numpy as np
from glob import glob
from os.path import basename

def process_data(fs, tiempo, folder):
    """
    A partir de los archivos almacenados en formato .csv, en el directorio indicado, 
    devuelve tres matrices con los datos de aceleración registrados (una por eje).
    Cada renglón de la matriz contiene los valores de aceleración para un gesto 
    (con una duración igual a 'tiempo'), y en el último elemento una etiqueta 
    identificando el gesto correspondiente (detallados en 'classmap').
    ------------------------
    INPUT:
    --------
    fs: value
    Frecuencia a la cual fueron muestreadas las señales.
    tiempo: value
    Duración (en segundos) de cada registro.
    folder: string
    Nombre de la carpeta que contiene los .csv con los registros.
    ------------------------
    OUTPUT:
    --------
    x_axis, y_axis, z_axis: array 
    Arreglos de dos dimensiones, conteniendo tantos renglones como cantidad de
    repeticiones de todos los gestos se hayan registrado, y en cada renglón los 
    valores de aceleración correspondientes a cada eje. El último valor de cada
    renglón contiene una etiqueta identificando el gesto.
    classmap: dict
    Diccionario conteniendo las etiquetas que identifican cada gesto. Hay tantas
    etiquetas como archivos .csv.
    """
    
    muestras = fs*tiempo
    canales = 3
    dataset = None
    classmap = {}

    for class_idx, filename in enumerate(glob('%s/*.csv' % folder)):       
        class_name = basename(filename)[:-4]
        print(filename)
        classmap[class_idx] = class_name
        samples = np.loadtxt(filename, dtype=float, delimiter=',')
        labels = np.ones((len(samples), 1)) * class_idx
        print("Se encontraron {} eventos de la clase:  {}".format(len(samples), class_name))
        samples = np.hstack((samples, labels))
        dataset = samples if dataset is None else np.vstack((dataset, samples))
    #dataset = np.delete(dataset,17,0)
    #dataset = np.delete(dataset,46,0)

    x_axis = np.ones((len(dataset), muestras+1))
    y_axis = np.ones((len(dataset), muestras+1))
    z_axis = np.ones((len(dataset), muestras+1))
    jump=0

    for capture in range(int(len(dataset))):
        jump=0
        for sample in range(int(len(dataset[1,:])/3)): 
            x_axis[capture, sample] = dataset[capture,jump]
            y_axis[capture, sample] = dataset[capture,jump+1]
            z_axis[capture, sample] = dataset[capture,jump+2]
            jump=jump+3
            for i in range(len(classmap)):
                if (dataset[capture,int(muestras*canales)] == i):  
                    x_axis[capture, int(muestras)] = i
                    y_axis[capture, int(muestras)] = i
                    z_axis[capture, int(muestras)] = i
    return x_axis, y_axis, z_axis, classmap                