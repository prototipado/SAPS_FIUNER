# -*- coding: utf-8 -*-
"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Script ejemplificando el uso de la función provista para levantar los registros
almacenados en .csv, el manejo de las señales y su graficación. 

Autor: Albano Peñalva
Fecha: Febrero 2025
"""

# Librerías
import process_data
import numpy as np
import matplotlib.pyplot as plt

#%% Lectura del dataset

FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 2 segundos

folder = 'dataset_clases' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)

#%% Graficación

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos

# Se crea un arreglo de gráficas, con tres columnas de gráficas 
# (correspondientes a cada eje) y tantos renglones como gesto distintos.
fig, axes = plt.subplots(len(classmap), 3, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5)

# Se recorren y grafican todos los registros
trial_num = 0
for gesture_name in classmap:                           # Se recorre cada gesto
    for capture in range(int(len(x))):                  # Se recorre cada renglón de las matrices
        if (x[capture, N] == gesture_name):             # Si en el último elemento se detecta la etiqueta correspondiente
            # Se grafica la señal en los tres ejes
            axes[gesture_name][0].plot(t, x[capture, 0:N], label="Trial {}".format(trial_num))
            axes[gesture_name][1].plot(t, y[capture, 0:N], label="Trial {}".format(trial_num))
            axes[gesture_name][2].plot(t, z[capture, 0:N], label="Trial {}".format(trial_num))
            trial_num = trial_num + 1

# Se le da formato a los ejes de cada gráfica
    axes[gesture_name][0].set_title(classmap[gesture_name] + " (Aceleración X)")
    axes[gesture_name][0].grid()
    axes[gesture_name][0].legend(fontsize=6, loc='upper right');
    axes[gesture_name][0].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][0].set_ylabel('Aceleración [G]', fontsize=10)
    axes[gesture_name][0].set_ylim(-6, 6)
    
    axes[gesture_name][1].set_title(classmap[gesture_name] + " (Aceleración Y)")
    axes[gesture_name][1].grid()
    axes[gesture_name][1].legend(fontsize=6, loc='upper right');
    axes[gesture_name][1].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][1].set_ylabel('Aceleración [G]', fontsize=10)
    axes[gesture_name][1].set_ylim(-6, 6)
    
    axes[gesture_name][2].set_title(classmap[gesture_name] + " (Aceleración Z)")
    axes[gesture_name][2].grid()
    axes[gesture_name][2].legend(fontsize=6, loc='upper right');
    axes[gesture_name][2].set_xlabel('Tiempo [s]', fontsize=10)
    axes[gesture_name][2].set_ylabel('Aceleración [G]', fontsize=10)
    axes[gesture_name][2].set_ylim(-6, 6)

plt.tight_layout()
plt.show()
