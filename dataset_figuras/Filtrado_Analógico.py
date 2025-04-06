# -*- coding: utf-8 -*-

"""
Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Analógico:
    Ejemplo de análisis para diseño de filtro antialiasing.
    Incluye importación de diseños de Analog Filter Wizard y simulaciones de LTSpice.

Autor: Albano Peñalva
Fecha: Febrero 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
import process_data
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard

plt.close('all')

#%% Configuración inicial y lectura de datos


FS = 500  # Frecuencia de muestreo original [Hz]
T = 3     # Duración de cada registro [s]
N = FS * T   #cantidad de muestras
ts = 1 / FS  #tiempo entre muestras
t = np.linspace(0, N*ts, N) #vector de muestras para poder representar la senial en temporal
N_half = N // 2  #mitad de la cantidad de muestras
freq = fft.fftfreq(N, ts)[:N_half]  #vector de frecuencias

# Parámetros del sensor
SENS = 300  # Sensibilidad [mV/g]
OFF = 1650   # Offset [mV]

# Lectura de datos
folder = 'dataset_clases'
x, y, z, classmap = process_data.process_data(FS, T, folder)  #se cargan las seniales
n_gestures = len(classmap) #cantidad de gestos

#%% Preprocesamiento y cálculo de FFTs

# Estructura para almacenar todas las FFTs
fft_data = {  #diccionario de diccionarios donde se van a almacenar los valores de la fft
    gesture: {'x': [], 'y': [], 'z': []}  #clave-valor clave numero de gesto, valor  diccionario con 3 claves x,y,z. para el gesto0 se crea una lista para el eje x que contiene vectores con la fft para la primer medicion, segunda y asi 
    for gesture in range(n_gestures)     #range genera numeros desde 0 hasta N-1 gestos 
}

# Cálculo de todas las FFTs
for capture in range(len(x)):
    gesture = int(x[capture, N])  #la última columna tiene la etiqueta del gesto
    
    # Cálculo de FFTs y conversion de G a V
    x_fft = (2/N) * np.abs(fft.fft((x[capture, :N]) * SENS + OFF))[:N_half]
    y_fft = (2/N) * np.abs(fft.fft((y[capture, :N]) * SENS + OFF))[:N_half]
    z_fft = (2/N) * np.abs(fft.fft((z[capture, :N]) * SENS + OFF))[:N_half]
    
    # Almacenamiento
    fft_data[gesture]['x'].append(x_fft) #gesture lo saco de la ultima columna de la medicion
    fft_data[gesture]['y'].append(y_fft)
    fft_data[gesture]['z'].append(z_fft)

#%% Graficación de FFTs

fig, axes = plt.subplots(n_gestures, 3, figsize=(30, 20))
fig.subplots_adjust(hspace=0.5)

for gesture in range(n_gestures):
    # Eje X
    for fft_x in fft_data[gesture]['x']:
        axes[gesture, 0].plot(freq, fft_x, alpha=0.5, linewidth=0.8)
    axes[gesture, 0].set_title(f"{classmap[gesture]} - Eje X", fontsize=10)
    axes[gesture, 0].set_xlim(0, 90)
    axes[gesture, 0].grid(True, alpha=0.3)
    axes[gesture, 0].set_ylabel('Amplitud [mV]', fontsize=8)

    # Eje Y
    for fft_y in fft_data[gesture]['y']:
        axes[gesture, 1].plot(freq, fft_y, alpha=0.5, linewidth=0.8)
    axes[gesture, 1].set_title(f"{classmap[gesture]} - Eje Y", fontsize=10)
    axes[gesture, 1].set_xlim(0, 90)
    axes[gesture, 1].grid(True, alpha=0.3)

    # Eje Z
    for fft_z in fft_data[gesture]['z']:
        axes[gesture, 2].plot(freq, fft_z, alpha=0.5, linewidth=0.8)
    axes[gesture, 2].set_title(f"{classmap[gesture]} - Eje Z", fontsize=10)
    axes[gesture, 2].set_xlim(0, 90)
    axes[gesture, 2].grid(True, alpha=0.3)
    axes[gesture, 2].set_xlabel('Frecuencia [Hz]', fontsize=8)

plt.tight_layout()
plt.show()

#%% Análisis para filtro antialiasing
AB=20
FS2 = AB * 3  # Nueva frecuencia de muestreo
nyquist_nuevo = FS2 / 2
idx_nyquist = np.abs(freq - nyquist_nuevo).argmin()

# Matrices de resultados
resultados = {
    'max_amplitudes': np.zeros((n_gestures, 3)),
    'frecuencias_max': np.zeros((n_gestures, 3)),
    'amplitudes_nyquist': np.zeros((n_gestures, 3))
}

for gesture in range(n_gestures):
    for i, axis in enumerate(['x', 'y', 'z']):
        # Inicializar valores
        max_amp = 0
        freq_max = 0
        amp_nyquist = 0
        
        # Analizar cada captura
        for fft_axis in fft_data[gesture][axis]:
            # Máxima amplitud sobre nyquist_nuevo
            mascara = freq >= nyquist_nuevo
            if np.any(mascara):
                amp_actual = np.max(fft_axis[mascara])
                if amp_actual > max_amp:
                    max_amp = amp_actual
                    freq_max = freq[mascara][np.argmax(fft_axis[mascara])]
            
            # Amplitud en nyquist_nuevo
            amp_actual_nyq = fft_axis[idx_nyquist]
            if amp_actual_nyq > amp_nyquist:
                amp_nyquist = amp_actual_nyq
        
        # Almacenar resultados
        resultados['max_amplitudes'][gesture, i] = max_amp
        resultados['frecuencias_max'][gesture, i] = freq_max
        resultados['amplitudes_nyquist'][gesture, i] = amp_nyquist

#%% Visualización de resultados
print("\nResultados del análisis para filtrado:")
for gesture in range(n_gestures):
    print(f"\nGesto: {classmap[gesture]}")
    print("Máximas amplitudes sobre {} Hz:".format(nyquist_nuevo))
    print("X: {:.2f} mV @ {:.1f} Hz".format(
        resultados['max_amplitudes'][gesture, 0],
        resultados['frecuencias_max'][gesture, 0]))
    print("Y: {:.2f} mV @ {:.1f} Hz".format(
        resultados['max_amplitudes'][gesture, 1],
        resultados['frecuencias_max'][gesture, 1]))
    print("Z: {:.2f} mV @ {:.1f} Hz".format(
        resultados['max_amplitudes'][gesture, 2],
        resultados['frecuencias_max'][gesture, 2]))
    
    print("\nAmplitudes en {} Hz:".format(nyquist_nuevo))
    print("X: {:.2f} mV".format(resultados['amplitudes_nyquist'][gesture, 0]))
    print("Y: {:.2f} mV".format(resultados['amplitudes_nyquist'][gesture, 1]))
    print("Z: {:.2f} mV\n".format(resultados['amplitudes_nyquist'][gesture, 2]))

#%% Parámetros del ADC y cálculos de atenuación
V_REF = 3300        # Tensión de referencia en mV
N_BITS = 12         # Resolución en bits
RES = V_REF / (2**N_BITS - 1)  # Resolución en mV

# Encontrar los peores casos de atenuación requerida
max_at = {
    'at_max': {'value': -np.inf, 'freq': 0, 'gesture': None, 'axis': None},
    'at_fs2': {'value': -np.inf, 'freq': 0, 'gesture': None, 'axis': None}
}

for gesture in range(n_gestures):
    for i, axis in enumerate(['x', 'y', 'z']):
        # Calcular atenuaciones requeridas
        h_max = resultados['max_amplitudes'][gesture, i]
        h_fs2 = resultados['amplitudes_nyquist'][gesture, i]
        f_max = resultados['frecuencias_max'][gesture, i]
        
        at_max = 20 * np.log10(h_max / RES) if h_max > 0 else -np.inf
        at_fs2 = 20 * np.log10(h_fs2 / RES) if h_fs2 > 0 else -np.inf
        
        # Actualizar peores casos
        if at_max > max_at['at_max']['value']:
            max_at['at_max'].update({
                'value': at_max,
                'freq': f_max,
                'gesture': classmap[gesture],
                'axis': axis.upper()
            })
            
        if at_fs2 > max_at['at_fs2']['value']:
            max_at['at_fs2'].update({
                'value': at_fs2,
                'freq': FS2/2,
                'gesture': classmap[gesture],
                'axis': axis.upper()
            })

# Mostrar requisitos
print("\nRequisitos de atenuación (peores casos):")
print(f"Banda de paso hasta {FS2/3} Hz")
print(f"Atenuación requerida en {max_at['at_max']['freq']:.1f} Hz: " +
      f"{max_at['at_max']['value']:.2f} dB " +
      f"(Gesto: {max_at['at_max']['gesture']}, Eje: {max_at['at_max']['axis']})")
      
print(f"Atenuación requerida en {FS2/2} Hz: " +
      f"{max_at['at_fs2']['value']:.2f} dB " +
      f"(Gesto: {max_at['at_fs2']['gesture']}, Eje: {max_at['at_fs2']['axis']})")

#%% Importación de respuestas de filtros
# Diseño de Analog Filter Wizard
f_diseño, mag_diseño = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')

# Simulación LTSpice
f_sim, mag_sim, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')

# Implementación real (datos de ejemplo)
f_impl = [1, 2, 5, 10, 11, 12, 15, 20, 21, 22, 25, 50, 100, 200, 500]
mag_impl = [0.0, 0.5, 1.5, 2.0, 1.5, 1.0, 0.0, 1.5, 2.0, 1.5, 0.0, -30, -60, -80, -90]

#%% Análisis comparativo de filtros
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_title('Comparación de respuestas de filtro', fontsize=14)
ax.set_xlabel('Frecuencia [Hz]', fontsize=12)
ax.set_ylabel('Atenuación [dB]', fontsize=12)
ax.set_xscale('log')
ax.grid(True, which='both', linestyle='--', alpha=0.6)

# Graficar todas las respuestas
ax.plot(f_diseño, mag_diseño, label='Diseño teórico', linewidth=2)
ax.plot(f_sim, mag_sim, label='Simulación LTSpice', linestyle='--')
ax.plot(f_impl, mag_impl, label='Implementación real', marker='o', markersize=6)

# Marcar requisitos
ax.axvline(FS2/2, color='purple', linestyle=':', 
          label=f'Nyquist nuevo ({FS2/2} Hz)')
ax.plot(max_at['at_max']['freq'], -max_at['at_max']['value'], 'rx', 
       markersize=10, label='Requisito máximo')
ax.plot(FS2/2, -max_at['at_fs2']['value'], 'r+', markersize=15, 
       label='Requisito en Nyquist')

ax.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.show()

#%% Análisis de margen de seguridad
def encontrar_atenuacion(frecuencias, respuesta, f_obj):
    idx = np.abs(frecuencias - f_obj).argmin()
    return respuesta[idx]

print("\nAtenuación obtenida vs requerida:")
for caso in ['at_max', 'at_fs2']:
    f_obj = max_at[caso]['freq']
    req = -max_at[caso]['value']
    
    at_diseño = encontrar_atenuacion(f_diseño, mag_diseño, f_obj)
    at_sim = encontrar_atenuacion(f_sim, mag_sim, f_obj)
    at_impl = encontrar_atenuacion(f_impl, mag_impl, f_obj)
    
    print(f"\nEn {f_obj:.1f} Hz:")
    print(f"- Requerido: {req:.2f} dB")
    print(f"- Diseño: {at_diseño:.2f} dB | Margen: {at_diseño - req:.2f} dB")
    print(f"- Simulación: {at_sim:.2f} dB | Margen: {at_sim - req:.2f} dB")
    print(f"- Implementación: {at_impl:.2f} dB | Margen: {at_impl - req:.2f} dB")