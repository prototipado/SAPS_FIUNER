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
from matplotlib import patches
import funciones_fft
from time import time
import sympy as sy
import filter_parameters
import process_code


plt.close('all') #ciero plots si hay abierto alguno

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
    for fft_x in fft_data[gesture]['x']:  #aqui es donde se itera sobre las mediciones en el ejex dando las diferentes capturas
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
AB=20 #definimos nuestro ancho de banda segun las graficas
FS2 = AB * 3  # Nueva frecuencia de muestreo, lo multiplicamos por 3 y no por 2 porque en la vida real los filtros no son ideales y multiplicarlo por 3 me da una tolerancia
nyquist_nuevo = FS2 / 2 
idx_nyquist = np.abs(freq - nyquist_nuevo).argmin() #se encuentra el valor de nyquist mas cercano en el vector de freq
                                                    #primero se calculan las diferencias, se le aplica valor absoluto y luego se encuentra el minimo de las diferencias 
                                                    #esto es un indice que posteriormente lo uso para obtener la amplitud en la freq nyquist
# Matrices de resultados
resultados = {  #diccionario de matrices 
    'max_amplitudes': np.zeros((n_gestures, 3)),
    'frecuencias_max': np.zeros((n_gestures, 3)),
    'amplitudes_nyquist': np.zeros((n_gestures, 3))
}

for gesture in range(n_gestures):
    for i, axis in enumerate(['x', 'y', 'z']):  #este bucle devuelve pares indices y elemento entonces la primera iteracion es x, la segunda y y la tercera z
       
        # Inicializar valores
        max_amp = 0
        freq_max = 0
        amp_nyquist = 0
        
        # Analizar cada captura
        for fft_axis in fft_data[gesture][axis]:  #se recorren los valores de la fft para cada eje dentro de cada gesto
            # se busca la máxima amplitud sobre nyquist_nuevo
            mascara = freq >= nyquist_nuevo #array booleano que devuelve positivo si la freq esta por encima de FS2_2
            if np.any(mascara): #se verifica si existe alguna frecuencia sobre el nyquist
                amp_actual = np.max(fft_axis[mascara]) #encuentra el valor maximo de amplitud en los valores donde la mascara es verdadera
                if amp_actual > max_amp: #si el maximo de la captura actual es mayor que la maxima amplitud registrada hasta el momento
                    max_amp = amp_actual #se asigna el maximo actual como el nuevo maximo de todas las capturas
                    freq_max = freq[mascara][np.argmax(fft_axis[mascara])] #se encuentra la maxima frecuencia para la mayor amplitud
                                                                            #np.argmax busca el valor de maxima amplitud en fft_axis y lo devuelve como indice
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
max_at = {   #diccionario
    'at_max': {'value': -np.inf, 'freq': 0, 'gesture': None, 'axis': None}, #value -np inf dice que el valor inicial es infinito negativo para asegurar que cualquier valor real lo supere 
    'at_fs2': {'value': -np.inf, 'freq': 0, 'gesture': None, 'axis': None}#freq, gesture y axis estan inicializados en none para ser remplazados posteriormente 
}

for gesture in range(n_gestures):
    for i, axis in enumerate(['x', 'y', 'z']):
        # Calcular atenuaciones requeridas
        h_max = resultados['max_amplitudes'][gesture, i] 
        h_fs2 = resultados['amplitudes_nyquist'][gesture, i]
        f_max = resultados['frecuencias_max'][gesture, i]
        
      
        at_max = 20 * np.log10(h_max / RES) if h_max > 0 else -np.inf #primero pasamos a db
        at_fs2 = 20 * np.log10(h_fs2 / RES) if h_fs2 > 0 else -np.inf #despues comparamos si las amplitudes son mayores a 0 para evitar errores matematicos si se da el caso de log entre 0 y 1
        
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

#%% Importar resultados de Diseño de Analog Filter Wizard
f, mag = import_AnalogFilterWizard('C:/Repositorio/SAPS_Gonzalez_C1_2025/DesignFiles/Data Files/Magnitude(dB).csv')


#%% Importar resultados de simulación en LTSpice
f_sim, mag_sim, _ = import_AC_LTSpice('C:/Repositorio/SAPS_Gonzalez_C1_2025/DesignFiles/Data Files/filtro.txt')

# Análisis de la atenuación del filtro simulado en las frecuencias de interés
F_AT1 = FS2/2
F_AT2 = f_max
# se calcula la atenuación en el punto mas cercano a la frecuencia de interés
at1 = mag_sim[np.argmin(np.abs(f_sim-F_AT1))] 
print("La atenuación del filtro simulado en {}Hz es de {:.2f}dB".format(F_AT1, at1))
at2 = mag_sim[np.argmin(np.abs(f_sim-F_AT2))] 
print("La atenuación del filtro simulado en {}Hz es de {:.2f}dB".format(F_AT2, at2))
print("\r")

#%% Comparación de las respuestas en frecuencia del filtro diseñado y el simulado 

# Se crea una gráfica para comparar los filtros 
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))

ax3.set_title('Filtro orden 4', fontsize=18)
ax3.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3.set_xscale('log')
ax3.grid(True, which="both")
ax3.plot(f,  mag, label='Diseñado')
ax3.plot(f_sim,  mag_sim, label='Simulado')
ax3.plot(FS2/2, -at_fs2, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(f_max, -at_max, marker='X', markersize=12, label='Requisito en máximo a partir de fs/2')
ax3.legend(loc="lower left", fontsize=15)

#%% Comparación con la respuestas en frecuencia del filtro implementado



f_impl = [1, 5, 10, 12, 14, 16, 18, 20,22, 26, 30, 35, 40, 45,50,70,100,120,150,200]
mag_impl = [0.09001002453,
-0.9554556359,
-3.658613672,
-4.838170871,
-6.390268027,
-8.049752728,
-9.818190784,
-12.52490752,
-14.11618858,
-17.36655761,
-20,
-23.58995015,
-26.58117439,
-29.27514586,
-31.86210297,
-39.64542466,
-48.76406377,
-52.87629479,
-56.97464649,
-62.49877473]
ax3.plot(f_impl,  mag_impl, label='Implementado')

plt.show()

#%% filtro digital

# Parámetros para el remuestreo de las señales
FS_resample = 60                        # Frecuencia de muestreo para la cual estan diseñados los filtros
N_resample = int(T*FS_resample)         # Longitud de las señales remuestreadas

# Se cargan los archivo generado mediante pyFDA
#filtro_fir = np.load('filtro_fir.npz', allow_pickle=True)
filtro_iir = np.load('4-9pb.npz', allow_pickle=True) 

# Se muestran parámetros de diseño
#print("Filtro FIR:")
#filter_parameters.filter_parameters('filtro_fir.npz')
#print("\r\n")
print("Filtro IIR:")
filter_parameters.filter_parameters('filtro_iir.npz')
print("\r\n")

# Se extraen los coeficientes de numerador y denominador
#Num_fir, Den_fir = filtro_fir['ba']     
Num_iir, Den_iir = filtro_iir['ba'] 

# Se expresan las funciones de transferencias (H(z))
z = sy.Symbol('z') # Se crea una variable simbólica z
Hz = sy.Symbol('H(z)')

# Numz_fir = 0
# Denz_fir = 0
# for i in range(len(Num_fir)): # Se arma el polinomio del numerador
#     Numz_fir += Num_fir[i] * np.power(z, -i)
# for i in range(len(Den_iir)): # Se arma el polinomio del denominador
#     Denz_fir += Den_fir[i] * np.power(z, -i)
# print("La función de transferencia del Filtro IIR es:")
# print(sy.pretty(sy.Eq(Hz, Numz_fir.evalf(3) / Denz_fir.evalf(3))))
# print("\r\n")

Numz_iir = 0
Denz_iir = 0
for i in range(len(Num_iir)): # Se arma el polinomio del numerador
    Numz_iir += Num_iir[i] * np.power(z, -i)
for i in range(len(Den_iir)): # Se arma el polinomio del denominador
    Denz_iir += Den_iir[i] * np.power(z, -i)
print("La función de transferencia del Filtro IIR es:")
print(sy.pretty(sy.Eq(Hz, Numz_iir.evalf(3) / Denz_iir.evalf(3)))) 
print("\r\n")

#%% Análisis de los Filtros 

# Se calcula la respuesta en frecuencia de los filtros
#f_fir, h_fir = signal.freqz(Num_fir, Den_fir, worN=f, fs=FS_resample)
f_iir, h_iir = signal.freqz(Num_iir, Den_iir, worN=f, fs=FS_resample)

# Se crea una gráfica 
fig2, ax2 = plt.subplots(2, 1, figsize=(15, 15), sharex=True)
fig2.suptitle("Señal de aceleración", fontsize=18)

# Se grafican las respuestas de los filtros
#ax2[0].plot(f_fir, abs(h_fir), label='Filtro FIR', color='orange')
#ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f_iir, abs(h_iir), label='Filtro IIR', color='green')
ax2[1].legend(loc="upper right", fontsize=15)

# Se evalúa la atenuación en las frecuncias de interés 
#_, h1_fir = signal.freqz(Num_fir, Den_fir, worN=[0.01, 20], fs=FS_resample)
_, h1_iir = signal.freqz(Num_iir, Den_iir, worN=[0.01, 20], fs=FS_resample)

#print("La atenuación del filtro FIR en 0.01Hz es de {:.2f}dB".format(20*np.log10(abs(h1_fir[0]))))
#print("La atenuación del filtro FIR en 20Hz es de {:.2f}dB".format(20*np.log10(abs(h1_fir[1]))))
print("La atenuación del filtro IIR en 0.01Hz es de {:.2f}dB".format(20*np.log10(abs(h1_iir[0]))))
print("La atenuación del filtro IIR en 20Hz es de {:.2f}dB".format(20*np.log10(abs(h1_iir[1]))))
print("\r\n")

# Se extraen polos y ceros de los filtros
#zeros_fir, polos_fir, k_fir =   filtro_fir['zpk']  
zeros_iir, polos_iir, k_iir =   filtro_iir['zpk']

# Se grafican las distribuciones de ceros y polos
fig3, ax3 = plt.subplots(1, 2, figsize=(15, 7))
fig3.suptitle("Distribución de Ceros y Polos en el plano Z", fontsize=18)

#ax3[0].set_title('Filtro FIR', fontsize=15)
#ax3[0].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
#ax3[0].plot(polos_fir.real, polos_fir.imag, 'x', label='Polos', color='red',
#            markersize=10, alpha=0.5)
#ax3[0].plot(zeros_fir.real, zeros_fir.imag, 'o', label='Ceros', color='none',
 #           markersize=10, alpha=0.5, markeredgecolor='blue')
#lim = 1.2 * np.max([np.max(abs(polos_fir)), np.max(abs(zeros_fir))])
#ax3[0].set_xlim(-lim, lim)
#ax3[0].set_ylim(-lim, lim)
#ax3[0].set_ylabel('Imag(z)', fontsize=15)
#ax3[0].set_xlabel('Real(z)', fontsize=15)
#ax3[0].grid()
#ax3[0].legend(loc="upper right", fontsize=12)

ax3[1].set_title('Filto IIR', fontsize=15)
ax3[1].add_patch(patches.Circle((0,0), radius=1, fill=False, alpha=0.1))
ax3[1].plot(polos_iir.real, polos_iir.imag, 'x', label='Polos', color='red',
            markersize=10, alpha=0.5)
ax3[1].plot(zeros_iir.real, zeros_iir.imag, 'o', label='Ceros', color='none',
            markersize=10, alpha=0.5, markeredgecolor='blue')
lim = 1.2 * np.max([np.max(abs(polos_iir)), np.max(abs(zeros_iir))])
ax3[1].set_xlim(-lim, lim)
ax3[1].set_ylim(-lim, lim)
ax3[1].set_ylabel('Imag(z)', fontsize=15)
ax3[1].set_xlabel('Real(z)', fontsize=15)
ax3[1].grid()
ax3[1].legend(loc="upper right", fontsize=12)

#%% Filtrado de la Señal 

# Se aplica el filtrado sobre la señal
#senial_fir = signal.lfilter(Num_fir, Den_fir, senial)
senial_iir = signal.lfilter(Num_iir, Den_iir, senial)

# Se grafican las señales filtradas
ax1[0].plot(t_resampled, senial_fir, label='Señal Filtrada (FIR)', color='red')
ax1[0].legend(loc="upper right", fontsize=15)
ax1[1].plot(t_resampled, senial_iir, label='Señal Filtrada (IIR)', color='purple')
ax1[1].legend(loc="upper right", fontsize=15)

# Se calculan y grafican sus espectros (normalizados)
f1_fir, senial_fir_fft_mod = funciones_fft.fft_mag(senial_fir, FS_resample)
f1_iir, senial_iir_fft_mod = funciones_fft.fft_mag(senial_iir, FS_resample)
ax2[0].plot(f1_fir, senial_fir_fft_mod/np.max(senial_fft_mod), 
            label='Senial Filtrada FIR', color='red')
ax2[0].legend(loc="upper right", fontsize=15)
ax2[1].plot(f1_iir, senial_iir_fft_mod/np.max(senial_fft_mod), 
            label='Senial Filtrada IIR', color='purple')
ax2[1].legend(loc="upper right", fontsize=15)
plt.show()

#%% Evaluar Performance del Filtro 

# Se aplica el filtrado sobre la señal 500 veces y se mide el tiempo requerido
# por el algoritmo
t_start_fir = time()
for i in range(500):
    senial_fir = signal.lfilter(Num_fir, Den_fir, senial)
t_end_fir = time()
t_start_iir = time()
for i in range(500):
    senial_iir = signal.lfilter(Num_iir, Den_iir, senial)
t_end_iir = time()

print("El algoritmo de filtrado FIR toma {:.3f}s".format(t_end_fir - t_start_fir))
print("El algoritmo de filtrado IIR toma {:.3f}s".format(t_end_iir - t_start_iir))













