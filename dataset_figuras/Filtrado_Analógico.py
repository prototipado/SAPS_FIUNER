# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Analógico:
    En el siguiente script se ejemplifica el proceso de análisis de señales 
    para la definición de requisitos para un filtro antialiasing.
    También se ejemplifica la importación de los resultados del diseño de filtros
    utilizando Analog Filter Wizard y de simulaciones realizadas mediante
    LTSpice.

Autor: Albano Peñalva
Fecha: Febrero 2025

"""

# Librerías
from scipy import signal
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
import process_data
from import_ltspice import import_AC_LTSpice
from import_analogfilterwizard import import_AnalogFilterWizard
from funciones_fft import fft_mag

plt.close('all') # cerrar gráficas anteriores

#%% Lectura del dataset

FS = 500 # Frecuencia de muestre: 500Hz
T = 3    # Tiempo total de cada registro: 2 segundos

folder = 'dataset_clases' # Carpeta donde se almacenan los .csv

x, y, z, classmap = process_data.process_data(FS, T, folder)
print("\r")

ts = 1 / FS                     # tiempo de muestreo
N = FS*T                        # número de muestras en cada regsitro
t = np.linspace(0, N * ts, N)   # vector de tiempos
N_half = N // 2  # Mitad de los puntos (solo positivos)
medicion = 1    # Contador para la graficacion
# Análisis frecuencial de las señales

SENS = 300      # Sensibilidad del ADXL335 [mV/g]
OFF  = 1650     # Offset del ADXL335 [mV]

#%%   FFT
fig, axes = plt.subplots(len(classmap), 3, figsize=(30, 20))
fig.subplots_adjust(hspace=0.5)

#se recorre cada gesto
for gesture_name in classmap:      
                         
    # Se recorre cada renglón de las matrices
    for capture in range(int(len(x))):
        
        # Si en el último elemento se detecta la etiqueta correspondiente
        if (x[capture, N] == gesture_name):
            
            # Cálculo y graficación de la FFT
            
            freq = fft.fftfreq(N, ts)[:N_half]  #Solo frecuencias positivas
            # FFT de la señal en X
            x_f = (fft.fft(x[capture, 0:N]) * SENS) + OFF 
            
            # FFT de la señal en Y
            y_f = (fft.fft(y[capture, 0:N]) * SENS) + OFF 
            
            # FFT de la señal en Z
            # Recorro el renglon correspondiente al valor de capture, desde 0 hasta N-1. Esto me da un subarreglo
            # que se corresponde a la fila de capture y a las primeras N columnas de la matriz.
            # Además ya hago la conversión de [G] a [V]
            z_f = (fft.fft(z[capture, 0:N]) * SENS) + OFF 
            
            # Como la transformada de fourier genera un espectro simetrico, se considera solamente la mitad y graficamos
            #desde 0 hasta nyquist, tambien como consideramos la mitad para mantener la relacion de parseval debemos multiplicar
            #por 2  la amplitud de la FFT, tambien debo dividir por N para normalizar la senial
            
            x_f_mod = (2 / N) * np.abs(x_f[:N_half])  
            y_f_mod = (2 / N) * np.abs(y_f[:N_half])  
            z_f_mod = (2 / N) * np.abs(z_f[:N_half])

# Graficacion de la FFT
            
            axes[gesture_name][0].plot(freq, x_f_mod, label="Medicion {}".format(medicion)) #graficacion eje x
            axes[gesture_name][1].plot(freq, y_f_mod, label="Medicion {}".format(medicion)) #graficacion eje y
            axes[gesture_name][2].plot(freq, z_f_mod, label="Medicion {}".format(medicion))#graficacion eje z 

           
            medicion = medicion+1 #paso a la siguiente medicion
            if medicion==22:
                 medicion=1


     

        axes[gesture_name][0].set_title(classmap[gesture_name] + " (Frecuencia X)")
        axes[gesture_name][0].grid() 
        axes[gesture_name][0].legend(fontsize=8, loc='upper right')
        axes[gesture_name][0].set_xlabel('Frecuencia [Hz]', fontsize=10)
        axes[gesture_name][0].set_ylabel('Magnitud', fontsize=10)

        axes[gesture_name][1].set_title(classmap[gesture_name] + " (Frecuencia Y)")
        axes[gesture_name][1].grid() 
        axes[gesture_name][1].legend(fontsize=8, loc='upper right')
        axes[gesture_name][1].set_xlabel('Frecuencia [Hz]', fontsize=10)
        axes[gesture_name][1].set_ylabel('Magnitud', fontsize=10)

        axes[gesture_name][2].set_title(classmap[gesture_name] + " (Frecuencia Z)")
        axes[gesture_name][2].grid() 
        axes[gesture_name][2].legend(fontsize=8, loc='upper right')
        axes[gesture_name][2].set_xlabel('Frecuencia [Hz]', fontsize=10)
        axes[gesture_name][2].set_ylabel('Magnitud', fontsize=10)

        
            #Se establecen limites en el eje x para observar frecuencias hasta los 20 Hz
        axes[gesture_name][0].set_xlim(0, 90)
        axes[gesture_name][1].set_xlim(0, 90)
        axes[gesture_name][2].set_xlim(0, 90)

#Se muestra el gráfico
plt.tight_layout()
plt.show()


#%%
# Se propone una nueva frecuencia de muestreo para el sistema
FS2 = 90                                    # Nueva frecuencia de muestreo: 60Hz
fs2_2 = freq[np.where(freq>=(FS2/2))][0]          # Valor más cercano a FS2/2

# Para determinar los requerimientos del antialiasing, primero analizamos el 
# contenido espectral de las señales por encima de FS2/2 en dos puntos (peores casos):
h_max_max=0 
f_max_max=0 
#se recorre cada gesto
for gesture_name in classmap:      
                         
    # Se recorre cada renglón de las matrices
    for capture in range(int(len(x))):
        
        # Si en el último elemento se detecta la etiqueta correspondiente
        if (x[capture, N] == gesture_name):    
            # Donde se encuentre el máximo a partir de FS2/2
            hx_max = np.max( x_f_mod [np.where(x_f_mod >= fs2_2) ] )
            hy_max = np.max( y_f_mod [np.where(y_f_mod >= fs2_2) ] )
            hz_max = np.max( z_f_mod [np.where(z_f_mod >= fs2_2) ] )

            fx_max = freq[ np.argmax(x_f_mod[np.where(freq >= fs2_2)])] + fs2_2
            fy_max = freq[ np.argmax(y_f_mod[np.where(freq >= fs2_2)])] + fs2_2
            fz_max = freq[ np.argmax(z_f_mod[np.where(freq >= fs2_2)])] + fs2_2
            
            if(hx_max>=h_max_max):
                h_max_max=hx_max
                f_max_max=fx_max
            if(hy_max>=h_max_max):
                h_max_max=hy_max
                f_max_max=fy_max
            if(hz_max>=h_max_max):
                h_max_max=hz_max
                f_max_max=fz_max
# Exactamente en FS2/2
    h_fs_2 = np.max( x_f_mod [np.where(freq == fs2_2) ] )
    f_fs_2 = freq[ np.argmax( h[ np.where(freq == fs2_2) ] ) ] + fs2_2

print(f"Interferencia de {h_max:.2f}mV en {f_max}Hz")
print(f"Interferencia de {h_fs_2:.2f}mV en {f_fs_2}Hz")
print("\r")

ax2.axvline(x=FS2/2, color="black", linestyle="--")
ax2.plot(f_max, h_max, marker='X', markersize=12, label='Amplitud en fs/2')
ax2.plot(f_fs_2, h_fs_2, marker='X', markersize=12, label='Máximo a partir de fs/2')
ax2.legend(loc='upper right');

#%% Determinar requerimienos del filtro antialiasing

# Parámetros ADC:
V_REF = 3300        # Tensión de referencia en mV
N_BITS = 12         # Resolución en bits

RES = V_REF/(2**N_BITS - 1)     # Resolución en mV

# Atenuaciones necesarias 
at_max = 20*np.log10(h_max/RES) 
at_fs2_2 = 20*np.log10(h_fs_2/RES)

print("Banda de paso hasta 25Hz")   # Banda de paso determinada en guía 1
print(f"Atenuación mayor a {at_max:.2f}dB en {f_max}Hz")
print(f"Atenuación mayor a {at_fs2_2:.2f}dB en {f_fs_2}Hz")
print("\r")

#%% Importar resultados de Diseño de Analog Filter Wizard
f, mag = import_AnalogFilterWizard('DesignFiles/Data Files/Magnitude(dB).csv')


#%% Importar resultados de simulación en LTSpice
f_sim, mag_sim, _ = import_AC_LTSpice('DesignFiles/SPICE Files/LTspice/ACAnalysis.txt')

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
ax3.plot(f_fs_2, -at_fs2_2, marker='X', markersize=12, label='Requisito en fs/2')
ax3.plot(f_max, -at_max, marker='X', markersize=12, label='Requisito en máximo a partir de fs/2')
ax3.legend(loc="lower left", fontsize=15)

#%% Comparación con la respuestas en frecuencia del filtro implementado

f_impl = [1, 2, 5, 10, 11, 12, 15, 20, 21, 22, 25, 50, 100, 200, 500]
mag_impl = [0.0, 0.5, 1.5, 2.0, 1.5, 1.0, 0.0, 1.5, 2.0, 1.5, 0.0, -30, -60, -80, -90]
ax3.plot(f_impl,  mag_impl, label='Implementado')

plt.show()