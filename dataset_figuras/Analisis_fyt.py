import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import process_data

fs = 500  #frecuencia de muestreo del sensor
ts=1/fs #tiempo entre muestras
T = 3 #tiempo de muestreo
N = fs*T #numero de muestras en la medicion
SENS = 0.3 #sensibilidad
OFF = 1.65 #offset
N_half = N // 2  # Mitad de los puntos (solo positivos)
medicion = 1


folder="dataset_clases" #nombre del directorio donde estan los datasets

# Carga de datos
x, y, z, classmap =process_data.process_data(fs, T, folder)
fig, axes = plt.subplots(len(classmap), 3, figsize=(30, 20))
fig.subplots_adjust(hspace=0.5)

#%%   FFT

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

# Graficacion
            
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

#%% Filtros

#a partir de ver las graficas puedo determinar que el ancho de banda de interes

Ab=30

#defino la fm de la banda de interes como 4 veces la AB

fm_interes = 4 * Ab

#defino la frecuencia de corte como la fm de la banda de interes