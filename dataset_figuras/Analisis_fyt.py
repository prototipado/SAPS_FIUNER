import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import process_data

# Parámetros clave
fs = 500  # Frecuencia de muestreo real (ajusta esto según tus datos!)
T = 3
N = fs*T

folder="dataset_clases"
# Cargar los datos usando tu función (asumiendo que ya está definida)
x_axis, y_axis, z_axis, classmap =process_data.process_data(fs, T, folder)  # Reemplaza "cargar_datos()" con tu función

# Seleccionar una medición (ej: primera fila) y extraer datos sin la etiqueta
capture_idx = 0  # Índice de la medición a analizar
x_signal = x_axis[capture_idx, :-1]  # Excluir la última columna (etiqueta)
y_signal = y_axis[capture_idx, :-1]
z_signal = z_axis[capture_idx, :-1]



# Preprocesamiento: Eliminar DC y aplicar ventana de Hanning
x_signal = x_signal - np.mean(x_signal)
y_signal = y_signal - np.mean(y_signal)
z_signal = z_signal - np.mean(z_signal)
window = np.hanning(N)
x_signal = x_signal * window
y_signal = y_signal * window
z_signal = z_signal * window

# Calcular FFT
xf = np.linspace(0.0, fs/2, N//2)  # Eje de frecuencias (0 a Nyquist)
yf_x = 2.0/N * np.abs(fft(x_signal))[:N//2]  # <-- Paréntesis cerrado después de fft(x_signal)
yf_y = 2.0/N * np.abs(fft(y_signal))[:N//2]  # <-- Misma corrección aquí
yf_z = 2.0/N * np.abs(fft(z_signal))[:N//2]  # <-- Y aquí

# Graficar
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(xf, yf_x, color='r')
plt.title('Espectro de Frecuencia - Eje X')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')

plt.subplot(3, 1, 2)
plt.plot(xf, yf_y, color='g')
plt.title('Espectro de Frecuencia - Eje Y')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')

plt.subplot(3, 1, 3)
plt.plot(xf, yf_z, color='b')
plt.title('Espectro de Frecuencia - Eje Z')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')

plt.tight_layout()
plt.show()