import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parámetros de la señal
fs = 1000          # Frecuencia de muestreo en Hz
T = 1.0            # Duración de la señal en segundos
n = int(T * fs)    # Número total de muestras

# Generar vector de tiempo
t = np.linspace(0, T, n, endpoint=False)

# Generar señal senoidal de 50 Hz
f0 = 50  # Frecuencia de la senoidal
signal = np.sin(2 * np.pi * f0 * t)

# Calcular la FFT de la señal
fft_values = fft(signal)
freqs = fftfreq(n, 1/fs)

# Tomar la mitad positiva del espectro
positive_freqs = freqs[:n // 2]
fft_magnitude = np.abs(fft_values[:n // 2]) * 2 / n  # Escalado para amplitud

# Graficar la señal en el dominio del tiempo
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Senoidal 50 Hz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal en el dominio temporal')
plt.legend()
plt.grid()

# Graficar el espectro de frecuencia
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, fft_magnitude, label='Espectro de la señal')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Espectro de Frecuencia')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
