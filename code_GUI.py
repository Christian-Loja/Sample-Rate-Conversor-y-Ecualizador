import numpy as np
import scipy.fftpack as fourier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as canva
import librosa.display
from scipy.signal import firwin, freqz, spectrogram
from glob import glob
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
from PyQt5.QtCore import QIODevice, QPoint
from PyQt5 import QtWidgets
import sounddevice as sd
import sys

# Se carga el archivo mp3:
audio_file = glob('C:/Users/Christian/Documents/U_Cuenca/Sistemas L y Señales/Trabajo_Muestreo_Ecualizador/ambient-piano.mp3')
# Cargar el archivo de audio en un array "audio_in" y extraer su sample rate en una variable "sr":
audio_in, sr = librosa.load(audio_file[0])
print(f'Sample Rate: {sr}')  # Sample rate de la señal de audio
# Cálculo de la TF de la señal de audio de entrada:
TF_entrada = fourier.fft(audio_in)

# Inicializacion de la interfaz de usuario:
class MyApp(QMainWindow):
    def __init__(self):
        # Configuracion de objetos (sliders, botones, etc):
        super(MyApp, self).__init__()
        loadUi('Design.ui', self)
        self.click_position = QPoint()
        self.gripSize = 10
        self.grip = QtWidgets.QSizeGrip(self)
        self.grip.resize(self.gripSize, self.gripSize)
        self.sub_bass_sld.valueChanged.connect(self.slider_values1)
        self.bass_sld.valueChanged.connect(self.slider_values2)
        self.low_mids_sld.valueChanged.connect(self.slider_values3)
        self.high_mids_sld.valueChanged.connect(self.slider_values4)
        self.presence_sld.valueChanged.connect(self.slider_values5)
        self.brilliance_sld.valueChanged.connect(self.slider_values6)
        l_m_values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        self.expand_L.addItems(l_m_values)
        self.decimar_M.addItems(l_m_values)
        self.play_bt.clicked.connect(self.play_boton)
        self.graf_xn = Canv_graf_xn()
        self.graf_yn = Canv_graf_yn()
        self.graf_zn = Canv_graf_zn()
        self.graf_xjw = Canv_graf_xjw()
        self.graf_yjw = Canv_graf_yjw()
        self.graf_zjw = Canv_graf_zjw()
        self.layout_xn.addWidget(self.graf_xn)
        self.layout_yn.addWidget(self.graf_yn)
        self.layout_zn.addWidget(self.graf_zn)
        self.layout_xjw.addWidget(self.graf_xjw)
        self.layout_yjw.addWidget(self.graf_yjw)
        self.layout_zjw.addWidget(self.graf_zjw)

    def resizeEvent(self, event):
        rect = self.rect()
        self.grip.move(rect.right()-self.gripSize, rect.bottom()-self.gripSize)

    # Actualizacion de la posicion de los sliders:
    def slider_values1(self, event):
        g1 = self.sub_bass_sld.value()
        self.sub_bas_label.setText(str(g1))

    def slider_values2(self, event):
        g2 = self.bass_sld.value()
        self.bas_label.setText(str(g2))

    def slider_values3(self, event):
        g3 = self.low_mids_sld.value()
        self.low_mids_label.setText(str(g3))

    def slider_values4(self, event):
        g4 = self.high_mids_sld.value()
        self.high_mids_label.setText(str(g4))

    def slider_values5(self, event):
        g5 = self.presence_sld.value()
        self.presence_label.setText(str(g5))

    def slider_values6(self, event):
        g6 = self.brilliance_sld.value()
        self.brilliance_label.setText(str(g6))

    # Funcion que arranca el procesamiento de la señal
    def play_boton(self, event):
        # Datos capturados de la interfaz de usuario:
        L = int(self.expand_L.currentText())
        M = int(self.decimar_M.currentText())
        # Ganancias capturadas de los sliders:
        g1 = self.sub_bass_sld.value()
        g2 = self.bass_sld.value()
        g3 = self.low_mids_sld.value()
        g4 = self.high_mids_sld.value()
        g5 = self.presence_sld.value()
        g6 = self.brilliance_sld.value()
        # Se envia el sample rate resultante hacia la interfaz de usuario:
        self.sr_out.setText(str((sr * L) / M))

        # Comienza proceso de expansion:
        expandir = list(audio_in)
        expandido = []
        for k in range(0, len(expandir)):
            expandido.append(expandir[k])
            if k != len(expandir) - 1:
                for r in range(0, L - 1):
                    expandido.append(0)

        expandido = np.array(expandido)
        # TF de la señal expandida:
        TF_expandido = fourier.fft(expandido)
        # Generacion del filtro para eliminar replicas
        h_bajo = firwin(numtaps=len(expandido), cutoff=(sr/2)/L, window='hamming', pass_zero=True, fs=sr+2)
        H_jw = fourier.fft(h_bajo)
        # Aplicacion del filtro por multiplicacion en frecuencia
        Ef = TF_expandido * H_jw
        # Transformada inversa de Fourier de señal expandida
        exp_fil = fourier.ifft(Ef)
        exp_fil = np.concatenate((exp_fil[len(exp_fil) // 2: len(exp_fil)], exp_fil[0: len(exp_fil) // 2]), axis=None)

        # Comienza proceso de decimacion sobre la señal expandida:
        decimar = list(np.real(exp_fil))
        y_n = [decimar[i] for i in range(0, len(decimar)) if i % M == 0]
        y_n = np.array(y_n)

        # TF de la señal decimada
        TF_decimado = fourier.fft(y_n)

        # Comienza proceso de exualizacion sobre la señal decimada:
        # Generacion de los 6 filtros paso banda multiplicados por su ganancia
        h_sub_bass = int(g1) * firwin(numtaps=len(y_n), cutoff=[16, 60], window='hamming', pass_zero=False, fs=32001)
        h_bass = int(g2) * firwin(numtaps=len(y_n), cutoff=[60, 250], window='hamming', pass_zero=False, fs=32001)
        h_low_mids = int(g3) * firwin(numtaps=len(y_n), cutoff=[250, 2000], window='hamming', pass_zero=False, fs=32001)
        h_high_mids = int(g4) * firwin(numtaps=len(y_n), cutoff=[2000, 4000], window='hamming', pass_zero=False, fs=32001)
        h_presence = int(g5) * firwin(numtaps=len(y_n), cutoff=[4000, 6000], window='hamming', pass_zero=False, fs=32001)
        h_brilliance = int(g6) * firwin(numtaps=len(y_n), cutoff=[6000, 16000], window='hamming', pass_zero=False, fs=32001)

        # Se suman las respuestas en frecuencia para formar un solo filtro multibanda
        EQ = (fourier.fft(h_sub_bass) + fourier.fft(h_bass) + fourier.fft(h_low_mids) + fourier.fft(h_high_mids) +
              fourier.fft(h_presence) + fourier.fft(h_brilliance))

        # Se aplica el filtro multibanda sobre la señal decimada:
        Ecualizar = TF_decimado * EQ

        # Transformada inversa de Fourier para obtener el audio resultante
        audio_out = fourier.ifft(Ecualizar)
        audio_out = np.concatenate((audio_out[len(audio_out) // 2: len(audio_out)], audio_out[0: len(audio_out) // 2])
                                   , axis=None)

        # Inicia el proceso de representacion grafica de las señales:
        # Eje temporal para las señales en el dominio del tiempo:
        time_axis_xn = np.arange(0, len(audio_in)) / sr
        frec_exp = fourier.fftfreq(len(expandido), 1/sr)
        time_axis_zn = np.arange(0, len(audio_out)) / ((sr * L) / M)
        # Gráfica de la señal x[n]
        self.graf_xn.plot_signal(time_axis_xn, audio_in)
        # Gráfica de la señal y[n]
        self.graf_yn.plot_signal(frec_exp, abs(TF_expandido))
        # Gráfica de la señal z[n]
        self.graf_zn.plot_signal(time_axis_zn, np.real(audio_out))
        # Eje de frecuencia para el dominio de la frecuencia:
        freq_axis_xjw = fourier.fftfreq(len(audio_in), 1/sr)
        freq_axis_yjw = np.fft.fftfreq(len(TF_decimado), 1 / sr)
        freq_axis_zjw = np.fft.fftfreq(len(Ecualizar), 1 / sr)
        # Gráfica del espectro X(jw)
        self.graf_xjw.plot_spectrum(freq_axis_xjw, np.abs(TF_entrada))
        # Gráfica del espectro Y(jw)
        self.graf_yjw.plot_spectrum(freq_axis_yjw, np.abs(TF_decimado))
        # Gráfica del espectro Z(jw)
        self.graf_zjw.plot_spectrum(freq_axis_zjw, np.abs(Ecualizar))

        # Comienza la reconstruccion del audio para reproducirlo:
        reconstruir = Ecualizar

        # Se eliminan los datos con frecuencias negativas:
        for i in range(len(reconstruir) // 2, len(reconstruir)):
            reconstruir[i] = 0

        # Transformada inversa de Fourier del espectro positivo, señal lista:
        audio_out = fourier.ifft(reconstruir)
        audio_out = np.concatenate((audio_out[len(audio_out) // 2: len(audio_out)], audio_out[0: len(audio_out) // 2])
                                   , axis=None)
        # Reproducir la señal de audio resultante:
        sd.play(np.real(audio_out), sr*L/M)

# Estructuras tipo clase para graficar las señales:
class Canv_graf_xn(canva):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, dpi=100, figsize=(5, 5), sharey=True, facecolor='white')
        super().__init__(self.fig)

    def plot_signal(self, time, signal):
        self.ax.clear()
        self.ax.plot(time, signal)
        self.ax.set_title('Audio de Entrada')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.draw()

class Canv_graf_yn(canva):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, dpi=100, figsize=(5, 5), sharey=True, facecolor='white')
        super().__init__(self.fig)

    def plot_signal(self, time, signal):
        self.ax.clear()
        self.ax.plot(time, signal)
        self.ax.set_title('Espectro de Expansion')
        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Magnitude')
        self.draw()

class Canv_graf_zn(canva):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, dpi=100, figsize=(5, 5), sharey=True, facecolor='white')
        super().__init__(self.fig)

    def plot_signal(self, time, signal):
        self.ax.clear()
        self.ax.plot(time, signal)
        self.ax.set_title('Audio de Salida')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.draw()

class Canv_graf_xjw(canva):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, dpi=100, figsize=(5, 5), sharey=True, facecolor='white')
        super().__init__(self.fig)

    def plot_spectrum(self, freq, spectrum):
        self.ax.clear()
        self.ax.plot(freq, spectrum)
        self.ax.set_title('Espectro Audio Entrada')
        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Magnitude')
        self.draw()


class Canv_graf_yjw(canva):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, dpi=100, figsize=(5, 5), sharey=True, facecolor='white')
        super().__init__(self.fig)

    def plot_spectrum(self, freq, spectrum):
        self.ax.clear()
        self.ax.plot(freq, spectrum)
        self.ax.set_title('Espectro de Decimation')
        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Magnitude')
        self.draw()

class Canv_graf_zjw(canva):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, dpi=100, figsize=(5, 5), sharey=True, facecolor='white')
        super().__init__(self.fig)

    def plot_spectrum(self, freq, spectrum):
        self.ax.clear()
        self.ax.plot(freq, spectrum)
        self.ax.set_title('Espectro Ecualizado')
        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Magnitude')
        self.draw()

# Inicia la app:
if __name__ == '__main__':
    app = QApplication(sys.argv)
    test_app = MyApp()
    test_app.show()
    sys.exit(app.exec_())
