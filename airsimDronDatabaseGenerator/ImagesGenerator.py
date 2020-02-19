import time
import numpy as np
import cv2
from matplotlib import pyplot as plt


class ImagesGenerator:
    def __init__(self, DronController1, DronController2):
        self.dron1 = DronController1
        self.dron2 = DronController2
        # Parametros de la cámara
        self.fovCamara = 90  # Grados
        self.anchoCamara = 1280  # Pixeles
        self.altoCamara = 720  # Pixeles
        self.ratio = self.anchoCamara / self.altoCamara

    # Devuleve una lista de nImagenes del dron 2 visto desde el dron 1
    def tomarImagenesAleatorias(self, nImagenes):
        imagenes = []
        for i in range(nImagenes):
            self.dron1.irAposeAleatoria()
            time.sleep(0.2)
            self.dron2.moverAleatorioAcampoDeVision(self.dron1.nombre)
            time.sleep(0.2)
            imagenes.append(self.dron1.tomarImagen())
        return imagenes

    # TODO: Añadir cálculo de parámetros distReal (m), orientación coordAncho
    #  (pixel), coordAlto (pixel), Radio (pixel) y ¿Bounding Box? al método
    # Devuleve una lista de nImagenes del dron 2 visto desde el dron 1
    def tomarImagenesAleatoriasConParametros(self, nImagenes):
        imagenes = []
        for i in range(nImagenes):
            self.dron1.irAposeAleatoria()
            time.sleep(0.2)
            maxAncho, maxAlto, ancho, alto, poseMovido = (
                self.dron2.moverAleatorioAcampoDeVisionPolares(
                    self.dron1.nombre))
            time.sleep(0.2)
            ima = self.dron1.tomarImagen()
            imagenes.append(ima)
            coordAncho, coordAlto = (
                self.calcularCoordenadas(maxAncho, maxAlto, ancho, alto))
            radioAncho = self.calcularRadio(np.sqrt(2) / 2, maxAncho)
            self.dibujarRadio(ima, radioAncho, coordAncho, coordAlto)
        return imagenes

    def calcularCoordenadas(self, maxAncho, maxAlto, ancho, alto):
        coordAncho = (self.anchoCamara / 2 / maxAncho * ancho +
                      self.anchoCamara / 2)
        coordAlto = self.altoCamara / 2 / maxAlto * alto + self.altoCamara / 2
        return coordAncho, coordAlto

    def calcularRadio(self, radioReal, maxAncho):
        radio = radioReal * self.anchoCamara / maxAncho / 2
        return radio

    @staticmethod
    def dibujarRadio(ima, radio, ancho, alto):
        img = cv2.circle(ima, (int(ancho), int(alto)), int(radio), (0, 0, 0), 5)
        plt.imshow(img)
        plt.show()
        return img
