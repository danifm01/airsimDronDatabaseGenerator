import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Clase encargada de tomar imágenes de un drón en el simulador airsim a
# partir de la cámara de otro drón.
class ImagesGenerator:
    def __init__(self, DronControllerVisor, DronControllerVisto, dataCalc):
        self.dron1 = DronControllerVisor
        self.dron2 = DronControllerVisto
        self.dataCalculator = dataCalc

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
    # Devuleve una lista de nImagenes del dron 2 visto desde el dron 1,
    # otra lista con las mismas imágenes marcado donde se encuentra el dron 2
    # y una lista de listas en la que se encuentran los parámetros de cada
    # imágen tomada.
    def tomarImagenesAleatoriasConParametros(self, nImagenes):
        imagenes = []
        imagenesMarcadas = []
        parametros = []
        for i in range(nImagenes):
            self.dron1.irAposeAleatoria()
            time.sleep(0.2)
            theta, phi, distancia, poseMovido = (
                self.dron2.moverAleatorioAcampoDeVision(
                    self.dron1.nombre))
            ima = self.dron1.tomarImagen(False)
            imagenes.append(ima)
            coordAncho, coordAlto = (
                self.dataCalculator.calcularCoordenadasImagen(distancia, theta,
                                                              phi))
            radioAncho = self.dataCalculator.calcularRadio(np.sqrt(2) / 2,
                                                           distancia)
            imagenesMarcadas.append(self.dibujarRadio(ima, radioAncho,
                                                      coordAncho, coordAlto))

            parametros.append(self.dataCalculator.calcularParametros())
        return imagenes, imagenesMarcadas, parametros

    # Dibuja una circunferencia verde en la imágen (ima) en la posición (alto y
    # ancho) especificadas en pixeles y con el radio indicado en pixeles.
    # Marca el centro de la circunfernecia mediante un punto rojo
    @staticmethod
    def dibujarRadio(ima, radio, ancho, alto):
        img = ima.copy()
        cv2.circle(img, (int(ancho), int(alto)), int(radio), (0, 255, 0), 3)
        cv2.circle(img, (int(ancho), int(alto)), 5, (255, 0, 0), -1)
        plt.imshow(img)
        plt.show()
        return img
