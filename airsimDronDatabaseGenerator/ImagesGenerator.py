import time

import airsim
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


# Clase encargada de tomar imágenes de un drón en el simulador airsim a
# partir de la cámara de otro drón.
class ImagesGenerator:
    def __init__(self, DronController1, DronController2):
        self.dron1 = DronController1
        self.dron2 = DronController2
        # Parametros de la cámara
        self.fovCamara = 90  # Grados
        self.anchoCamara = 1280  # Pixeles
        self.altoCamara = 720  # Pixeles
        self.ratio = self.anchoCamara / self.altoCamara
        self.focal = self.anchoCamara / 2 / (
            np.tan(np.radians(self.fovCamara / 2)))  # distancia focal

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
            theta, phi, distancia, poseMovido = (
                self.dron2.moverAleatorioAcampoDeVisionPolares(
                    self.dron1.nombre))
            # time.sleep(0.2)
            ima = self.dron1.tomarImagen(False)
            imagenes.append(ima)
            coordAncho, coordAlto = (
                self.calcularCoordenadasImagen(distancia, theta, phi))
            radioAncho = self.calcularRadio(np.sqrt(2) / 2, distancia)
            self.dibujarRadio(ima, radioAncho, coordAncho, coordAlto)
        return imagenes

    # Devuelve la posición en la imágen en pixeles en la que se encuentra el
    # objeto especificado por su distancia a la cámara y sus giros respecto a
    # esta
    def calcularCoordenadasImagen(self, distancia, theta, phi):
        rot = Rotation.from_euler('ZYX', [theta, phi, 0], True)
        position = rot.apply([distancia, 0, 0])
        # Ajustar sistemas de coordenadas
        position = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).dot(position)
        cameraMatrix = np.array(
            [[-self.focal, 0, self.anchoCamara / 2],
             [0, -self.focal, self.altoCamara / 2],
             [0, 0, 1]])
        imagePosition = cameraMatrix.dot(position.T)
        imagePosition = imagePosition / imagePosition[2]
        return imagePosition[0], imagePosition[1]

    # Determina el valor en pixeles que tiene un objeto en función de su
    # radio y su distancia
    def calcularRadio(self, radioReal, distancia):
        radio = radioReal * self.focal / distancia
        return radio

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
