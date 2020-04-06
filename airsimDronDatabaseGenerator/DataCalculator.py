import numpy as np
from scipy.spatial.transform import Rotation
import airsim


class DataCalculator:
    def __init__(self, cliente, dronVisor, dronVisto):
        self.client = cliente
        self.dronVisor = dronVisor
        self.dronVisto = dronVisto
        self.poseVisor = None
        self.poseVisto = None
        # Parametros de la cámara
        self.fovCamara = 90  # Grados
        self.anchoCamara = 1280  # Pixeles
        self.altoCamara = 720  # Pixeles
        self.ratio = self.anchoCamara / self.altoCamara
        self.focal = self.anchoCamara / 2 / (
            np.tan(np.radians(self.fovCamara / 2)))  # distancia focal
        self.updatePose()

    def updatePose(self):
        self.poseVisor = self.client.simGetVehiclePose(self.dronVisor)
        self.poseVisto = self.client.simGetVehiclePose(self.dronVisto)

    def calcularDistancia(self, updatePose=True):
        if updatePose:
            self.updatePose()
        return self.poseVisor.position.distance_to(self.poseVisto.position)

    def orientacionAbsolutaVisor(self, updatePose=True):
        if updatePose:
            self.updatePose()
        return self.poseVisor.orientation

    def orientacionAbsolutaVisto(self, updatePose=True):
        if updatePose:
            self.updatePose()
        return self.poseVisto.orientation

    def orientacionRelativaVisto(self, updatePose=True):
        if updatePose:
            self.updatePose()
        qr1 = self.poseVisto.orientation.w_val
        qx1 = self.poseVisto.orientation.x_val
        qy1 = self.poseVisto.orientation.y_val
        qz1 = self.poseVisto.orientation.z_val
        # Inversión del quaternio del dron visor
        qr2 = self.poseVisor.orientation.w_val
        qx2 = -self.poseVisor.orientation.x_val
        qy2 = -self.poseVisor.orientation.y_val
        qz2 = -self.poseVisor.orientation.z_val
        # Composición de ambos cuaternios
        w_quat = qr1 * qr2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        x_quat = qr1 * qx2 + qr2 * qx1 + qy1 * qz2 - qy2 * qz1
        y_quat = qr1 * qy2 + qr2 * qy1 + qz1 * qx2 - qz2 * qx1
        z_quat = qr1 * qz2 + qr2 * qz1 + qx1 * qy2 - qx2 * qy1
        return airsim.Quaternionr(x_quat, y_quat, z_quat, w_quat)

    # Devuelve la posición en la imágen en pixeles en la que se encuentra el
    # objeto especificado por su distancia a la cámara y sus giros respecto a
    # esta
    def calcularCoordenadasImagen(self, distancia, theta, phi, updatePose=True):
        if updatePose:
            self.updatePose()
        rot = Rotation.from_euler('ZYX', [phi, theta, 0], True)
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
    # radio real y su distancia a la cámara
    def calcularRadio(self, radioReal, distancia):
        radio = radioReal * self.focal / distancia
        return radio

    def calcularParametros(self, distancia, theta, phi):
        self.updatePose()
        xIma, yIma = self.calcularCoordenadasImagen(distancia, theta, phi,
                                                    False)
        radioIma = self.calcularRadio(np.sqrt(2) / 2, distancia)
        orientacionVisor = self.orientacionAbsolutaVisor(False)
        orientacionVisto = self.orientacionAbsolutaVisto(False)
        orientacionRelativa = self.orientacionRelativaVisto(False)
        return [distancia, phi, theta, xIma, yIma, radioIma, orientacionVisor,
                orientacionVisto, orientacionRelativa]
