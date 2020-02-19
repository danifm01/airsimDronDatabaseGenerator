import random

import airsim
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


class DroneController:
    def __init__(self, nombre, client):
        self.client = client

        self.nombre = nombre
        self.pose = self.client.simGetVehiclePose(self.nombre)

        # Parametros de la cámara
        self.anchoCamara = 1280  # Pixeles
        self.altoCamara = 720  # Pixeles
        self.fovHorCamara = 90  # Grados
        self.fovVerCamara = (self.fovHorCamara * self.altoCamara /
                             self.anchoCamara)  # Grados
        self.ratio = self.anchoCamara / self.altoCamara

    # Mueve el dron a las coordenadas y con la orientación indicada como
    # parámetro
    def teleportDron(self, x, y, z, pitch, roll, yaw):
        self.pose.position = airsim.Vector3r(x, y, z)
        self.pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
        self.client.simSetVehiclePose(self.pose, True, self.nombre)
        return self.pose

    # Mueve el dron a una pose aleatoria con valores de posición entre los
    # límites pasados como parámetros y como orientación valores de pitch
    # y roll entre -0.8 y 0.8 radianes y de yaw entre -pi y pi
    def irAposeAleatoria(self, x_max=10, x_min=-10, y_max=10, y_min=-10,
                         z_max=-1, z_min=-10):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        pitch = random.uniform(-0.8, 0.8)
        roll = random.uniform(-0.8, 0.8)
        yaw = random.uniform(-np.pi, np.pi)
        self.teleportDron(x, y, z, pitch, roll, yaw)

    # Devuelve una imagen de la escena en formato np array RGB uint8.
    # Si mostrar es True se imprime la imágen devuelta.
    def tomarImagen(self, mostrar=True):
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
            self.nombre)
        response = responses[0]
        ima1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        imaBGR = ima1d.reshape(response.height, response.width, 3)
        imaRGB = cv2.cvtColor(imaBGR, cv2.COLOR_BGR2RGB)
        if mostrar:
            plt.imshow(imaRGB)
            plt.show()
        return imaRGB

    # Entra como parámetro el nombre del dron cuyo campo de visión se utilizará.
    # El dron se moverá a una posición y orientación aleatoria dentro del
    # campo de visión del dronVisor.
    def moverAleatorioAcampoDeVision(self, dronVisor):
        distanciaPlano = random.uniform(2, 10)
        maxAncho, maxAlto = self.calcularDesviacionMaxima(distanciaPlano)

        poseVisor = self.client.simGetVehiclePose(dronVisor)
        poseMovido, ancho, alto = self.calcularPoseEnCampoDeVision(
            distanciaPlano, maxAlto, maxAncho, poseVisor)
        self.client.simSetVehiclePose(poseMovido, True, self.nombre)
        return maxAncho, maxAlto, ancho, alto, poseMovido

    # Entra como parámetro el nombre del dron cuyo campo de visión se utilizará.
    # El dron se moverá a una posición y orientación aleatoria dentro del
    # campo de visión del dronVisor.
    def moverAleatorioAcampoDeVisionPolares(self, dronVisor):
        distancia = random.uniform(2, 10)
        maxTheta = self.fovHorCamara / 2
        maxRho = self.fovVerCamara / 2
        poseVisor = self.client.simGetVehiclePose(dronVisor)
        theta = random.uniform(-maxTheta, maxTheta)
        rho = random.uniform(-maxRho, maxRho)

        poseMovido = self.calcularPoseRelativa(distancia, theta, rho, poseVisor)
        self.client.simSetVehiclePose(poseMovido, True, self.nombre)
        return maxTheta, maxRho, theta, rho, poseMovido

    def calcularDesviacionMaxima(self, distanciaPlano):
        maxAncho = distanciaPlano * np.tan(np.radians(self.fovHorCamara / 2))
        maxAlto = maxAncho / self.ratio
        return maxAncho, maxAlto

    def calcularPoseRelativa(self, distancia, theta, rho, poseVisor):
        poseMovido = airsim.Pose(airsim.Vector3r(), airsim.Quaternionr())
        poseMovido.position.x_val += distancia * np.sin(theta) * np.cos(rho)
        poseMovido.position.y_val += distancia * np.sin(theta) * np.sin(rho)
        poseMovido.position.z_val += distancia * np.cos(theta)
        rot = Rotation.from_quat([poseVisor.orientation.x_val,
                                  poseVisor.orientation.y_val,
                                  poseVisor.orientation.z_val,
                                  poseVisor.orientation.w_val])
        nuevaPosition = rot.apply([poseMovido.position.x_val,
                                   poseMovido.position.y_val,
                                   poseMovido.position.z_val])
        nuevaPosition += [poseVisor.position.x_val, poseVisor.position.y_val,
                          poseVisor.position.z_val]
        poseMovido.position = airsim.Vector3r(nuevaPosition[0],
                                              nuevaPosition[1],
                                              nuevaPosition[2])
        poseMovido = poseVisor
        return poseMovido

    # TODO: Definir posición aleatoria
    @staticmethod
    def calcularPoseEnCampoDeVision(distanciaPlano, maxAlto, maxAncho,
                                    poseVisor):
        poseMovido = airsim.Pose(airsim.Vector3r(), airsim.Quaternionr())
        poseMovido.position.x_val += distanciaPlano
        ancho = random.uniform(-maxAncho, maxAncho)
        alto = random.uniform(-maxAlto, maxAlto)
        # ancho = 0
        # alto = 0
        poseMovido.position.y_val += ancho
        poseMovido.position.z_val += alto
        rot = Rotation.from_quat([poseVisor.orientation.x_val,
                                  poseVisor.orientation.y_val,
                                  poseVisor.orientation.z_val,
                                  poseVisor.orientation.w_val])
        nuevaPosition = rot.apply([poseMovido.position.x_val,
                                   poseMovido.position.y_val,
                                   poseMovido.position.z_val])
        nuevaPosition += [poseVisor.position.x_val, poseVisor.position.y_val,
                          poseVisor.position.z_val]
        poseMovido.position = airsim.Vector3r(nuevaPosition[0],
                                              nuevaPosition[1],
                                              nuevaPosition[2])
        return poseMovido, ancho, alto

    # TODO: Crear metodo que devuelva los parámetros necesarios: ima, maxAncho,
    #  maxAlto, ancho, alto...
