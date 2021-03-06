import copy
import random
import time

import airsim
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


# Clase encargada de controlar un dron en airsim
class DroneController:
    # Es necesario introducir el nombre del dron en airsim y el cliente de la
    # conexión a airsim
    def __init__(self, nombre, client):
        self.client = client

        self.nombre = nombre
        self.pose = self.client.simGetVehiclePose(self.nombre)

        # Parametros de la cámara
        self.cameraPosition = np.array([0.46, 0, 0])  # Metros
        self.anchoCamara = 1920  # Pixeles
        self.altoCamara = 1080  # Pixeles
        self.fovHorCamara = 84.  # Grados
        self.fovVerCamara = (self.fovHorCamara * self.altoCamara /
                             self.anchoCamara)  # Grados
        self.ratio = self.anchoCamara / self.altoCamara

    # Mueve el dron a las coordenadas y con la orientación indicada como
    # parámetro. Unidades en metros y radianes.
    def teleportDron(self, x, y, z, pitch, roll, yaw):
        self.pose.position = airsim.Vector3r(x, y, z)
        self.pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
        self.client.simSetVehiclePose(self.pose, True, self.nombre)
        return self.pose

    # Mueve el dron a una pose aleatoria con valores de posición entre los
    # límites pasados como parámetros y como orientación valores de pitch
    # y roll entre -0.8 y 0.8 radianes y de yaw entre -pi y pi
    def irAposeAleatoria(self, x_max=100, x_min=-100, y_max=100, y_min=-100,
                         z_max=-1, z_min=-10):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        pitch = random.uniform(-0.8, 0.8)
        roll = random.uniform(-0.8, 0.8)
        yaw = random.uniform(-np.pi, np.pi)
        self.teleportDron(x, y, z, pitch, roll, yaw)

    def getPose(self):
        self.pose = self.client.simGetVehiclePose(self.nombre)
        return self.pose

    # Devuelve una imagen de la escena en formato np array RGB uint8. La
    # imágen se toma colocando la cámara en la posición en la que se
    # encuentra el drón con su misma orientación, sin tener en consideración la
    # posición de la cámara relativa al dron.
    # Si mostrar es True se imprime la imágen devuelta.
    def tomarImagen(self, mostrar=True):
        # Coloca la cámara en la posición del dron
        poseDronInicial = self.client.simGetVehiclePose(self.nombre)
        rot = Rotation.from_quat([poseDronInicial.orientation.x_val,
                                  poseDronInicial.orientation.y_val,
                                  poseDronInicial.orientation.z_val,
                                  poseDronInicial.orientation.w_val])
        cameraPos = rot.apply(self.cameraPosition)
        poseDronNueva = copy.deepcopy(poseDronInicial)
        poseDronNueva.position.x_val -= cameraPos[0]
        poseDronNueva.position.y_val -= cameraPos[1]
        poseDronNueva.position.z_val -= cameraPos[2]
        self.client.simSetVehiclePose(poseDronNueva, True, self.nombre)
        time.sleep(0.5)

        # Toma de imágenes
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
             airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True,
                                 False),
             airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True,
                                 False),
             airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
             airsim.ImageRequest("0", airsim.ImageType.Segmentation, False,
                                 False),
             ],
            self.nombre)
        imagenes = []
        for index, response in enumerate(responses):
            ima1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            try:
                try:
                    ima = ima1d.reshape(response.height, response.width, 3)
                except ValueError:
                    ima1d = np.asarray(response.image_data_float,
                                       dtype=np.float)
                    ima = ima1d.reshape(response.height, response.width)
                if index == 0:
                    ima = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
            # Si se produce un error inesperado se devuelve -1
            except:
                return -1
            imagenes.append(ima)
        if mostrar:
            plt.imshow(imagenes[0])
            plt.show()

        # # Colocación del dron en la posición inicial
        self.client.simSetVehiclePose(poseDronInicial, True, self.nombre)
        time.sleep(0.5)
        return imagenes

    # Entra como parámetro el nombre del dron cuyo campo de visión se utilizará.
    # El dron se moverá a una posición y orientación aleatoria dentro del
    # campo de visión del dronVisor. Devuelve la distancia a la cámara,
    # los valores de los ángulos relativos a ella y la pose a la que se mueve
    # el dron.
    def moverAleatorioAcampoDeVision(self, dronVisor, distanciaBase,
                                     variacionDist=2.5):
        # distancia = random.uniform(2, 10)
        distancia = distanciaBase + random.uniform(-variacionDist,
                                                   variacionDist)
        maxPhi = self.fovHorCamara / 2
        maxTheta = self.fovVerCamara / 2
        poseVisor = self.client.simGetVehiclePose(dronVisor)
        theta = random.uniform(-maxTheta, maxTheta)
        phi = random.uniform(-maxPhi, maxPhi)
        poseMovido = self.calcularPoseRelativa(distancia, theta, phi, poseVisor)
        # Se añade una rotación aleatoria al dron movido
        pitch = random.uniform(-0.8, 0.8)
        roll = random.uniform(-0.8, 0.8)
        yaw = random.uniform(-np.pi, np.pi)
        poseMovido.orientation = airsim.to_quaternion(pitch, roll, yaw)

        self.client.simSetVehiclePose(poseMovido, True, self.nombre)
        return theta, phi, distancia, poseMovido

    # Mueve el dron a una posición relativa a la de otro dron. Entran como
    # parámetros el nombre del dron que se usará de referencia, la distancia
    # en metros a la que se colocará el dron y los ángulos theta y phi en
    # radianes que determinan la posición en coordenadas esféricas.
    def moverRelativoAcampoDeVision(self, dronVisor, distancia, theta, phi,
                                    pitch=None, roll=None, yaw=None):
        poseVisor = self.client.simGetVehiclePose(dronVisor)
        poseMovido = self.calcularPoseRelativa(distancia, theta, phi, poseVisor)
        # Se añade una rotación aleatoria al dron movido
        if pitch is None:
            pitch = random.uniform(-0.8, 0.8)
        if roll is None:
            roll = random.uniform(-0.8, 0.8)
        if yaw is None:
            yaw = random.uniform(-np.pi, np.pi)
        poseMovido.orientation = airsim.to_quaternion(pitch, roll, yaw)

        self.client.simSetVehiclePose(poseMovido, True, self.nombre)
        return theta, phi, distancia, poseMovido

    # Cálcula la pose del dron en el sistema global a partir de las coordenadas
    # esféricas relativas al dronVisor
    @staticmethod
    def calcularPoseRelativa(distancia, theta, phi, poseVisor):
        poseMovido = airsim.Pose(airsim.Vector3r(), airsim.Quaternionr())
        rot = Rotation.from_euler('ZYX', [phi, theta, 0], True)
        position = rot.apply([distancia, 0, 0])
        poseMovido.position.x_val += position[0]
        poseMovido.position.y_val += position[1]
        poseMovido.position.z_val += position[2]

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
        return poseMovido

    def comprobarPoseCorrecta(self):
        self.pose = self.client.simGetVehiclePose(self.nombre)
        # Comprobación de que se encuentra por encima del suelo
        if self.pose.position.z_val >= 0:
            return False
        # Comprobación de que no ha colisionado
        if self.client.simGetCollisionInfo(self.nombre).has_collided:
            return False
        return True
