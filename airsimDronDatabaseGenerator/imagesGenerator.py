from scipy.spatial.transform import Rotation as R
import random
import airsim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


class ImagesGenerator:
    def __init__(self, nombreDron1='Drone1', nombreDron2='Drone2'):
        self.dron = [nombreDron1, nombreDron2]

        # Genera el cliente de airsim para un multirotor
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        pose1 = self.client.simGetVehiclePose(self.dron[0])
        pose2 = self.client.simGetVehiclePose(self.dron[1])
        self.pose = [pose1, pose2]
        # Parametros de la cámara
        self.fovCamara = 90  # Grados
        self.anchoCamara = 1280  # Pixeles
        self.altoCamara = 720  # Pixeles
        self.ratio = self.anchoCamara / self.altoCamara

    # Toma como parámetro el número del dron que se va a utilizar y
    # devuelve una imagen de la escena en formato np array RGB uint8.
    # Si mostrar es True se imprime la imágen devuelta.
    def tomarImagen(self, dronNumber=0, mostrar=True):
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene,
                                 False, False)], self.dron[dronNumber])

        response = responses[0]
        ima1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        imaBGR = ima1d.reshape(response.height, response.width, 3)
        imaRGB = cv2.cvtColor(imaBGR, cv2.COLOR_BGR2RGB)

        if mostrar:
            plt.imshow(imaRGB)
            plt.show()
        return imaRGB

    # Mueve el dron indicdo por el su número correspondiente (0 para el
    # primero y 1 para el segundo a las coordenadas y con la orientación
    # indicada como parámetro
    def teleportDron(self, dronNumber, x, y, z, pitch, roll, yaw):
        self.pose[dronNumber].position = airsim.Vector3r(x, y, z)
        self.pose[dronNumber].orientation = airsim.to_quaternion(pitch, roll,
                                                                 yaw)
        self.client.simSetVehiclePose(self.pose[dronNumber], True,
                                      self.dron[dronNumber])
        return self.pose[dronNumber]

    # Devuleve una lista de nImagenes del drone 2 visto desde el dron 1
    def tomarImagenesAleatorias(self, nImagenes):
        imagenes = []

        x_max = 10
        x_min = -10
        y_max = 10
        y_min = -10
        z_max = -1
        z_min = -10

        for i in range(nImagenes):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = random.uniform(z_min, z_max)
            pitch = random.uniform(-0.8, 0.8)
            roll = random.uniform(-0.8, 0.8)
            yaw = random.uniform(-np.pi, np.pi)
            self.teleportDron(0, x, y, z, pitch, roll, yaw)
            self.teleportDron(0, x, y, z, 0, 0, 0)

            time.sleep(0.5)
            self.moverAleatorioAcampoDeVision(self.dron[1], self.dron[0])

            imagenes.append(self.tomarImagen(0))
        return imagenes

    # Entra como parámetro el nombre de los dos drones utilizados. El
    # dronMovido se moverá a una posición y orientación aleatorio dentro del
    # campo de visión del dronVisor.
    def moverAleatorioAcampoDeVision(self, dronMovido, dronVisor):
        distanciaPlano = random.uniform(2, 10)
        maxAncho = (distanciaPlano * np.tan(np.radians(self.fovCamara / 2)) *
                    np.sin(np.arctan(self.ratio)))
        maxAlto = maxAncho / self.ratio

        poseVisor = self.client.simGetVehiclePose(dronVisor)
        poseMovido = airsim.Pose(airsim.Vector3r(), airsim.Quaternionr())

        poseMovido.position.x_val += distanciaPlano
        poseMovido.position.y_val += maxAncho
        poseMovido.position.z_val += maxAlto

        rot = R.from_quat([poseVisor.orientation.x_val,
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
        self.client.simSetVehiclePose(poseMovido, True, dronMovido)
