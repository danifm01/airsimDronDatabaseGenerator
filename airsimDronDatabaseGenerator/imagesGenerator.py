import airsim
import numpy as np
import matplotlib.pyplot as plt
import cv2


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

    def teleportDron(self, dronNumber, x, y, z, pitch, roll, yaw):
        self.pose[dronNumber].position = airsim.Vector3r(x, y, z)
        self.pose[dronNumber].orientation = airsim.to_quaternion(pitch, roll,
                                                                 yaw)
        self.client.simSetVehiclePose(self.pose[dronNumber], True,
                                      self.dron[dronNumber])
