import airsim
import cv2
import matplotlib.pyplot as plt
import numpy as np

from DroneController import DroneController
from ImagesGenerator import ImagesGenerator
from DataCalculator import DataCalculator
from DataOrganizer import DataOrganizer

# Devuelve el cliente de airsim para un multirotor
def crearCliente():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


# Toma como parámetro el cliente de airsim y devuelve una imagen de la
# escena en formato np array BGR uint8
def tomarImagen(client, dronName):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
        dronName)
    response = responses[0]
    ima1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    imaBGR = ima1d.reshape(response.height, response.width, 3)
    return imaBGR


def mostrarImagen(imaBGR):
    plt.imshow(cv2.cvtColor(imaBGR, cv2.COLOR_BGR2RGB))
    plt.show()


def teleportDron(dronName, client, pose=airsim.Pose()):
    client.simSetVehiclePose(pose, True, dronName)


def main():
    cliente = crearCliente()

    dron1 = DroneController('Drone1', cliente)
    dron2 = DroneController('Drone2', cliente)
    dataCalc = DataCalculator(cliente, 'Drone1', 'Drone2')

    generador = ImagesGenerator(dron1, dron2, dataCalc)
    imagenes, imagenesMarcadas, parametros = (
        generador.tomarImagenesAleatoriasConParametros(2))
    organizador = DataOrganizer(imagenes, imagenesMarcadas, parametros)
    organizador.crearDataBase()


if __name__ == "__main__":
    main()
