import airsim
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imagesGenerator import ImagesGenerator


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
    generador = ImagesGenerator('Drone1', 'Drone2')
    generador.teleportDron(0, 1, 1, -1, 1, 1, 1)
    generador.tomarImagen(0)
    # client = crearCliente()
    #
    # teleportDron('Drone1', client,
    #              airsim.Pose(airsim.Vector3r(3, 0, -0.5),
    #                          airsim.to_quaternion(1, -1, 3.14 / 2)))
    #
    # imaBGR = tomarImagen(client, 'Drone1')
    # mostrarImagen(imaBGR)
    #
    # imaBGR = tomarImagen(client, 'Drone2')
    # mostrarImagen(imaBGR)

    # client.takeoffAsync().join()
    # client.moveToPositionAsync(-10, 10, -10, 5).join()
    #
    # imaBGR = tomarImagen(client)
    # mostrarImagen(imaBGR)
    #
    # client.moveToPositionAsync(0, 0, 2, 5).join()
    # client.landAsync().join()


if __name__ == "__main__":
    main()
