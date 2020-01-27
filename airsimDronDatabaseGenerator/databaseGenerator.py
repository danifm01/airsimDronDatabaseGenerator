import airsim
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Devuelve el cliente de airsim para un multirotor
def crearCliente():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


# Toma como parámetro el cliente de airsim y devuelve una imagen de la
# escena en formato np array BGR uint8
def tomarImagen(client):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_bgr = img1d.reshape(response.height, response.width, 3)
    return img_bgr


def mostrarImagen(imaBGR):
    plt.imshow(cv2.cvtColor(imaBGR, cv2.COLOR_BGR2RGB))
    plt.show()


def main():
    client = crearCliente()

    img_bgr = tomarImagen(client)
    mostrarImagen(img_bgr)

    client.takeoffAsync().join()
    client.moveToPositionAsync(-10, 10, -10, 5).join()

    img_bgr = tomarImagen(client)
    mostrarImagen(img_bgr)

    client.moveToPositionAsync(0, 0, 2, 5).join()
    client.landAsync().join()


if __name__ == "__main__":
    main()

