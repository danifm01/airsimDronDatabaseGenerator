import time


class ImagesGenerator:
    def __init__(self, DronController1, DronController2):
        self.dron1 = DronController1
        self.dron2 = DronController2

    # Devuleve una lista de nImagenes del dron 2 visto desde el dron 1
    def tomarImagenesAleatorias(self, nImagenes):
        imagenes = []
        for i in range(nImagenes):
            self.dron1.irAposeAleatoria()
            time.sleep(0.5)
            self.dron2.moverAleatorioAcampoDeVision(self.dron1.nombre)
            imagenes.append(self.dron1.tomarImagen())
        return imagenes
