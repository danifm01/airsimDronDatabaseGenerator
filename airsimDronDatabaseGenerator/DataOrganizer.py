import pandas as pd
import numpy as np
import cv2
import os


class DataOrganizer:
    def __init__(self, imagenes: list = None, imagenesMarcadas: list = None,
                 parametros: list = None):
        if imagenes is None:
            imagenes = []
        if imagenesMarcadas is None:
            imagenesMarcadas = []
        if parametros is None:
            parametros = []
        self.imagenes = imagenes
        self.imagenesMarcadas = imagenesMarcadas
        self.parametros = parametros

    def addImagenes(self, imagenes: list):
        for ima in imagenes:
            self.imagenes.append(ima)

    def addImagenesMarcadas(self, imagenesMarcadas: list):
        for ima in imagenesMarcadas:
            self.imagenesMarcadas.append(ima)

    def addParametros(self, parametros: list):
        for param in parametros:
            self.parametros.append(param)

    def addInfo(self, imagenes, imagenesMarcadas, parametros):
        self.addImagenes(imagenes)
        self.addImagenesMarcadas(imagenesMarcadas)
        self.addParametros(parametros)

    def crearDataFrame(self):
        nombres = []
        for index, ima in enumerate(self.imagenes):
            nombre = 'Imagen ' + str(index) + '.jpg'
            cv2.imwrite(
                os.path.join(os.path.dirname(__file__), 'Imagenes\\') + nombre,
                ima)
            nombres.append(nombre)
        data = pd.DataFrame(np.array(self.parametros),
                            columns=['distancia', 'phi', 'theta', 'xIma',
                                     'yIma', 'radioIma', 'orientacionVisor',
                                     'orientacionVisto',
                                     'orientacionRelativa'])
        data.insert(0, 'nombre', nombres)
        return data
