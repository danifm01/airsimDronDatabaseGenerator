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

    def crearDataBase(self):
        nombres = []
        dirName = self.crearDirectorio('Data\\Imagenes')

        for index, ima in enumerate(self.imagenes):
            nombre = str(index) + '_Imagen_Blocks' + '.jpg'
            imaBGR = cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dirName, nombre), imaBGR)
            nombres.append(nombre)

        dirName = self.crearDirectorio('Data\\ImagenesMarcadas')
        for index, ima in enumerate(self.imagenesMarcadas):
            nombre = str(index) + '_Imagen_Marcada_Blocks' + '.jpg'
            imaBGR = cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dirName, nombre), imaBGR)

        data = pd.DataFrame(np.array(self.parametros),
                            columns=['distancia', 'phi', 'theta', 'xIma',
                                     'yIma', 'radioIma', 'orientacionVisor',
                                     'orientacionVisto',
                                     'orientacionRelativa'])
        data.insert(0, 'nombre', nombres)
        dirName = self.crearDirectorio('Data\\Parametros')
        data.to_csv(os.path.join(dirName, 'Parametros.csv'), index=False)
        return data

    @staticmethod
    def crearDirectorio(ruta: str):
        dirName = os.path.dirname(__file__)
        dirName = dirName[:dirName.rfind('\\')]
        for directorio in ruta.split('\\'):
            dirName = os.path.join(dirName, directorio)
            if not os.path.exists(dirName):
                os.mkdir(dirName)
                print("Directory ", dirName, " Created ")
            else:
                print("Directory ", dirName, " already exists")
        return dirName
