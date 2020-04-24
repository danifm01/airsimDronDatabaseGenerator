import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm


class DataOrganizer:
    def __init__(self, imagenes: list = None, imagenesMarcadas: list = None,
                 imagenesBounding: list = None, parametros: list = None):
        if imagenes is None:
            imagenes = []
        if imagenesMarcadas is None:
            imagenesMarcadas = []
        if imagenesBounding is None:
            imagenesBounding = []
        if parametros is None:
            parametros = []
        self.imagenes = imagenes
        self.imagenesMarcadas = imagenesMarcadas
        self.imagenesBounding = imagenesBounding
        self.parametros = parametros
        self.imaNames = {
            0: "Scene",
            1: "DepthPlanner",
            2: "DepthPerspective",
            3: "DepthVis",
            4: "Segmentation",
        }

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
        baseDir = 'Data\\Imagenes'
        entorno = 'Blocks'
        for key in self.imaNames:
            self.crearDirectorio(f'{baseDir}\\{self.imaNames[key]}')
        baseDir = self.crearDirectorio(baseDir)
        ultima = self.comprobarUltimaImagen(f'{baseDir}\\{self.imaNames[0]}')
        for index, setIma in enumerate(tqdm(self.imagenes)):
            for i, ima in enumerate(setIma):
                nombre = f'{index + 1 + ultima}_{i}_Imagen_{entorno}'
                # Imagen de la escena
                if i == 0:
                    imaBGR = cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(baseDir + "\\" + self.imaNames[i],
                                     f'{nombre}.png'),
                        imaBGR)
                    nombres.append(nombre)
                # Guardar np array de profundidad en perspectiva y planar
                elif i == 1 or i == 2:
                    np.save(os.path.join(baseDir + "\\" + self.imaNames[
                        i], f'{nombre}.npy'), ima)
                # Guarda las imágenes de profundidad para visión y segmentación
                else:
                    cv2.imwrite(
                        os.path.join(baseDir + "\\" + self.imaNames[i],
                                     f'{nombre}.png'),
                        ima)

        dirName = self.crearDirectorio('Data\\ImagenesMarcadas')
        for index, setIma in enumerate(tqdm(self.imagenesMarcadas)):
            nombre = str(index) + '_Imagen_Marcada_Blocks' + '.png'
            imaBGR = cv2.cvtColor(setIma, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dirName, nombre), imaBGR)

        dirName = self.crearDirectorio('Data\\ImagenesBoundingBox')
        for index, setIma in enumerate(tqdm(self.imagenesBounding)):
            nombre = str(index) + '_Imagen_Marcada_Blocks' + '.png'
            imaBGR = cv2.cvtColor(setIma, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dirName, nombre), imaBGR)

        data = pd.DataFrame(np.array(self.parametros),
                            columns=['distancia', 'phi', 'theta', 'xIma',
                                     'yIma', 'radioIma', 'x1BB', 'y1BB',
                                     'x2BB', 'y2BB',
                                     'orientacionVisor_x',
                                     'orientacionVisor_y',
                                     'orientacionVisor_z',
                                     'orientacionVisor_w',
                                     'orientacionVisto_x',
                                     'orientacionVisto_y',
                                     'orientacionVisto_z',
                                     'orientacionVisto_w',
                                     'orientacionRelativa_x',
                                     'orientacionRelativa_y',
                                     'orientacionRelativa_z',
                                     'orientacionRelativa_w'])
        data.insert(0, 'nombre', nombres)
        dirName = self.crearDirectorio('Data\\Parametros')
        if os.path.exists(os.path.join(dirName, f'Parametros_{entorno}.csv')):
            data.to_csv(os.path.join(dirName, f'Parametros_{entorno}.csv'),
                        mode="a", index=False, header=False)
        else:
            data.to_csv(os.path.join(dirName, f'Parametros_{entorno}.csv'),
                        mode="a", index=False, header=True)
        return data

    @staticmethod
    def comprobarUltimaImagen(path):
        numeros = []
        for ima in os.listdir(path):
            numeros.append(int(ima.split("_", 1)[0]))
        if not numeros:
            return -1
        return max(numeros)

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
