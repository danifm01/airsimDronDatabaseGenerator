import numpy as np
from transforms3d.quaternions import quat2mat


class DataCalculator:
    def __init__(self, cliente, dronVisor, dronVisto):
        self.client = cliente
        self.dronVisor = dronVisor
        self.dronVisto = dronVisto
        self.poseVisor = None
        self.poseVisto = None
        self.updatePose()

    def updatePose(self):
        self.poseVisor = self.client.simGetVehiclePose(self.dronVisor)
        self.poseVisto = self.client.simGetVehiclePose(self.dronVisto)

    def calcularDistancia(self):
        return self.poseVisor.position.distance_to(self.poseVisto.position)

    def orientacionAbsolutaVisor(self):
        return self.poseVisor.orientation

    def orientacionAbsolutaVisto(self):
        return self.poseVisto.orientation

    def orientacionRelativaVisto(self):
        qr1 = self.poseVisto.orientation.w_val
        qx1 = self.poseVisto.orientation.x_val
        qy1 = self.poseVisto.orientation.y_val
        qz1 = self.poseVisto.orientation.z_val
        # Inversión del quaternio del dron visor
        qr2 = self.poseVisor.orientation.w_val
        qx2 = -self.poseVisor.orientation.x_val
        qy2 = -self.poseVisor.orientation.y_val
        qz2 = -self.poseVisor.orientation.z_val
        # Composición de ambos cuaternios
        w_quat = qr1 * qr2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        x_quat = qr1 * qx2 + qr2 * qx1 + qy1 * qz2 - qy2 * qz1
        y_quat = qr1 * qy2 + qr2 * qy1 + qz1 * qx2 - qz2 * qx1
        z_quat = qr1 * qz2 + qr2 * qz1 + qx1 * qy2 - qx2 * qy1
        return w_quat, x_quat, y_quat, z_quat

    def calcularParametros(self):
        pass
