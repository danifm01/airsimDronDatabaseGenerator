import numpy as np


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

    def calcularParametros(self):
        pass
