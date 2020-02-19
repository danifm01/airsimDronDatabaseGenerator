from unittest import TestCase
import airsim
import DroneController
import DatabaseGenerator


class TestDroneController(TestCase):
    pose0 = airsim.Pose()
    cliente = DatabaseGenerator.crearCliente()
    dron = DroneController.DroneController('Drone1', cliente)

    def assertPose(self, poseEsperada, pose):
        self.assertAlmostEqual(poseEsperada.position.x_val,
                               pose.position.x_val, delta=1e-10)
        self.assertAlmostEqual(poseEsperada.position.y_val,
                               pose.position.y_val, delta=1e-10)
        self.assertAlmostEqual(poseEsperada.position.z_val,
                               pose.position.z_val, delta=1e-10)
        self.assertAlmostEqual(poseEsperada.orientation.w_val,
                               pose.orientation.w_val, delta=1e-10)
        self.assertAlmostEqual(poseEsperada.orientation.x_val,
                               pose.orientation.x_val, delta=1e-10)
        self.assertAlmostEqual(poseEsperada.orientation.y_val,
                               pose.orientation.y_val, delta=1e-10)
        self.assertAlmostEqual(poseEsperada.orientation.z_val,
                               pose.orientation.z_val, delta=1e-10)

    def test_calcular_pose_relativa(self):
        pose = self.dron.calcularPoseRelativa(5., 0., 0., self.pose0)
        self.assertPose(airsim.Pose(airsim.Vector3r(5, 0, 0)), pose)
        pose = self.dron.calcularPoseRelativa(5., 90, 0., self.pose0)
        self.assertPose(airsim.Pose(airsim.Vector3r(0, 5, 0)), pose)
        pose = self.dron.calcularPoseRelativa(5., 90, 90., self.pose0)
        self.assertPose(airsim.Pose(airsim.Vector3r(0, 0, 5)), pose)
