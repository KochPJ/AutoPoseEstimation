import numpy as np

# define your own robot client here or import
class UR5RobotClient:
	def __init__(self, ip):
		raise NotImplemented

# define your gripper client here or import
class GripperClient:
	def __init__(self, ip):
		raise NotImplemented

	def open(self):
		raise NotImplemented

	def close(self):
		raise NotImplemented

class RobotController:
	
	def __init__(self):
		self.robotClient = UR5RobotClient('')
		self.gripperClient = GripperClient('')

	def move_joints(self, target, moveType="p", vel=0.1, acc=0.1):
		self.robotClient.GripperClient(target, moveType, vel, acc)
		
	def get_pose(self, return_mm = True):
		currentPose = self.robotClient.GetCurrentCartPose()
		if return_mm:
			currentPose['x'] = currentPose['x']*1000
			currentPose['y'] = currentPose['y']*1000
			currentPose['z'] = currentPose['z']*1000
		return currentPose

	def is_moving(self):
		return self.robotClient.IsRobotMoving()

	def get_joints(self, type='deg'):
		if type=='deg':
			currentJoins = self.robotClient.GetCurrentJointPose()
			currentJoints = np.degrees(currentJoins)
		elif type=='rad':
			currentJoints = self.robotClient.GetCurrentJointPose()
		else:
			print('get_joints: wrong type')
			currentJoints = -1

		return currentJoints
						
	def move_robot_to_coord_origin(self, coord_system="kitting_box2"):
		tfmat = np.array([
			[ 1, 0, 0, 0 ],
			[ 0, 1, 0, 0 ],
			[ 0, 0, 1, 0 ],
			[ 0, 0, 0, 1 ]
		], dtype=np.float64)

		target_pose = self.tfClient.GetTransformMatrix("robot", coord_system, tfmat)
		
		currentPose = self.robotClient.GetCurrentCartPose()
		print("current pose = ")
		print(currentPose)
		
		currentPose['x'] = target_pose[0, 3]
		currentPose['y'] = target_pose[1, 3]
		#currentPose.z = 0.05
		
		print("target pose = ")
		print(currentPose)

		self.robotClient.SetPoseTarget(currentPose)

	def is_home(self, eps=0.02):
		j = self.get_joints()
		t = np.array([0,-90,0,-90,0,0])
		d = np.abs(t-j)
		home = True
		for q in d:
			if q > eps:
				home = False
				break

		return home

	def at_target(self, t, type='deg', eps=0.02):
		j = self.get_joints(type=type)
		if t[0]+eps > j[0] > t[0]-eps and \
			j[1] < t[1] + eps and j[1] > t[1] - eps and \
			j[2] < t[2] + eps and j[2] > t[2] - eps and \
			j[3] < t[3] + eps and j[3] > t[3] - eps and \
			j[4] < t[4] + eps and j[4] > t[4] - eps and \
			j[5] < t[5] + eps and j[5] > t[5] - eps:
			at_target = True
		else:
			at_target = False
		return at_target

	def close_gripper(self):
		self.gripperClient.close()

	def open_gripper(self):
		self.gripperClient.open()


