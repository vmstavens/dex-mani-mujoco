import numpy as np
import mujoco as mj
from typing import List, Tuple
from math import radians

class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.__x = x
        self.__y = y
        self.__z = z

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        """Set the x-coordinate."""
        self.__x = value

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        """Set the y-coordinate."""
        self.__y = value

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, value):
        """Set the z-coordinate."""
        self.__z = value

    def __repr__(self) -> str:
        return "(x={:.2f}, y={:.2f}, z={:.2f})".format(self.x, self.y, self.z)

    def numpy(self,dtype=np.float64):
        return np.array([self.x, self.y, self.z],dtype=dtype)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'Vector3D':
        """Create a Position from a numpy array."""
        assert array.size == 3, "Input array must have three elements for position representation"

        # Assuming the order of elements in the array is [x, y, z]
        x, y, z = array

        return cls(x=x, y=y, z=z)

    def __iter__(self):
        """Return an iterable for the Position coordinates."""
        return iter((self.__x, self.__y, self.__z))

    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        """Add two vectors element-wise."""
        return Vector3D(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z
        )

class Quaternion:
    def __init__(self, w: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 1.0):
        self.__x: float = x
        self.__y: float = y
        self.__z: float = z
        self.__w: float = w

    @property
    def x(self) -> float:
        """Get the x component of the quaternion."""
        return self.__x

    @x.setter
    def x(self, value: float) -> None:
        """Set the x component of the quaternion."""
        self.__x = value

    @property
    def y(self) -> float:
        """Get the y component of the quaternion."""
        return self.__y

    @y.setter
    def y(self, value: float) -> None:
        """Set the y component of the quaternion."""
        self.__y = value

    @property
    def z(self) -> float:
        """Get the z component of the quaternion."""
        return self.__z

    @z.setter
    def z(self, value: float) -> None:
        """Set the z component of the quaternion."""
        self.__z = value

    @property
    def w(self) -> float:
        """Get the w component of the quaternion."""
        return self.__w

    @w.setter
    def w(self, value: float) -> None:
        """Set the w component of the quaternion."""
        self.__w = value

    def __repr__(self) -> str:
        """String representation of the quaternion."""
        return "(w={:.2f}, x={:.2f}, y={:.2f}, z={:.2f})".format(self.w, self.x, self.y, self.z)

    def numpy(self):
        """Convert the quaternion to a numpy array."""
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'Quaternion':
        """Create a Quaternion from a numpy array."""
        assert len(array) == 4, "Input array must have four elements for quaternion representation"

        # Assuming the order of elements in the array is [w, x, y, z]
        w, x, y, z = array

        return cls(w=w, x=x, y=y, z=z)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Multiply two quaternions."""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w

        return Quaternion(w=w, x=x, y=y, z=z)

    def conjugate(self) -> 'Quaternion':
        """
        Calculate the conjugate of a quaternion.

        Parameters:
            q (Quaternion): Input quaternion [w, x, y, z].

        Returns:
            Quaternion: Conjugate quaternion.
        """
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)

    @classmethod
    def from_axis_angle(cls, axis: Vector3D, angle: float) -> 'Quaternion':
        """
        Create a quaternion from an axis and an angle.

        Parameters:
            axis (List[float]): Axis of rotation [x, y, z].
            angle (float): Angle of rotation in radians.

        Returns:
            Quaternion: Quaternion representing the rotation.
        """

        axis = axis.numpy()
        axis /= np.linalg.norm(axis)  # Normalize the axis

        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        w = cos_half
        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half

        return cls(w=w, x=x, y=y, z=z)

class Pose:
    def __init__(self, pos: Vector3D, quat: Quaternion):
        self.__position = pos
        self.__quaternion = quat
    @property
    def position(self) -> Vector3D:
        """Get the position of the pose."""
        return self.__position

    @position.setter
    def position(self, value: Vector3D) -> None:
        """Set the position of the pose."""
        self.__position = value

    @property
    def quaternion(self) -> Quaternion:
        """Get the quaternion of the pose."""
        return self.__quaternion

    @quaternion.setter
    def quaternion(self, value: Quaternion) -> None:
        """Set the quaternion of the pose."""
        self.__quaternion = value

    def __repr__(self) -> str:
        return f"Pose( pos = {self.position}, quat = {self.quaternion} )"
    
    def numpy(self):
        return np.array([
            self.position.x, 
            self.position.y, 
            self.position.z,
            self.quaternion.w,
            self.quaternion.x,
            self.quaternion.y,
            self.quaternion.z])

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'Pose':
        """Create a Pose from a numpy array."""
        assert array.size == 7, "Input array must have four elements for pose (pos, quat) representation"
        
        # Assuming the order of elements in the array is [w, x, y, z]
        x, y, z, qw, qx, qy, qz = array
        pos = Vector3D(x=x,y=y,z=z)
        quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
        return cls(pos=pos,quat=quat)

    def lift(self, z_lift: float) -> 'Pose':
        """Lift the pose by changing the z-coordinate."""
        new_position = Vector3D(x=self.position.x, y=self.position.y, z=self.position.z + z_lift)
        return Pose(pos=new_position, quat=self.quaternion)

    def rot_x(self, angle: float, in_degrees: bool = False) -> 'Pose':
        """Rotate the pose around the local x-axis."""
        if in_degrees:
            angle = radians(angle)

        # Rotate the global x-axis by the current orientation to get the local x-axis
        local_x_axis = rotate_vector(Vector3D(1, 0, 0), self.quaternion)

        # Create a quaternion representing the rotation around the local x-axis
        rotation_quaternion = Quaternion.from_axis_angle(local_x_axis, angle)

        # Rotate the position in the local coordinate system
        rotated_position = rotate_vector(self.position, rotation_quaternion)

        # Update the position and quaternion
        new_position = Vector3D(x=rotated_position.x, y=rotated_position.y, z=rotated_position.z)
        new_quaternion = self.quaternion * rotation_quaternion

        # Return a new Pose with the updated position and quaternion
        return Pose(pos=new_position, quat=new_quaternion)

    def rot_y(self, angle: float, in_degrees: bool = False) -> 'Pose':
        """Rotate the pose around the y-axis."""
        if in_degrees:
            angle = radians(angle)

        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        rotated_position = Vector3D.from_numpy(np.dot(rotation_matrix, self.position.numpy()))

        # Update quaternion accordingly
        rotation_quaternion = rpy_to_quaternion(roll=0, pitch=angle, yaw=0)
        new_quaternion = self.quaternion * rotation_quaternion

        # Return a new Pose with the updated position and quaternion
        return Pose(pos=Vector3D(x=rotated_position.x, y=rotated_position.y, z=rotated_position.z),
                    quat=new_quaternion)

    def rot_z(self, angle: float, in_degrees: bool = False) -> 'Pose':
        """Rotate the pose around the z-axis."""
        if in_degrees:
            angle = radians(angle)

        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_position = Vector3D.from_numpy(np.dot(rotation_matrix, self.position.numpy()))

        # Update quaternion accordingly
        rotation_quaternion = rpy_to_quaternion(roll=0, pitch=0, yaw=angle)
        new_quaternion = self.quaternion * rotation_quaternion

        # Return a new Pose with the updated position and quaternion
        return Pose(pos=Vector3D(x=rotated_position.x, y=rotated_position.y, z=rotated_position.z),
                    quat=new_quaternion)

    @classmethod
    def copy(cls, pose: 'Pose'):
        return cls(pos=pose.position, quat=pose.quaternion)

    def translate(self, delta_position: Vector3D) -> 'Pose':
        """Translate the pose by a given delta_position."""
        new_position = self.position + delta_position
        # return Pose(pos=new_position, quat=Quaternion(w=1.0,x=0.0,y=0.0,z=0.0))
        return Pose(pos=new_position, quat=self.quaternion)

    def rotate(self, delta_orientation: Quaternion) -> 'Pose':
        """Rotate the pose by a given delta_orientation."""
        new_quaternion = delta_orientation * self.quaternion
        return Pose(pos=self.position, quat=new_quaternion)

    def relative_transform(self, delta_position: Vector3D, delta_orientation: Quaternion) -> 'Pose':
        """Apply both translation and rotation as relative transformations."""
        new_position = self.position + delta_position
        new_quaternion = delta_orientation * self.quaternion
        return Pose(pos=new_position, quat=new_quaternion)

def normalize_quaternion(q: Quaternion) -> Quaternion:
    """
    Normalize a quaternion.

    Parameters:
        q (numpy.ndarray): Quaternion [w, x, y, z].

    Returns:
        Quaternion: Normalized quaternion.
    """

    q = q.numpy()

    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Cannot normalize a quaternion with zero norm.")
    
    q = q / norm

    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])

def quaternion_slerp(q1: Quaternion, q2: Quaternion, alpha: float) -> Quaternion:
    """
    Spherical Linear Interpolation (slerp) between two quaternions.
    """

    q1 = q1.numpy()
    q2 = q2.numpy()

    dot_product = np.dot(q1, q2)

    # Ensure shortest path by flipping one quaternion
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta_0 = np.arccos(dot_product)
    
    # Handle the case where theta is close to zero
    if np.abs(theta_0 * alpha) < 1e-5:
        q_interpolated = (1.0 - alpha) * q1 + alpha * q2
    else:
        theta = theta_0 * alpha
        try:
            # Attempt to calculate slerp
            q_interpolated = (np.sin((1 - alpha) * theta) * q1 + np.sin(alpha * theta) * q2) / np.sin(theta)
        except ZeroDivisionError:
            # If np.sin(theta) is close to zero, use linear interpolation as a fallback
            q_interpolated = (1.0 - alpha) * q1 + alpha * q2

    return normalize_quaternion(Quaternion.from_numpy(q_interpolated))

def quaternion_to_rpy(q: Quaternion) -> Tuple[float,float,float]:
    """
    Convert quaternion to roll-pitch-yaw angles.

    Parameters:
        q (numpy.ndarray): Quaternion [qx, qy, qz, qw].

    Returns:
        numpy.ndarray: Roll-pitch-yaw angles [roll, pitch, yaw].
    """

    q = q.numpy()

    # Extract quaternion components
    qx, qy, qz, qw = q

    # Calculate roll (x-axis rotation)
    roll_x = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    # Calculate pitch (y-axis rotation)
    sin_pitch = 2 * (qw * qy - qz * qx)
    pitch_y = np.arcsin(np.clip(sin_pitch, -1, 1))

    # Calculate yaw (z-axis rotation)
    yaw_z = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    return (roll_x, pitch_y, yaw_z)

def rpy_to_quaternion(roll:float = 0.0, pitch:float = 0.0, yaw:float = 0.0) -> Quaternion:
    """
    Convert roll-pitch-yaw angles to quaternion.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return Quaternion(w=qw,x=qx,y=qy,z=qz)

def get_pose(name:str, model: mj.MjModel, data: mj.MjData) -> Pose:
    
    if name == "right_shadow_hand":
        current_pose = data.qpos[:7]

        current_pos = Vector3D(
            x = current_pose[0],
            y = current_pose[1],
            z = current_pose[2])

        current_ori = Quaternion(
            x = current_pose[3],
            y = current_pose[4],
            z = current_pose[5],
            w = current_pose[6])

        return Pose(pos = current_pos, quat = current_ori)

    # Find the object ID by name
    obj_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)

    if obj_id == -1:
        raise ValueError(f"Object with name '{name}' not found.")

    # Get the object pose from the simulation data
    obj_pose = data.xpos[obj_id], data.xquat[obj_id]

    # Extract position and orientation
    pos = obj_pose[0]
    quat = obj_pose[1]

    position = Vector3D(x = pos[0], y = pos[1], z = pos[2])
    
    quaternion = Quaternion(w = quat[0], x = quat[1], y = quat[2], z = quat[3])

    return Pose(pos = position, quat = quaternion)

def generate_pose_trajectory(start_pose: Pose, end_pose: Pose, n_steps:int ) -> List[Pose]:
    poses = []
    steps = np.linspace(0,1,n_steps)
    # poses.append(start_pose)
    for i in steps:
        interpolated_quaternion = quaternion_slerp(
            q1    = start_pose.quaternion, 
            q2    = end_pose.quaternion, 
            alpha = i)

        x = start_pose.position.x + i * (end_pose.position.x - start_pose.position.x)
        y = start_pose.position.y + i * (end_pose.position.y - start_pose.position.y)
        z = start_pose.position.z + i * (end_pose.position.z - start_pose.position.z)

        p = Vector3D(x=x, y=y, z=z)

        pose = Pose(pos=p, quat=interpolated_quaternion)
        poses.append( pose ) 
        # poses.append( list( np.append([x,y,z],interpolated_quaternion) ) )

    # poses.append(end_pose)
    return poses

def rotate_vector(vector: Vector3D, quaternion: Quaternion) -> Vector3D:
    """Rotate a vector using a quaternion."""
    # Convert the vector to a quaternion
    vector_quaternion = Quaternion(x=vector.x, y=vector.y, z=vector.z)

    # Rotate the vector using quaternion multiplication
    rotated_vector_quaternion = quaternion * vector_quaternion * quaternion.conjugate()

    # Return the rotated vector as a numpy array
    return Vector3D(
        x = rotated_vector_quaternion.x,
        y = rotated_vector_quaternion.y,
        z = rotated_vector_quaternion.z)