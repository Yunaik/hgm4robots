from pybullet_utils.bullet_client import BulletClient
import pybullet as _p
from pybullet import GUI, DIRECT, SHARED_MEMORY, LINK_FRAME, ER_BULLET_HARDWARE_OPENGL, \
    STATE_LOGGING_VIDEO_MP4, POSITION_CONTROL, VELOCITY_CONTROL, TORQUE_CONTROL, COV_ENABLE_RENDERING, \
    COV_ENABLE_GUI, COV_ENABLE_RGB_BUFFER_PREVIEW, COV_ENABLE_SHADOWS, COV_ENABLE_DEPTH_BUFFER_PREVIEW, \
    COV_ENABLE_SEGMENTATION_MARK_PREVIEW, COV_ENABLE_TINY_RENDERER, URDF_USE_INERTIA_FROM_FILE, URDF_USE_SELF_COLLISION, \
    JOINT_REVOLUTE, JOINT_PRISMATIC, ER_TINY_RENDERER
import os
import numpy as np
from numpy import pi


def loadPebble(_p, fixed_obstacles=False):
    obstacle_ids = []
    """Adjustable parameters"""
    amount_of_obstacles = 400
    x_width = 4.
    y_width = 4.

    urdfFlags = URDF_USE_SELF_COLLISION
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _p.setGravity(0, 0, -9.81)

    """Medium box"""
    for obstacle_no in range(amount_of_obstacles):
        x = (np.random.random()-0.5)*x_width
        y = (np.random.random()-0.5)*y_width
        z = np.random.random()*0.05 if not fixed_obstacles else 0.

        roll = np.random.random()*pi/2.
        pitch = np.random.random()*pi/2.
        yaw = np.random.random()*pi/2.

        quat = _p.getQuaternionFromEuler([roll, pitch, yaw])

        _ = _p.loadURDF(dir_path + "/urdf/mediumBox.urdf",
                        basePosition=[x, y, z],
                        baseOrientation=quat,
                        useFixedBase=fixed_obstacles,
                        flags=urdfFlags)
        obstacle_ids.append(_)

    return obstacle_ids


def loadV(_p, pitch_inclination=5, roll_inclination=0):
    urdfFlags = URDF_USE_SELF_COLLISION
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _p.setGravity(0, 0, -9.81)

    # Plane 1
    x = 1.25+0.25/2
    y = -1.
    z = np.sin(pitch_inclination*np.pi/180)*0.75/2

    roll = roll_inclination*pi/180
    pitch = -pitch_inclination*pi/180.
    yaw = 0
    print("Plane 1 loaded with: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]" % (
        x, y, z, roll, pitch, yaw))

    quat = _p.getQuaternionFromEuler([roll, pitch, yaw])

    plank1 = _p.loadURDF(dir_path + "/urdf/plane_obstacle2.urdf",
                         basePosition=[x, y, z],
                         baseOrientation=quat,
                         useFixedBase=True,
                         flags=urdfFlags)

    _p.changeDynamics(plank1, -1, lateralFriction=1.5)
    # Plane 2
    x = 2.25-0.25/2
    y = -1
    z = np.sin(pitch_inclination*np.pi/180)*0.75/2
    print("Z: %.2f" % z)

    roll = -roll_inclination*pi/180
    pitch = pitch_inclination*pi/180.
    yaw = 0.0

    quat = _p.getQuaternionFromEuler([roll, pitch, yaw])

    print("Plane 2 loaded with: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]" % (
        x, y, z, roll, pitch, yaw))
    plank2 = _p.loadURDF(dir_path + "/urdf/plane_obstacle2.urdf",
                         basePosition=[x, y, z],
                         baseOrientation=quat,
                         useFixedBase=True,
                         flags=urdfFlags)
    _p.changeDynamics(plank2, -1, lateralFriction=1.5)
    return plank1, plank2


def loadSlipPlate(_p, pos_x=0.0, pos_y=0.0, lateral_friction=0.7):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _p.setGravity(0, 0, -9.81)

    print("Slippery plane loaded with: [%.2f, %.2f]" % (pos_x, pos_y))

    plank1 = _p.loadURDF(dir_path + "/urdf/plane_obstacle.urdf",
                         basePosition=[pos_x, pos_y, 0.01],
                         useFixedBase=True)

    _p.changeDynamics(plank1, -1, lateralFriction=lateral_friction)

    return plank1


def loadSlab(_p):
    obstacle_ids = []

    urdfFlags = URDF_USE_SELF_COLLISION
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _p.setGravity(0, 0, -9.81)

    """Slab"""
    for x in range(10):
        x *= 0.4
        x -= 2
        for y in range(10):
            y *= 0.4
            y -= 2
            z = 0.

            roll = np.random.random()*40*pi/180-20*pi/180
            pitch = np.random.random()*40*pi/180-20*pi/180
            yaw = 0.

            quat = _p.getQuaternionFromEuler([roll, pitch, yaw])
            _ = _p.loadURDF(dir_path + "/urdf/slab.urdf",
                            basePosition=[x, y, z],
                            baseOrientation=quat,
                            useFixedBase=True,
                            flags=urdfFlags)
            obstacle_ids.append(_)
    return obstacle_ids


def loadPlank(_p, physics_freq, fixed_obstacles=False):
    obstacle_ids = []
    urdfFlags = URDF_USE_SELF_COLLISION
    dir_path = os.path.dirname(os.path.realpath(__file__))

    _p.setTimeStep(1./physics_freq)

    _p.setGravity(0, 0, -9.81)

    """Plank"""
    plank_amount = 25
    for y in range(plank_amount):
        x = (np.random.random()-0.5)*2  # -0.25
        y = (np.random.random()-0.5)*2  # -0.25
        z = np.random.random()*0.4

        roll = 0.
        pitch = 0.
        yaw = np.random.random()*180*pi/180-90*pi/180

        quat = _p.getQuaternionFromEuler([roll, pitch, yaw])

        plank = _p.loadURDF(dir_path + "/urdf/plank.urdf",
                            basePosition=[x, y, z],
                            baseOrientation=quat,
                            useFixedBase=fixed_obstacles,
                            flags=urdfFlags)
        obstacle_ids.append(plank)
        if fixed_obstacles:
            _p.changeDynamics(plank, -1, mass=20)
    _time = 0.
    print("Running simulation for a second to let the planks settle on ground")
    while _time < 1.0:
        _p.stepSimulation()
        _time += 1./physics_freq
    return obstacle_ids


def loadStep(_p):
    urdfFlags = URDF_USE_SELF_COLLISION
    dir_path = os.path.dirname(os.path.realpath(__file__))

    plane = _p.loadURDF(dir_path + "/plane/plane_obstacle.urdf")

    _p.setGravity(0, 0, -9.81)

    x = np.random.random()
    y = 0.
    z = np.random.random()*0.1

    roll = 0
    pitch = 0
    yaw = (np.random.random()-0.5)/2.

    quat = _p.getQuaternionFromEuler([roll, pitch, yaw])

    _ = _p.loadURDF(dir_path + "/urdf/plane_obstacle.urdf",
                    basePosition=[x, y, z],
                    baseOrientation=quat,
                    useFixedBase=True,
                    flags=urdfFlags)


def loadSeesaw1(_p, start_front=True):
    obstacle_ids = []
    urdfFlags = URDF_USE_SELF_COLLISION
    dir_path = os.path.dirname(os.path.realpath(__file__))

    """Seesaw"""
    x = 0.
    y = 0.
    z = 0.2

    roll = 0.
    pitch = -0.1 if start_front else 0.1
    yaw = 0.

    quat = _p.getQuaternionFromEuler([roll, pitch, yaw])

    _ = _p.loadURDF(dir_path + "/urdf/seesaw_link.urdf",
                    basePosition=[x, y, z],
                    baseOrientation=quat,
                    flags=urdfFlags)
    obstacle_ids.append(_)
    _time = 0.
    print("Running simulation for a second to let the seesaw settle on ground")
    while _time < 1.0:
        _p.stepSimulation()
        _time += 1./1000
    return obstacle_ids


def loadSeesaw2(_p, start_front=True):
    urdfFlags = URDF_USE_SELF_COLLISION
    dir_path = os.path.dirname(os.path.realpath(__file__))

    """Seesaw"""
    x = 0.01 if start_front else -0.01
    y = 0.
    z = 0.1

    roll = 0.
    pitch = 0.1 if start_front else -0.1
    yaw = 0.

    quat = _p.getQuaternionFromEuler([roll, pitch, yaw])

    _ = _p.loadURDF(dir_path + "/urdf/seesaw_bottom.urdf",
                    basePosition=[0, 0, 0.],
                    baseOrientation=quat,
                    useFixedBase=True,
                    flags=urdfFlags)

    _ = _p.loadURDF(dir_path + "/urdf/seesaw_top.urdf",
                    basePosition=[x, y, z],
                    baseOrientation=quat,
                    flags=urdfFlags)
