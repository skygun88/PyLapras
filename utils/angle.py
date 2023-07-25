import numpy as np
from typing import Any, Dict, List, Tuple

def calculate_difference(robot_loc: Tuple[float, float], user_loc: Tuple[float, float])->float:
    np_robot = np.array(robot_loc)
    np_user = np.array(user_loc)
    
    vector = np_user - np_robot
    unit = np.array([1, 0])
    ang1 = np.arctan2(*vector[::-1])
    ang2 = np.arctan2(*unit[::-1])
    angle_360 = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    result_angle = angle_360 if angle_360 <= 180 else angle_360 - 360
    return result_angle

def calculate_rotation(robot_angle: float, user_angle: float)->int:
    robot_360 = robot_angle if robot_angle > 0 else robot_angle + 360
    user_360 = user_angle if user_angle > 0 else user_angle + 360

    difference = user_360 - robot_360
    result_angle = difference if difference <= 180 else difference - 360
    result_angle = result_angle if result_angle > -180 else result_angle + 360

    return int(result_angle)

def calculate_distance(robot_loc: Tuple[float, float], user_loc: Tuple[float, float])->float:
    np_robot = np.array(robot_loc)
    np_user = np.array(user_loc)
    
    distance = np.sqrt(np.sum(np.power(np_user-np_robot, 2)))
    return float(distance)