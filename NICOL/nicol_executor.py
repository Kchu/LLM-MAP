#!/usr/bin/env python
"""
Implementation of Large Language Model Coordinator for NICOL Bimanual Robot
Copyright 2024, Planet Earth
"""
import numpy as np
import math
from transforms3d.euler import euler2quat, quat2euler
import time

from nicol_api.nicol_env import NicolFactory
from nicol_api.base import NicolPose
from coppeliasim_zmqremoteapi_client import *

import argparse, json, time
import scipy.spatial.distance as dist

############################ Initialize ######################################
nicol = NicolFactory().create_nicol('coppelia', scene="./nicol.ttt", start_scene=False, talker=False)
head  = nicol.head()    # 3D-Printed head structure
left  = nicol.left()    # Left  OpenManipulator + RH8D
right = nicol.right()   # Right OpenManipulator + RH8D
sim = nicol.nicol_adapter.sim
left_sensor = sim.getObject('/l_sensor')
left_attachpoint = sim.getObject('/l_palm_attachPoint')
right_sensor = sim.getObject('/r_sensor')
right_attachpoint = sim.getObject('/r_palm_attachPoint')
client = nicol.nicol_adapter.client
# print("\n"*30, "#"*114, '\n', "#"*114, '\n', 'The simulation starts!')

# Set up useful dicts and global varibles.
############################################################################################################################################################
ORIENTATION_DICT_LEFT = {
    'Horizontally_Up': euler2quat(0, np.pi/2, np.pi/2),
    'Horizontally_Down': euler2quat(np.pi, -np.pi/2, np.pi/2),
    'Vertical': euler2quat(np.pi/2, np.pi, np.pi),
    'Horizontally_Slanted_Up': euler2quat(2*np.pi/3, 5*np.pi/6, np.pi),
    'Horizontally_Slanted_Down': euler2quat(-np.pi, -1*np.pi/4, np.pi*5/8),
    'Vertically_Slanted': euler2quat(np.pi/2, np.pi, np.pi*4/3),
    'Vertically_Down': euler2quat(8*np.pi/8, -np.pi/8, 4*np.pi/8)}
ORIENTATION_DICT_RIGHT = {
    'Horizontally_Up': euler2quat(0, np.pi/2, np.pi/2),
    'Horizontally_Down': euler2quat(np.pi, -np.pi/2, np.pi/2),
    'Vertical': euler2quat(np.pi/2, 0, np.pi),
    'Horizontally_Slanted_Up': euler2quat(2*np.pi/8, np.pi/6, np.pi),
    'Horizontally_Slanted_Down': euler2quat(np.pi, -np.pi*6/8, np.pi*5/8),
    'Vertically_Slanted': euler2quat(np.pi/2, 0, np.pi*2/3),
    'Vertically_Down': euler2quat(8*np.pi/8, 0, 3*np.pi/8)}
RIGHT_FINGER_STAT, RIGHT_HAND_STAT = 'Open', 'Vertical'
LEFT_FINGER_STAT, LEFT_HAND_STAT = 'Open', 'Vertical'
############################################################################################################################################################

def reset_global():
    global RIGHT_FINGER_STAT
    global LEFT_FINGER_STAT
    RIGHT_FINGER_STAT = LEFT_FINGER_STAT = 'Open'
    global LEFT_HAND_STAT
    global RIGHT_HAND_STAT
    LEFT_HAND_STAT = RIGHT_HAND_STAT = 'Vertical'

# NICOL Skill Functions 
############################################################################################################################################################

def look_at(obj_name:str):
    if obj_name =="left":
        head.set_pose_target(NicolPose([0.7, 0.2, 0.9], [0,0,0,0]))
    elif obj_name =="right":
        head.set_pose_target(NicolPose([0.7, -0.2, 0.9], [0,0,0,0]))
    elif obj_name =="center":
        head.set_pose_target(NicolPose([0.7, 0, 0.9], [0,0,0,0]))
    else:
        human_pose = sim.getObjectPosition(sim.getObject('/Bill'), sim.handle_world)
        new_pose = [human_pose[0], human_pose[1], human_pose[2]+1.5]
        obj_name = "human user"
        head.set_pose_target(NicolPose(new_pose, [0,0,0,0]))
    return f"The NICOL has looked at the {obj_name}."

# move both hands simultaneously
#######################################################################################################################
def move_both_to_poses(obj_name:str):
    handle = sim.getObject('/'+obj_name)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)
    left_pose = left.get_eef_pose().position.as_list()
    right_pose = right.get_eef_pose().position.as_list()
    mid_point = [(left_pose[i]+right_pose[i])/2 for i in range(0,3)]
    left_pose_target = [(left_pose[i]+ new_pose[i]-mid_point[i]) for i in range(0, 3)]
    right_pose_target = [(right_pose[i]+ new_pose[i]-mid_point[i]) for i in range(0, 3)]
    left_ori = left.get_eef_pose().orientation
    left_ori = [left_ori.x, left_ori.y, left_ori.z, left_ori.w]
    right_ori = right.get_eef_pose().orientation
    right_ori = [right_ori.x, right_ori.y, right_ori.z, right_ori.w]
    steps = 0
    right_step_pose = left_pose
    left_step_pose = left_pose
    step_length = 0.06
    steps = 0
    for i in range(0, 3):
        if math.floor((left_pose_target[i]-left_pose[i])/step_length) > steps:
            steps = math.floor((left_pose_target[i]-left_pose[i])/step_length)
    head.set_pose_target(NicolPose([new_pose[0], new_pose[1], new_pose[2]+0.5], [0,0,0,0]))
    for i in range(steps):
        right_step_pose = [min(right_pose_target[0], right_pose[0]+i*step_length), min(right_pose_target[1], right_pose[1]+i*step_length), min(right_pose_target[2], right_pose[2]+i*step_length)]
        left_step_pose = [min(left_pose_target[0], left_pose[0]+i*step_length), min(left_pose_target[1], left_pose[1]+i*step_length), min(left_pose_target[2], left_pose[2]+i*step_length)]
        nicol.set_pose_target_for_both_arms(NicolPose(left_step_pose, left_ori), NicolPose(right_step_pose,right_ori))
        nicol.set_pose_target_for_both_arms(NicolPose(left_step_pose, left_ori), NicolPose(right_step_pose,right_ori))
    nicol.set_pose_target_for_both_arms(NicolPose(left_pose_target, left_ori), NicolPose(right_pose_target,right_ori))
    nicol.set_pose_target_for_both_arms(NicolPose(left_pose_target, left_ori), NicolPose(right_pose_target,right_ori))
    result = f"The NICOL robot's both hands have moved simultaneously to the new positon where the {obj_name} is at the middle point of two hands."
    return result

# side grasp
#######################################################################################################################
def side_grasp(side: str, obj_name: str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    if side == 'left':
        hand_stat = LEFT_HAND_STAT
        finger_stat = LEFT_FINGER_STAT
        hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        hand_stat = RIGHT_HAND_STAT
        finger_stat = RIGHT_FINGER_STAT
        hand_finger_info = ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    handle = sim.getObject('/'+obj_name)
    handle_ori = sim.getObjectOrientation(handle, sim.handle_world)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)
    Sensor, AttachPoint, Controller = (left_sensor, left_attachpoint, left) if side == 'left' else (right_sensor, right_attachpoint, right)
    if (side=="right" and new_pose[1]>0.3) or (side=="left" and new_pose[1]<-0.3): 
        return f"The robot's {side} hand failed to move to {obj_name}, out of its area." + hand_finger_info
    # if sim.checkProximitySensor(Sensor, sim.handle_world)[0]:
    #     return f"The {side} hand failed to grasp {obj_name}, as the {obj_name} is grasped by the other hand." + hand_finger_info
    detected = sim.checkProximitySensor(Sensor, sim.handle_all)[0]
    if finger_stat == 'Closed' and detected:
        return f"The {side} hand failed to grasp {obj_name}, as the left hand is already occupied." + hand_finger_info
    if hand_stat == 'Hold_Up': release(side)
    if "bowl" in obj_name: return "failed. The bowl is not allowed to be grasped!" + hand_finger_info
    result = move_single_to_pose(side, target = obj_name, obj_name=None, ori_angle='Vertical', off_set=None)
    if 'failed' in result:
        return result + hand_finger_info
    quat = [[np.pi, - np.pi, - np.pi, - np.pi, - np.pi], 
            [np.pi, - np.pi, - np.pi, - np.pi, - np.pi], 
            [np.pi, -2.9, -1.8, -1.8, -1.8]]
            # [np.pi, - 2.8, - 1.4, - 1.4, - 1.4]]
    detected = sim.checkProximitySensor(Sensor, sim.handle_all)[0]
    if detected:
        read_result = sim.readProximitySensor(Sensor)
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, AttachPoint, True)
        obj_name_grasped = sim.getObjectAlias(read_result[3], -1)
        if '_respondable' in obj_name_grasped:
            obj_name_grasped = obj_name_grasped.replace('_respondable', '')
        for sub_quat in quat:
            Controller.set_joint_position_for_hand(sub_quat, block=True)
        if side == 'left':
            LEFT_HAND_STAT, LEFT_FINGER_STAT = 'Vertical', 'Closed'
        else:
            RIGHT_HAND_STAT, RIGHT_FINGER_STAT = 'Vertical', 'Closed'
        result = f"The robot's {side} hand has grasped {obj_name_grasped}, and is holding the {obj_name_grasped}."
    else:
        result = f"The robot's {side} hand failed to grasp anything, as there is no object detected."
    time.sleep(0.5)
    sim.setObjectOrientation(handle, -1, handle_ori)
    if side == 'left':
        hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        hand_finger_info = ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    return result + hand_finger_info

# top grasp
#######################################################################################################################
def top_grasp(side: str, obj_name: str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    if side == 'left':
        hand_stat = LEFT_HAND_STAT
        finger_stat = LEFT_FINGER_STAT
        hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        hand_stat = RIGHT_HAND_STAT
        finger_stat= RIGHT_FINGER_STAT
        hand_finger_info = ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    handle = sim.getObject('/'+obj_name)
    handle_ori = sim.getObjectOrientation(handle, sim.handle_world)
    new_pose = sim.getObjectPosition(handle, sim.handle_world)
    Sensor, AttachPoint, Controller = (left_sensor, left_attachpoint, left) if side == 'left' else (right_sensor, right_attachpoint, right)
    # if sim.checkProximitySensor(Sensor, handle)[0]:
    #     return f"The {side} hand failed to grasp {obj_name}, as the {obj_name} is grasped by the other hand." + hand_finger_info
    if (side=="right" and new_pose[1]>0.3) or (side=="left" and new_pose[1]<-0.3): 
        return f"The robot's {side} hand failed to move to {obj_name} (with pose: {new_pose}), out of its area. " + hand_finger_info
    detected = sim.checkProximitySensor(Sensor, sim.handle_all)[0]
    if finger_stat == 'Closed' and detected:
        return f"The {side} hand failed to grasp {obj_name}, as the {side} hand is already occupied." + hand_finger_info
    if side == "left":
        right_detected = sim.checkProximitySensor(right_sensor, handle)[0]
        if right_detected: return f"The {side} hand failed to grasp {obj_name}, as it is still grasped by another hand!" + hand_finger_info
    if side == "right":
        left_detected = sim.checkProximitySensor(left_sensor, handle)[0]
        if left_detected: return f"The {side} hand failed to grasp {obj_name}, as it is still grasped by another hand!" + hand_finger_info
    if hand_stat == 'Hold_Up': release(side)
    if "bowl" in obj_name: return "failed. The bowl is not allowed to be grasped!" + hand_finger_info
    result = move_single_to_pose(side, target = obj_name, obj_name=None, ori_angle='Horizontally_Down', off_set="up")
    result = move_single_to_pose(side, target = obj_name, obj_name=None, ori_angle='Horizontally_Down', off_set=None)

    quat = [[np.pi, - np.pi, - np.pi, - np.pi, - np.pi],
            [np.pi, - np.pi/2, - 0.3, 0, 0],
            [np.pi, - 2, - 0.3, 0, 0]]

    detected = sim.checkProximitySensor(Sensor, sim.handle_all)[0]
    # attach the graspable object
    if detected:
        # read_result = sim.readProximitySensor(Sensor)
        # detected_handle = read_result[3]
        # detected_obj_ori = sim.getObjectOrientation(detected_handle, sim.handle_world)
        # sim.setObjectInt32Parameter(handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(handle, AttachPoint, True)
        # obj_name_grasped = sim.getObjectAlias(handle, -1)
        if '_respondable' in obj_name:
            obj_name = obj_name.replace('_respondable', '')
        for sub_quat in quat:
            Controller.set_joint_position_for_hand(sub_quat)
        result = f"The robot's {side} hand has grasped {obj_name} from the top, and is lifting the {obj_name}."
        if side == 'left':
            LEFT_HAND_STAT, LEFT_FINGER_STAT = 'Horizontally_Down', 'Closed'
        else:
            RIGHT_HAND_STAT, RIGHT_FINGER_STAT = 'Horizontally_Down', 'Closed'
    else:
        result = f"The robot's {side} hand failed to grasp anything from the top, as there is no appropriate object detected."
    # return result
    time.sleep(0.5)
    sim.setObjectOrientation(handle, -1, handle_ori)
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT

def move_single_to_pose(side: str, target: str, obj_name = None, ori_angle = None, off_set=None):
    global LEFT_FINGER_STAT, LEFT_HAND_STAT, RIGHT_FINGER_STAT, RIGHT_HAND_STAT
    hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT if side == 'left' else ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    
    NICOL_Dict, CONTROLLER, SENSOR, CONTROLLER_Offset, CONTROLLER_HAND_STAT, CONTROLLER_FINGER_STAT = (ORIENTATION_DICT_LEFT, left, left_sensor, '_left', LEFT_HAND_STAT, LEFT_FINGER_STAT) if side == 'left' else (ORIENTATION_DICT_RIGHT, right, right_sensor, '_right', RIGHT_HAND_STAT, RIGHT_FINGER_STAT)
    ## check if it exists such target
    try: 
        target_handle = sim.getObject('/'+target)
        new_pose = sim.getObjectPosition(target_handle, sim.handle_world)
    except Exception as e:
        return f"The robot's {side} hand failed to move to {target} as the {target} is not detected in the scene." + hand_finger_info
    ## check if the target is in the area
    if (side=="right" and new_pose[1]>0.3) or (side=="left" and new_pose[1]<-0.3): 
        return f"The robot's {side} hand failed to move to {target}, out of its area." + hand_finger_info
    ## check if it exists such object
    if obj_name:
        try: 
            obj_handle = sim.getObject('/'+obj_name)
            obj_pose = sim.getObjectPosition(obj_handle, sim.handle_world)
        except Exception as e:
            return f"The robot's {side} hand failed to move {obj_name} as the {obj_name} is not detected in the scene." + hand_finger_info
        detected = sim.checkProximitySensor(SENSOR, obj_handle)[0]
        if not detected:
            return f"The robot's {side} hand failed to move {obj_name}, as the {obj_name} is not detected around the hand." + hand_finger_info
    if "point" in target:
        try:
            point_sensor = sim.getObject('/'+target+'_sensor')
            if sim.checkProximitySensor(point_sensor, sim.handle_all)[0]:
                return f"The robot's {side} hand failed to move to {target}, as the {target} is already occupied by the other object." + hand_finger_info
        except Exception as e: pass
    
    ori_angle = ori_angle if ori_angle else CONTROLLER_HAND_STAT
    if ori_angle=='Vertical':
        try: 
            handle = sim.getObject('/'+target + CONTROLLER_Offset)
            new_pose = sim.getObjectPosition(handle, sim.handle_world)
        except Exception as e:
            pass
    elif ori_angle=='Horizontally_Down':
        try: 
            handle = sim.getObject('/'+target+'_top')
            new_pose = sim.getObjectPosition(handle, sim.handle_world)
        except Exception as e: 
            new_pose = [new_pose[0], new_pose[1], new_pose[2]+0.015]
    new_quat = NICOL_Dict[ori_angle]

    offset_list = {
        'up': (0, 0, 0.25), 'above': (0, 0, 0.18),
        'down': (0, 0, -0.18), 'below': (0, 0, -0.12),
        'left': (0, 0.2, 0.02), 'right': (0, -0.2, 0.02)
    }

    dx, dy, dz = offset_list.get(off_set, (0, 0, 0))
    new_pose_w_offset = [new_pose[0] + dx, new_pose[1] + dy, new_pose[2] + dz]

    if off_set:
        CONTROLLER.set_pose_target(NicolPose(new_pose_w_offset, new_quat))
    else:
        EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
        new_pose_offset =  [new_pose[0], new_pose[1], new_pose[2]+0.15]
        if target != "serve_point":
            CONTROLLER.set_pose_target(NicolPose(new_pose_offset, new_quat))
        CONTROLLER.set_pose_target(NicolPose(new_pose, new_quat))

    result = f"The robot's {side} hand has moved " + (
        f"above the {target}." if off_set == 'up' else
        f"below the {target}." if off_set == 'down' else
        f"to {target}."
    )

    return result + hand_finger_info

    
def delta_move(side, direction, distance=None, ori_single=None):
    hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT if side == 'left' else ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    CONTROLLER = left if side == 'left' else right
    NICOL_DICT = ORIENTATION_DICT_LEFT if side == 'left' else ORIENTATION_DICT_RIGHT
    EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
    ori = CONTROLLER.get_eef_pose().orientation
    # ori = NICOL_DICT[LEFT_HAND_STAT] if side == 'left' else NICOL_DICT[RIGHT_HAND_STAT]
    ori = [ori.x, ori.y, ori.z, ori.w]
    # ori = [ori.w, ori.x, ori.y, ori.z]
    distance = distance if distance else 0.06
    if direction in ['right', 'y-decrease']:
        new_pose = [EE_POSE[0], EE_POSE[1]-distance, EE_POSE[2]]
    elif direction in ['left', 'y-increase']:
        new_pose = [EE_POSE[0], EE_POSE[1]+distance, EE_POSE[2]]
    elif direction in ['front', 'x-increase']:
        new_pose = [EE_POSE[0]+distance, EE_POSE[1], EE_POSE[2]]
    elif direction in ['behind', 'back', 'x-decrease']:
        new_pose = [EE_POSE[0]-distance, EE_POSE[1], EE_POSE[2]]
    elif direction == 'down':
        new_pose = [EE_POSE[0], EE_POSE[1], EE_POSE[2]-distance]
    elif direction == 'up':
        new_pose = [EE_POSE[0], EE_POSE[1], EE_POSE[2]+distance]
    else: new_pose = [EE_POSE[0], EE_POSE[1], EE_POSE[2]]
    if ori_single:
        CONTROLLER.set_pose_target(NicolPose(new_pose, NICOL_DICT[ori_single]))
    else:
        CONTROLLER.set_pose_target(NicolPose(new_pose, ori))
    result = f"The robot's {side} hand has moved {direction} for {distance} meters."
    return result + hand_finger_info

def release(side:str, obj_name=None, point=None):
    global LEFT_FINGER_STAT, LEFT_HAND_STAT, RIGHT_FINGER_STAT, RIGHT_HAND_STAT
    if side == 'left':
        CONTROLLER, CONTROLLER_Sensor = left, left_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_LEFT
        LEFT_FINGER_STAT = "Open"
        CONTROLLER_Ori = LEFT_HAND_STAT
        # delta_move('left', 'right', 0.02)
    elif side == 'right':
        CONTROLLER, CONTROLLER_Sensor = right, right_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_RIGHT
        RIGHT_FINGER_STAT = "Open"
        CONTROLLER_Ori = RIGHT_HAND_STAT
        # delta_move('right', 'left', 0.02)
    else:
        print("You didn't choose the correct side for NICOL!")
    quat = [-np.pi, - np.pi, - np.pi, - np.pi, - np.pi]
    CONTROLLER.set_joint_position_for_hand(quat)

    if obj_name:
        obj_handle = sim.getObject('/'+obj_name)
        # point_handle = sim.getObject('/'+point)
        # point_pose = sim.getObjectPosition(point_handle, sim.handle_world)
        if "block" in obj_name:
            # sim.setObjectInt32Parameter(obj_handle, sim.shapeintparam_static, 1)
            sim.setObjectInt32Parameter(obj_handle, sim.shapeintparam_respondable, 0)
            sim.setObjectParent(obj_handle, -1, True)
            # sim.setObjectPosition(obj_handle, point_handle, [0, 0, 0.1])
            # sim.setObjectOrientation(obj_handle, point_handle, [0, 0, 0])
            time.sleep(0.5)
    if point:
        if ("cup" in obj_name) and ("serve" in point):
            result = "failed! The release command is not allowed here!"
            if side == 'left':
                return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
            else:
                return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    else:
        detected = sim.checkProximitySensor(CONTROLLER_Sensor, sim.handle_all)[0]
        if detected:
            read_result = sim.readProximitySensor(CONTROLLER_Sensor)
            detected_handle = read_result[3]
            sim.setObjectParent(detected_handle, -1, True)
            obj_name = sim.getObjectAlias(detected_handle, -1)
            if '_respondable' in obj_name:
                obj_name = obj_name.replace('_respondable', '')
    if side == 'left' and LEFT_HAND_STAT == 'Vertical':
        delta_move('left', 'up', 0.15, ori_single=LEFT_HAND_STAT)
        delta_move('left', 'left', 0.1, ori_single=LEFT_HAND_STAT)
        # new_pose = [EE_POSE[0], EE_POSE[1]+0.03, EE_POSE[2]+0.06]
    if side == 'left' and LEFT_HAND_STAT == 'Horizontally_Down':
        delta_move('left', 'up', 0.08, ori_single=LEFT_HAND_STAT)
        delta_move('left', 'left', 0.3, ori_single=LEFT_HAND_STAT)
    if side == 'right' and RIGHT_HAND_STAT == 'Vertical':
        delta_move('right', 'up', 0.15, ori_single=RIGHT_HAND_STAT)
        delta_move('right', 'right', 0.1, ori_single=RIGHT_HAND_STAT)
    if side == 'right' and RIGHT_HAND_STAT == 'Horizontally_Down':
        delta_move('right', 'up', 0.08, ori_single=RIGHT_HAND_STAT)
        delta_move('right', 'right', 0.3, ori_single=RIGHT_HAND_STAT)

    if obj_name:
        result = f"The robot's {side} hand has been opened, and the grasped {obj_name} has been released."
    else:
        result = f"The robot's {side} hand has been opened."
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT


def place(side:str, source_obj:str, target_obj:str):
    global LEFT_FINGER_STAT, LEFT_HAND_STAT, RIGHT_FINGER_STAT, RIGHT_HAND_STAT
    hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT if side == 'left' else ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    if side == 'left':
        CONTROLLER, CONTROLLER_Sensor, DETECT_Sensor = left, left_sensor, right_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_LEFT
        LEFT_FINGER_STAT = "Open"
        CONTROLLER_Ori = LEFT_HAND_STAT
    elif side == 'right':
        CONTROLLER, CONTROLLER_Sensor, DETECT_Sensor = right, right_sensor, left_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_RIGHT
        RIGHT_FINGER_STAT = "Open"
        CONTROLLER_Ori = RIGHT_HAND_STAT
    else:
        print("You didn't choose the correct side for NICOL!")
    EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
    CONTROLLER.set_pose_target(NicolPose(EE_POSE, CONTROLLER_DICT["Horizontally_Down"]))
    source_obj_handle = sim.getObject('/'+source_obj)
    target_obj_handle = sim.getObject('/'+target_obj)
    source_obj_pos = sim.getObjectPosition(source_obj_handle, -1)
    target_obj_pos = sim.getObjectPosition(target_obj_handle, -1)
    if (side=="right" and target_obj_pos[1]>0.3) or (side=="left" and target_obj_pos[1]<-0.3): 
        return f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as the {target_obj} is out of its area." + hand_finger_info
    source_obj_ori = sim.getObjectOrientation(source_obj_handle, -1)
    detected = sim.checkProximitySensor(CONTROLLER_Sensor, source_obj_handle)[0]
    if not detected:
        return f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as the {source_obj} is not detected in on the hand." + hand_finger_info
    detected = sim.checkProximitySensor(DETECT_Sensor, target_obj_handle)[0]
    if detected:
        return f"The robot's {side} hand failed to place {source_obj} on {target_obj}, as the {source_obj} is grasped by another hand." + hand_finger_info
    quat = [-np.pi, - np.pi, - np.pi, - np.pi, - np.pi]
    if "bowl" in target_obj:
        delta_move(side, "down", 0.12)
        # if source_obj == "apple":
        #     delta_move(side, "front", 0.05)
        CONTROLLER.set_joint_position_for_hand(quat)
        time.sleep(3)
        bowl = sim.getObject('/bowl_respondable')
        bowl_pos = sim.getObjectPosition(bowl, -1)
        if abs(source_obj_pos[0] - bowl_pos[0]) < 0.25 and abs(source_obj_pos[1] - bowl_pos[1]) < 0.25 and source_obj_pos[2] > bowl_pos[2]:
            print("The released object is above the bowl!")
            sim.setObjectInt32Parameter(source_obj_handle, sim.shapeintparam_static, 1)
            sim.setObjectInt32Parameter(source_obj_handle, sim.shapeintparam_respondable, 0)
            sim.setObjectParent(source_obj_handle, bowl, True)
            if source_obj_handle == sim.getObject("/banana"):
                sim.setObjectPosition(source_obj_handle, bowl, [-0.05, -0.07, 0.01])
            elif source_obj_handle == sim.getObject("/apple"):
                sim.setObjectPosition(source_obj_handle, bowl, [-0.012, -0.146, +0.02])
            sim.setObjectOrientation(source_obj_handle, bowl, source_obj_ori)
            time.sleep(0.5)
        if side == 'right':
            delta_move('right', 'up', 0.05)
            delta_move('right', 'right', 0.2)
        elif side == 'left':
            delta_move('left', 'up', 0.05)
            delta_move('left', 'left', 0.2)
    elif "block" in target_obj:
        # delta_move(side, 'down', 0.08, ori_single='Horizontally_Down')
        new_pose = [target_obj_pos[0], target_obj_pos[1], target_obj_pos[2]+0.12]
        CONTROLLER.set_pose_target(NicolPose(new_pose, CONTROLLER_DICT['Horizontally_Down']))
        time.sleep(1)
        CONTROLLER.set_joint_position_for_hand(quat)
        if abs(source_obj_pos[0] - target_obj_pos[0]) < 0.3 and abs(source_obj_pos[1] - target_obj_pos[1]) < 0.3 and source_obj_pos[2] > target_obj_pos[2]:
            sim.setObjectInt32Parameter(source_obj_handle, sim.shapeintparam_static, 1)
            sim.setObjectParent(source_obj_handle, -1, True)
            sim.setObjectParent(source_obj_handle, target_obj_handle, True)
            sim.setObjectPosition(source_obj_handle, target_obj_handle, [0, 0, -0.05])
            sim.setObjectOrientation(source_obj_handle, -1, [-np.pi, 0, np.pi/2])
        if side == 'right':
            delta_move('right', 'up', 0.075)
            delta_move('right', 'right', 0.2)
        elif side == 'left':
            delta_move('left', 'up', 0.075)
            delta_move('left', 'left', 0.2)
    result = f"The robot's {side} hand has already placed the grasped {source_obj} on the {target_obj}."
    if side == 'left':
        LEFT_FINGER_STAT = "Open"
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        RIGHT_HAND_STAT = "Open"
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT

def reset(side:str):
    if side == 'left':
        release('left')
        left.set_joint_position_for_hand([-np.pi] * 5, block=True)
        left.set_joint_position([-1.57] + [0.] * 7, block=True)
        global LEFT_FINGER_STAT, LEFT_HAND_STAT
        LEFT_FINGER_STAT, LEFT_HAND_STAT = 'Open', 'Vertical'
        result = f"The robot's {side} hand has been reset to its original position."
    elif side == 'right':
        release('right')
        right.set_joint_position_for_hand([-np.pi] * 5, block=True)
        right.set_joint_position([1.57] + [0.] * 7, block=True)
        global RIGHT_FINGER_STAT, RIGHT_HAND_STAT
        RIGHT_FINGER_STAT, RIGHT_HAND_STAT = 'Open', 'Vertical'
        result = f"The robot's {side} hand has been reset to its original position."
    else:
        return "You didn't choose the correct side for NICOL!"
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    # return result

def flip_down(side:str):
    # try:
    ball = sim.getObject("/big_ball")
    yellow_cup = sim.getObjectHandle('/yellow_cup_respondable')
    blue_cup = sim.getObjectHandle('/blue_cup_respondable')
    yellow_pos = sim.getObjectPosition(yellow_cup, -1)
    blue_pos = sim.getObjectPosition(blue_cup, -1)
    # except Exception as e:
    #     return 'There is no water inside the container!'
    if side == 'left':
        CONTROLLER = left
        global LEFT_HAND_STAT
        # LEFT_HAND_STAT = 'Horizontally_Down'
        CONTROLLER_sensor = left_sensor
        CONTROLLER_DICT = ORIENTATION_DICT_LEFT
        EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
        # new_pose_1 = EE_POSE
        new_pose_1 = [EE_POSE[0]-0.01, EE_POSE[1]-0.01, EE_POSE[2]-0.04]
        # if  sim.checkProximitySensor(CONTROLLER_sensor, ball)[0]:
        # sim.setObjectParent(ball, -1, True)
        # sim.setObjectInt32Parameter(ball, sim.shapeintparam_static, 0)
    elif side == 'right':
        CONTROLLER = right
        global RIGHT_HAND_STAT
        # RIGHT_HAND_STAT = 'Horizontally_Down'
        CONTROLLER_DICT = ORIENTATION_DICT_RIGHT
        CONTROLLER_sensor = right_sensor
        EE_POSE = CONTROLLER.get_eef_pose().position.as_list()
        # new_pose_1 = EE_POSE
        new_pose_1 = [EE_POSE[0]-0.01, EE_POSE[1]-0.02, EE_POSE[2]-0.04]
        # if  sim.checkProximitySensor(CONTROLLER_sensor, ball)[0]:

    else:
        print("You didn't choose the correct side for NICOL!")
    # prepare to pour out
    ori_1 = CONTROLLER_DICT['Horizontally_Slanted_Down']
    CONTROLLER.set_pose_target(NicolPose(new_pose_1, ori_1))
    time.sleep(0.5)
    # check if the blue cup is above the yellow cup
    if abs(blue_pos[0] - yellow_pos[0]) < 0.04 and abs(blue_pos[1] - yellow_pos[1]) < 0.04 and blue_pos[2] > yellow_pos[2]:
        print("Blue cup is above Yellow cup! Transferring the ball...")
    # transfer the ball to the yellow cup
        sim.setObjectInt32Parameter(ball, sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(ball, sim.shapeintparam_respondable, 0)
        sim.setObjectParent(ball, yellow_cup, True)
        sim.setObjectPosition(ball, yellow_cup, [0.03, -0.01, -0.02])
        time.sleep(0.5)
    # back to vertical position
    ori_2= CONTROLLER_DICT['Vertical']
    new_pose_2 = [new_pose_1[0], new_pose_1[1], new_pose_1[2]+ 0.02]
    # return [[new_pose_1, ori_1], [new_pose_2, ori_2]]
    CONTROLLER.set_pose_target(NicolPose(new_pose_2, ori_2))
    result = f"The robot's {side} hand's is fliped down for once for pouring out water."
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT


def hold_up_single(side:str, obj_name:str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    quat = [-np.pi, -np.pi, -np.pi*0.75, -np.pi*0.75, -np.pi*0.75]
    obj_pose = sim.getObjectPosition(sim.getObject('/'+obj_name), sim.handle_world)
    obj_pose = [obj_pose[0], obj_pose[1], obj_pose[2]+0.18]
    if side == 'left':
        detected = sim.checkProximitySensor(left_sensor, sim.handle_all)[0]
        if obj_pose[1]<-0.25:
            return "hold_up is failed, out of its area." + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
        if LEFT_FINGER_STAT == 'Closed' and detected:
            return "The left hand is already occupied for grasping and holding some object." + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
        else:
            release('left')
    else:
        detected = sim.checkProximitySensor(right_sensor, sim.handle_all)[0]
        if obj_pose[1]>0.25:
            return "hold_up is failed, out of its area." + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
        if RIGHT_FINGER_STAT == 'Closed' and detected:
            return "the right hand is already occupied for grasping and holding some object." + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
        else:
            release('right')    
    # if obj_name == "Bowl":
    if side == 'left':
        left_target_pose = sim.getObjectPosition(sim.getObject('/'+obj_name +'_left'), sim.handle_world)
        if LEFT_HAND_STAT != 'Horizontally_Slanted_Up':
            left_poses = [[0.6, 0.3, 1.0],
                    [left_target_pose[0], left_target_pose[1]+0.01, left_target_pose[2]+0.18],
                    # [left_target_pose[0], left_target_pose[1]+0.04, left_target_pose[2]],
                    [left_target_pose[0]-0.02, left_target_pose[1]+0.04, left_target_pose[2]]]
            for i in range(3):
                left.set_pose_target(NicolPose(left_poses[i], ORIENTATION_DICT_LEFT['Horizontally_Slanted_Up']))
            # LEFT_EE_POSE = left.get_eef_pose().position.as_list()
            # LEFT_EE_POSE = [LEFT_EE_POSE[0]+0.03, LEFT_EE_POSE[1]-0.07, LEFT_EE_POSE[2]+0.01]
            new_pose = [left_target_pose[0]+0.02, left_target_pose[1]-0.05, left_target_pose[2]]
            left.set_pose_target(NicolPose(new_pose, ORIENTATION_DICT_LEFT['Horizontally_Slanted_Up']))
            LEFT_HAND_STAT = 'Horizontally_Slanted_Up'
            LEFT_FINGER_STAT = 'Hold_Up'
            left.set_joint_position_for_hand(quat)
    else:
        right_target_pose = sim.getObjectPosition(sim.getObject('/'+obj_name+'_right'), sim.handle_world)
        if RIGHT_HAND_STAT != 'Horizontally_Slanted_Up':
            right_poses = [[0.6, -0.3, 1.0],
                    [right_target_pose[0], right_target_pose[1]-0.01, right_target_pose[2]+0.18],
                    # [right_target_pose[0], right_target_pose[1]-0.04, right_target_pose[2]],
                    [right_target_pose[0]-0.02, right_target_pose[1]-0.06, right_target_pose[2]]]
            for i in range(3):
                right.set_pose_target(NicolPose(right_poses[i], ORIENTATION_DICT_RIGHT['Horizontally_Slanted_Up']))
            # RIGHT_EE_POSE = right.get_eef_pose().position.as_list()
            # RIGHT_EE_POSE = [RIGHT_EE_POSE[0]+0.03, RIGHT_EE_POSE[1]+0.02, RIGHT_EE_POSE[2]+0.01]
            new_pose = [right_target_pose[0]-0.02, right_target_pose[1]+0.05, right_target_pose[2]]
            right.set_pose_target(NicolPose(new_pose, ORIENTATION_DICT_RIGHT['Horizontally_Slanted_Up']))
            right.set_joint_position_for_hand(quat)
            RIGHT_HAND_STAT = 'Horizontally_Slanted_Up'
            RIGHT_FINGER_STAT = 'Hold_Up'
    left_detected = sim.checkProximitySensor(left_sensor, sim.handle_all)[0]
    right_detected = sim.checkProximitySensor(right_sensor, sim.handle_all)[0]
    # attach the graspable object
    if left_detected and side == 'left':
        read_result = sim.readProximitySensor(left_sensor)
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, left_attachpoint, True)
        obj_name_grasped = sim.getObjectAlias(read_result[3], -1)
        if '_respondable' in obj_name_grasped:
            obj_name_grasped = obj_name_grasped.replace('_respondable', '')
    elif right_detected and side == 'right':
        read_result = sim.readProximitySensor(right_sensor)
        detected_handle = read_result[3]
        sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(detected_handle, right_attachpoint, True)
        obj_name_grasped = sim.getObjectAlias(read_result[3], -1)
        if '_respondable' in obj_name_grasped:
            obj_name_grasped = obj_name_grasped.replace('_respondable', '')
    result = f"The robot's {side} hand is prepared to hold the {obj_name} from the side."
    if side == 'left':
        return result + ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT
    else:
        return result + ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT

def push_to(side:str, source_obj:str, target_obj:str):
    global LEFT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_HAND_STAT, RIGHT_FINGER_STAT
    hand_finger_info = ' LEFT_HAND: ' + LEFT_HAND_STAT + ' LEFT_FINGER: ' + LEFT_FINGER_STAT if side == 'left' else ' RIGHT_HAND: ' + RIGHT_HAND_STAT + ' RIGHT_FINGER: ' + RIGHT_FINGER_STAT
    if (side == 'left' and LEFT_FINGER_STAT == 'Closed') or (side == 'right' and RIGHT_FINGER_STAT == 'Closed'):
        return f"failed! The robot's {side} hand is already occupied for grasping some object." + hand_finger_info
    if "area" not in target_obj:
        return f"failed! It is not allowed to push {source_obj} to {target_obj}." + hand_finger_info
    try: 
        source_handle = sim.getObject('/'+source_obj)
        source_handle_respond = sim.getObject('/'+source_obj+'_respondable')
    except Exception as e: return f"failed! The object {source_obj} is not in the scene." + hand_finger_info
    try:
        target_handle = sim.getObject('/'+target_obj)
        target_pose = np.array(sim.getObjectPosition(target_handle, sim.handle_world))
    except Exception as e:
        return f"failed! The object {target_obj} is not in the scene." + hand_finger_info
    source_pose = np.array(sim.getObjectPosition(source_handle, sim.handle_world))
    target_posee = np.array(sim.getObjectPosition(target_handle, sim.handle_world))
    origin_orientation = sim.getObjectOrientation(source_handle, sim.handle_world)
    if (side=="right" and source_pose[1]>0.3) or (side=="left" and source_pose[1]<-0.3): 
        return f"The robot's {side} hand failed to push {source_obj}, out of its area." + hand_finger_info
    if (side=="right" and target_posee[1]>0.3) or (side=="left" and target_posee[1]<-0.3): 
        return f"The robot's {side} hand failed to push to {target_obj}, out of its area." + hand_finger_info
    # sim.setObjectInt32Parameter(source_obj_handle, sim.shapeintparam_static, 1)
    # except Exception as e:
        # return f"The object {source_obj} is not in the scene." + hand_finger_info
    if side == 'left':
        release('left')
        # delta_move('left', 'left', 0.06)
        LEFT_HAND_STAT = 'Vertical'
        # result = move_single_to_pose(side, source_obj, 'Vertical')
        if target_pose[1]>0:
            new_pose = np.array(sim.getObjectPosition(sim.getObject('/'+source_obj+'_right'), sim.handle_world))
        else:
            new_pose = np.array(sim.getObjectPosition(sim.getObject('/'+source_obj+'_left'), sim.handle_world))
            # left.set_pose_target(NicolPose(new_pose, ORIENTATION_DICT_LEFT['Vertical']))
        new_poses = [[new_pose[0], new_pose[1], new_pose[2]+0.3],
                    [new_pose[0], new_pose[1], new_pose[2]+0.1],
                    [new_pose[0], new_pose[1], new_pose[2]+0.01]]
        for pos in new_poses:
            left.set_pose_target(NicolPose(pos, ORIENTATION_DICT_LEFT['Vertical']))
        # if 'failed' in result: return result + hand_finger_info
        # if sim.checkProximitySensor(left_sensor, sim.handle_all)[0]:
        #     read_result = sim.readProximitySensor(left_sensor)
        #     detected_handle = read_result[3]
        # sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(source_handle_respond, left_attachpoint, True)
        move_single_to_pose(side, target=target_obj, ori_angle='Vertical', off_set='left')
        sim.setObjectParent(source_handle_respond, -1, True)
        if target_pose[1]>0:
            delta_move(side, 'right', 0.1)
            delta_move(side, 'up', 0.2)
        else:
            delta_move(side, 'left', 0.1)
            delta_move(side, 'up', 0.2)
        sim.setObjectOrientation(source_handle, sim.handle_world, origin_orientation)
        return f"The robot's left hand has pushed the {source_obj} to the {target_obj}. LEFT_HAND: {LEFT_HAND_STAT} LEFT_FINGER: {LEFT_FINGER_STAT}"
    else:
        release('right')
        # delta_move(side, 'right', 0.06)
        RIGHT_HAND_STAT = 'Vertical'
        if target_pose[1]<0:
            new_pose = np.array(sim.getObjectPosition(sim.getObject('/'+source_obj+'_left'), sim.handle_world))
            # right.set_pose_target(NicolPose(new_pose, ORIENTATION_DICT_LEFT['Vertical']))
        else:
            new_pose = np.array(sim.getObjectPosition(sim.getObject('/'+source_obj+'_right'), sim.handle_world))
            # right.set_pose_target(NicolPose(new_pose, ORIENTATION_DICT_LEFT['Vertical']))
        new_poses = [[new_pose[0], new_pose[1], new_pose[2]+0.3],
                    [new_pose[0], new_pose[1], new_pose[2]+0.1],
                    [new_pose[0], new_pose[1], new_pose[2]+0.01]]
        for pos in new_poses:
            right.set_pose_target(NicolPose(pos, ORIENTATION_DICT_RIGHT['Vertical']))
        # result = move_single_to_pose(side, source_obj, 'Vertical')
        # if 'failed' in result: return result + hand_finger_info
        # if sim.checkProximitySensor(right_sensor, sim.handle_all)[0]:
        #     read_result = sim.readProximitySensor(right_sensor)
        #     detected_handle = read_result[3]
        #     sim.setObjectInt32Parameter(detected_handle, sim.shapeintparam_static, 1)
        sim.setObjectParent(source_handle_respond, right_attachpoint, True)
        move_single_to_pose('right', target=target_obj, ori_angle='Vertical', off_set='right')
        sim.setObjectParent(source_handle_respond, -1, True)
        if target_pose[1]>0:
            delta_move(side, 'left', 0.1)
            delta_move(side, 'up', 0.2)
        else:
            delta_move(side, 'right', 0.1)
            delta_move(side, 'up', 0.2)
        sim.setObjectOrientation(source_handle, sim.handle_world, origin_orientation)
        return f"The robot's right hand has pushed the {source_obj} to the {target_obj}. RIGHT_HAND: {LEFT_HAND_STAT} RIGHT_FINGER: {LEFT_FINGER_STAT}"

def main(args):
    command = args.command
    para = json.loads(args.para)
    global LEFT_HAND_STAT, RIGHT_HAND_STAT, LEFT_FINGER_STAT, RIGHT_FINGER_STAT
    if args.side == 'left': LEFT_HAND_STAT, LEFT_FINGER_STAT = args.hand_state, args.finger_state
    elif args.side == 'right': RIGHT_HAND_STAT, RIGHT_FINGER_STAT = args.hand_state, args.finger_state
    if command == 'side_grasp':
        return side_grasp(args.side, **para)
    elif command == 'top_grasp':
        return top_grasp(args.side, **para)
    elif command == 'release': 
        return release(args.side, **para)
    elif command == 'move_above': 
        return move_single_to_pose(args.side,**para)
    elif command == 'pour': 
        return flip_down(args.side, **para)
    elif command == 'hold_up': 
        return hold_up_single(args.side, **para)
    elif command == 'move_single_to_pose': 
        return move_single_to_pose(args.side, **para)
    elif command == 'wait' or command == "None": 
        side = args.side
        hand_stat = f'LEFT_HAND: {LEFT_HAND_STAT}' if side == 'left' else f'RIGHT_HAND: {RIGHT_HAND_STAT}'
        finger_stat = f'LEFT_FINGER: {LEFT_FINGER_STAT}' if side == 'left' else f'RIGHT_FINGER: {RIGHT_FINGER_STAT}'
        return f"The robot's {side} hand is keeping the current status. {hand_stat} {finger_stat}" 
    elif command == 'reset': 
        return reset(args.side)
    elif command == 'place':
        return place(args.side, **para)
    elif command == 'push_to': 
        return push_to(args.side, **para)
    elif command == 'co_move_to':
        result = move_both_to_poses(**para)
        side = args.side
        hand_stat = f' LEFT_HAND: {LEFT_HAND_STAT}' if side == 'left' else f' RIGHT_HAND: {RIGHT_HAND_STAT}'
        finger_stat = f' LEFT_FINGER: {LEFT_FINGER_STAT}' if side == 'left' else f' RIGHT_FINGER: {RIGHT_FINGER_STAT}'
        return result + hand_stat + finger_stat
    elif command == 'co_hold':
        return hold_up_single(args.side, **para)
    elif command == 'look_at':
        return look_at(**para)
    else: 
        return f'You chosed the unknown command for the chosed hand.'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subprocess Controller Parameters")
    parser.add_argument('--side', type=str, help="Which side of the hand that executes the command")
    parser.add_argument('--command', type=str, help="The command for the side controller")
    parser.add_argument('--para', type=str, help="The parameters for the command")
    parser.add_argument('--hand_state', type=str, help="The hand state for the side controller")
    parser.add_argument('--finger_state', type=str, help="The finger state for the side controller")
    args = parser.parse_args()
    result = main(args)
    result = 'LEFT_RESULT: ' + result if args.side == 'left' else 'RIGHT_RESULT: ' + result
    print(result)