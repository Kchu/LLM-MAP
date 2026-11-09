#!/usr/bin/env python
"""
Task design for NICOL Bimanual Robot
Tasks: ServeWater, ServeFruit, StackBlock
Copyright 2024, Planet Earth
"""
import random
import os, time
from nicol_api.nicol_env import NicolFactory
from nicol_api.base import NicolPose
from coppeliasim_zmqremoteapi_client import *
import itertools
import json

path = os.path.dirname(os.path.abspath(__file__))

"""
Task design structure:
task -- load model
task -- add description
task -- check success condition
"""

nicol = NicolFactory().create_nicol('coppelia', scene="./nicol.ttt", start_scene=False, talker=False)
head  = nicol.head()    # 3D-Printed head structure
left  = nicol.left()    # Left OpenManipulator + RH8D
right = nicol.right()   # Right OpenManipulator + RH8D
sim = nicol.nicol_adapter.sim

random.seed(666)

#########################################################################################
class ServeWaterTask():
    def __init__(self) -> None:
        try: 
            sim.removeModel(sim.getObject('/ServeFruit'))
        except Exception as e: pass
        try:
            sim.removeModel(sim.getObject('/StackBlock'))
        except Exception as e: pass
        try:
            self.blue_cup = sim.getObject("/Origin_blue_cup")
            self.yellow_cup = sim.getObject("/Origin_yellow_cup")
            self.ball = sim.getObject("/big_ball")
            self.serve_point = sim.getObject("/serve_point")
            self.store_point = sim.getObject("/store_point")
            self.box = sim.getObject("/bbox")
            self.sensor_1 = sim.getObject("/yellow_cup_sensor")
            self.sensor_2 = sim.getObject("/serve_point_sensor")
            self.sensor_3 = sim.getObject("/bbox_sensor")
            self.human = sim.getObject('/Bill')
            self.human_goal = sim.getObject('/goalDummy')
        except Exception as e:
            self.model_handle = sim.loadModel(path+'/NICOL/ServeWater.ttm')
            self.blue_cup = sim.getObject("/Origin_blue_cup")
            self.yellow_cup = sim.getObject("/Origin_yellow_cup")
            self.serve_point = sim.getObject("/serve_point")
            self.store_point = sim.getObject("/store_point")
            self.box = sim.getObject("/bbox")
            self.ball = sim.getObject("/big_ball")
            self.sensor_1 = sim.getObject("/yellow_cup_sensor")
            self.sensor_2 = sim.getObject("/serve_point_sensor")
            self.sensor_3 = sim.getObject("/bbox_sensor")
            self.human = sim.getObject('/Bill')
            self.human_goal = sim.getObject('/goalDummy')
        self.name = "ServeWater"
        self.sim = sim
        self.text_queries = ["human face", "blue cup", "yellow cup", "brown box", "red ball"]
        self.owl_threshold = 0.04
        self.positions = {}
        self.areas = {}
        sim.setObjectInt32Parameter(sim.getObject("/blue_cup_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/yellow_cup_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/blue_cup_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/yellow_cup_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(self.ball, sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(self.ball, sim.shapeintparam_respondable, 0)
     
    def reset(self, config=None):
        if config:
            self.config = config
            self.positions = config["objects"]
            self.areas = config["area"]
            self.type = config["type"]
            y_human = self.positions["human"][1]
            sim.setObjectPose(self.human, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.human_goal, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.serve_point, sim.handle_world, self.positions["serve_point"]+ [0,0,0,1])
            time.sleep(2)
            sim.setObjectPose(self.box, sim.handle_world, self.positions["box"] +[0, -0.7, 0, 0.7])
            sim.setObjectPose(self.store_point, self.box, self.positions["store_point"] + [0, 0, 0, 1])
            sim.setObjectPose(self.blue_cup, sim.handle_world, self.positions["blue_cup"] + [0, 0, 0, 1])
            sim.setObjectPose(self.yellow_cup, sim.handle_world, self.positions["yellow_cup"] + [0, 0, 0, 1])
            # if self.positions["ball"]:
            parentHandle = sim.getObjectHandle('/blue_cup_respondable')
            # else:
                # parentHandle = sim.getObjectHandle('/yellow_cup_respondable')
            sim.setObjectParent(self.ball, parentHandle, True)
            sim.setObjectPosition(self.ball, parentHandle, [0.03, -0.01, -0.02])
        else:
            y_human = random.choice([0.5, -0.5, 0])
            if y_human > 0: self.areas["human"] = "left"
            elif y_human < 0: self.areas["human"] = "right"
            else: self.areas["human"] = "center"
            sim.setObjectPose(self.human, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.human_goal, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.serve_point, sim.handle_world, [0.8, y_human, 1.1, 0,0,0,1])
            self.positions["human"] = [1.1, y_human, 0]
            self.positions["serve_point"] = [0.8, y_human, 1.1]
            time.sleep(2)
            z1 = z2 = 0.86
            x1, y1 = random.choice([[0.4, 0.4], [0.4, -0.4], [0.4, 0]])
            x2, y2 = random.choice([[0.7, 0.4], [0.7, -0.4], [0.7, 0]])
            if y1>0: self.areas["blue cup"] = "left"
            elif y1<0: self.areas["blue cup"] = "right"
            else: self.areas["blue cup"] = "center"
            if y2>0: self.areas["yellow cup"] = "left"
            elif y2<0: self.areas["yellow cup"] = "right"
            else: self.areas["yellow cup"] = "center"

            box_pose = random.choice([[0.55, 0.6, 0.83], [0.55, -0.6, 0.83]])
            self.positions["box"] = box_pose
            self.areas["brown box"] = "left" if box_pose[1] > 0 else "right"
            
            sim.setObjectPose(self.box, sim.handle_world, box_pose +[0,-0.7,0,0.7])
            if box_pose[1] > 0:
                sim.setObjectPose(self.store_point, self.box, [0.06, 0.025, 0, 0,0,0,1])
                self.positions["store_point"] = [0.06, 0.025, 0]
            else:
                sim.setObjectPose(self.store_point, self.box, [0.06, -0.025, 0, 0,0,0,1])
                self.positions["store_point"] = [0.06, -0.025, 0]
            sim.setObjectPose(self.blue_cup, sim.handle_world, [x1, y1, z1, 0,0,0,1])
            sim.setObjectPose(self.yellow_cup, sim.handle_world, [x2, y2, z2, 0,0,0,1])
            self.positions["blue_cup"] = [x1, y1, z1]
            self.positions["yellow_cup"] = [x2, y2, z2]
            
            # if random.choice([0, 1]):
            parentHandle = sim.getObjectHandle('/blue_cup_respondable')
            self.positions["ball"] = 1
            self.areas["red ball"] = "blue cup"
            # else:
            #     parentHandle = sim.getObjectHandle('/yellow_cup_respondable')
            #     self.positions["ball"] = 0
            #     self.areas["red ball"] = "yellow cup"
            sim.setObjectParent(self.ball, parentHandle, True)
            sim.setObjectPosition(self.ball, parentHandle, [0.03, -0.01, -0.02])
            self.type = ["blue: " + self.areas["blue cup"], "yellow: " + self.areas["yellow cup"], "human: " + self.areas["human"], "store: " + self.areas["brown box"], "ball: " + self.areas["red ball"]]

            # self.blue_cup_pose =  [x1, y1, z1]
            # self.yellow_cup_pose = [x2, y2, z2]
            # self.store_point_pose = sim.getObjectPose(self.store_point, -1)[0:3]
            self.task_des = f"""\n
### Task Information: \n
## Task Description: There are two cups placed on the table: blue cup and yellow cup.
## Task goal: Serve the yellow cup with water inside to the human at the serve_point, which is right in front of the human, while the blue cup is put at the store_point, which is at the central of the brown box.

You are not allowed to release objects at the serve_point.
## Task related objects: blue_cup, yellow_cup
## Task related points: serve_point, overlap_point_1, overlap_point_2
"""
            self.task_des_visual = f"""\n
# Based on the vision analyzation, the human is standing in front of the {self.areas["human"]} area, the blue cup is at the {self.areas['blue cup']} area, the yellow cup is at the {self.areas['yellow cup']} area, the brown box is at the {self.areas['brown box']} area, with the store_point in the central, and the water is inside the {self.areas['red ball']} at the present.
"""
            self.config = {"objects": self.positions, "area":self.areas, "task_des": self.task_des, "task_des_visual": self.task_des_visual, "type": self.type}
    def check_success(self):
        yellow_cup_respond = sim.getObject("/yellow_cup_respondable")
        blue_cup_respond = sim.getObject("/blue_cup_respondable")
        detect_result_1 = sim.checkProximitySensor(self.sensor_1, self.ball)
        detect_result_2 = sim.checkProximitySensor(self.sensor_2, yellow_cup_respond)
        detect_result_3 = sim.checkProximitySensor(self.sensor_3, blue_cup_respond)
        if detect_result_1[0] and detect_result_2[0] and detect_result_3[0]:
            return True
        else:
            return False
        
    def get_image_from_eyes(self, index):
        head.set_pose_target(NicolPose([0.75, 0.0, 1.1], [0, 0, 0, 0]))
        self.image = nicol.head_.get_left_eye_camera_img(True, prefix=f"{self.name}_{index}")
        head.set_pose_target(NicolPose([0.75, 0.0, 1.4], [0, 0, 0, 0]))
        return self.image


#########################################################################################
class ServeFruitTask():
    def __init__(self) -> None:
        try: sim.removeModel(sim.getObject('/ServeWater'))
        except Exception as e: pass
        try: sim.removeModel(sim.getObject('/StackBlock'))
        except Exception as e: pass
        try:
            self.bowl_object = sim.getObject("/Origin_bowl")
            self.apple_object = sim.getObject("/apple")
            self.banana_object = sim.getObject("/banana")
            self.target_sensor = sim.getObject("/serve_point_sensor")
            # self.box = sim.getObject("/bbox")
        except Exception as e:
            sim.loadModel(path + '/NICOL/ServeFruit.ttm')
            self.bowl_object = sim.getObject("/Origin_bowl")
            self.apple_object = sim.getObject("/apple")
            self.banana_object = sim.getObject("/banana")
            self.target_sensor = sim.getObject("/serve_point_sensor")
            # self.box = sim.getObject("/bbox")
        self.bowl_respond = sim.getObject("/bowl_respondable")
        self.human = sim.getObject('/Bill')
        self.human_goal = sim.getObject('/goalDummy')
        sim.setObjectInt32Parameter(sim.getObject("/bowl_respondable"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/apple"), sim.shapeintparam_static, 1)
        sim.setObjectInt32Parameter(sim.getObject("/banana"), sim.shapeintparam_static, 1)
        # sim.setObjectInt32Parameter(sim.getObject("/Bowl_respondable"), sim.shapeintparam_static, 1)
        # sim.setObjectInt32Parameter(self.apple_object, sim.shapeintparam_static, 1)
        # sim.setObjectInt32Parameter(self.banana_object, sim.shapeintparam_static, 1)
        self.name = "ServeFruit"
        self.sim = sim
        self.text_queries = ["human face", "red bowl", "apple", "banana"]
        self.owl_threshold = 0.03
        self.objects = {}
        self.areas = {}

    def reset(self, config=None):
        sim.setObjectPose(self.human_goal, sim.handle_world, [3, 0, 0, 0,0,-1,0])
        sim.setObjectPose(self.human, sim.handle_world, [3, 0, 0, 0,0,-1,0])
        sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, 0, 0, 0,0,-1,0])
        if config:
            self.positions = config["objects"]
            self.areas = config["area"]
            self.type = config["type"]
            sim.setObjectPose(self.apple_object, sim.handle_world, self.positions["apple"] +[0, 0, 0, 1])
            sim.setObjectPose(self.banana_object, sim.handle_world, self.positions["banana"] + [0, 0, -4, 1])
            sim.setObjectPose(self.bowl_object, sim.handle_world, self.positions["bowl"] + [0, 0, 0, 1])
        else:
            z1, z2, z3 = 0.82, 0.81, 0.83
            x1, y1 = random.choice([[0.4, 0.5], [0.4, -0.5]])
            x2, y2 = random.choice([[0.5, 0.65], [0.5, -0.65]])
            x3, y3 = random.choice([[0.6, 0.4], [0.6, -0.4], [0.6, 0]])
            sim.setObjectPose(self.apple_object, sim.handle_world, [x1, y1, z1, 0, 0, 0, 1])
            sim.setObjectPose(self.banana_object, sim.handle_world, [x2, y2, z2, 0, 0, -4, 1])
            sim.setObjectPose(self.bowl_object, sim.handle_world, [x3, y3, z3, 0, 0, 0, 1])
            self.objects["apple"] = [x1, y1, z1]
            self.areas["apple"] = "left" if y1 > 0 else "right"
            self.objects["banana"] = [x2, y2, z2]
            self.areas["banana"] = "left" if y2 > 0 else "right"
            self.objects["bowl"] = [x3, y3, z3]
            if y3 > 0: self.areas["bowl"] = "left"
            elif y3< 0: self.areas["bowl"] = "right"
            else: self.areas["bowl"] = "center"
            self.type = [self.areas["apple"], self.areas["banana"], self.areas["bowl"]]
            self.task_des = f"""\n
### Task Information: \n
## Task Description: There are three objects at the table: an apple, a banana, and a large bowl (not graspable with one single hand).
## Task Goal: Serve the bowl with fruits inside at the serve_point, which is right in front of the human.
## Task related objects: apple, banana, bowl
## Task related points: serve_point
You are not allowed to release objects at the serve_point.
"""
            self.task_des_visual = f"""
# Based on the vision analyzation, the human is standing in the central area, with the red bowl at the {self.areas['bowl']} area, the apple at the {self.areas['apple']} area, and the banana at the {self.areas['banana']} area.
"""
            self.config = {"objects": self.objects, "area": self.areas, "task_des": self.task_des, "task_des_visual": self.task_des_visual, "type": self.type}
    def check_success(self):
        apple_detected = sim.checkProximitySensor(self.target_sensor, self.apple_object)[0]
        banana_detected = sim.checkProximitySensor(self.target_sensor, self.banana_object)[0]
        bowl_detected = sim.checkProximitySensor(self.target_sensor, self.bowl_respond)[0]
        if apple_detected>0 and banana_detected>0 and bowl_detected>0:
            return True
        else:
            return False
        
    def get_image_from_eyes(self, index):
        head.set_pose_target(NicolPose([0.75, 0.0, 1.1], [0, 0, 0, 0]))
        self.image = nicol.head_.get_left_eye_camera_img(True, prefix=f"{self.name}_{index}")
        head.set_pose_target(NicolPose([0.75, 0.0, 1.4], [0, 0, 0, 0]))
        return self.image 

#########################################################################################
class StackBlocksTask():
    def __init__(self, config="simple") -> None:
        try: sim.removeModel(sim.getObject('/ServeWater'))
        except Exception as e: pass
        try: sim.removeModel(sim.getObject('/ServeFruit'))
        except Exception as e: pass
        self.colors = ["red", "yellow", "green", "blue", "purple", "black"]
        try:
            # self.blocks = {}
            # self.sensors = {}

            # for color in self.colors:
            #     obj = sim.getObject(f"/{color}_block")
            #     self.blocks[color] = obj
            #     self.sensors[color] = sim.getObject(f"/{color}_block_sensor")
            self.human = sim.getObject('/Bill')
            self.human_goal = sim.getObject('/goalDummy')
            self.serve_1 = sim.getObject('/serve_point_1')
            self.serve_1_sensor = sim.getObject('/serve_point_1_sensor')
            self.serve_2 = sim.getObject('/serve_point_2')
            self.serve_2_sensor = sim.getObject('/serve_point_2_sensor')
            self.tray = sim.getObject('/tray')
            self.tray_ori = sim.getObjectOrientation(self.tray, -1)
        except Exception as e:
            sim.loadModel(path + '/NICOL/StackBlock.ttm')
            self.human = sim.getObject('/Bill')
            self.human_goal = sim.getObject('/goalDummy')
            self.serve_1 = sim.getObject('/serve_point_1')
            self.serve_1_sensor = sim.getObject('/serve_point_1_sensor')
            self.serve_2 = sim.getObject('/serve_point_2')
            self.serve_2_sensor = sim.getObject('/serve_point_2_sensor')
            self.tray = sim.getObject('/tray')
            self.tray_ori = sim.getObjectOrientation(self.tray, -1)
        self.name = f"StackBlocks-{config}"
        self.sim = sim
        self.text_queries = ["human face", "red cube", "green cube", "blue cube", "yellow cube", "purple cube", "black cube"]
        self.owl_threshold = 0.01
        if config == "simple":
            self.name = "StackBlocks-simple"
            self.block_nums = 4
            self.stacking_options = (2,2), (3,1), (4,)
        elif config == "complex":
            self.name = "StackBlocks-complex"
            self.block_nums = 5
            self.stacking_options = (3,2), (4,1), (5,)
        elif config == "super":
            self.name = "StackBlocks-super"
            self.block_nums = 6
            self.stacking_options = (3,3), (4,2), (5,1), (6,)
        self.blocks = {}
        self.positions = {}
        self.areas = {}

    def reset(self, config=None):
        def format_stack(stack):
            return " over ".join(stack[::-1])  # Bottom to top order

        ## import config from outside
        if config:
            self.positions = config["objects"]
            self.stacks = config["goal"]
            self.areas = config["area"]
            self.type = config["type"]
            for color in self.colors:
                pose = list(self.positions[color])
                block = sim.getObject(f"/{color}_block")
                sensor = sim.getObject(f"/{color}_block_sensor")
                sim.setObjectPose(block, -1, pose + [1, 0, 0, 0])
                self.blocks[color] = {
                    "object": block,
                    "sensor": sensor,
                    "position": pose,
                }
            y_human = self.positions["human"][1]
            tray_pose = self.positions["tray"]
            if y_human > 0: y_serve1, y_serve2 = 0.5, 0.35
            elif y_human < 0: y_serve1, y_serve2 = -0.5, -0.35
            else: y_serve1, y_serve2 = 0.1, -0.1
            sim.setObjectPose(self.human_goal, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            # sim.setObjectPose(self.human, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.serve_1, sim.handle_world, [0.75, y_serve1, 0.88, 0,0,0,1])
            sim.setObjectPose(self.serve_2, sim.handle_world, [0.75, y_serve2, 0.88, 0,0,0,1])
            sim.setObjectPosition(self.tray, -1, tray_pose)
            sim.setObjectOrientation(self.tray, -1, self.tray_ori)
            # time.sleep(1)
        ## randomly set up task configs
        else:
            x_coords = [0.4, 0.6]
            y_coords = [-0.65, -0.5, -0.35, -0.15, 0, 0.15, 0.35, 0.5, 0.65]
            # x_coords = [0.4, 0.6]
            # y_coords = [-0.72, -0.6, -0.47, -0.35, -0.15, 0.15, 0.35, 0.47, 0.6, 0.72]
            z_coord = 0.83
            y_human = random.choice([0.7, -0.7, 0])
            sim.setObjectPose(self.human, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.human_goal, sim.handle_world, [3, y_human, 0, 0,0,-1,0])
            sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, y_human, 0, 0,0,-1,0])
            time.sleep(1)
            if y_human > 0: 
                y_tray = 0.45
                y_serve1, y_serve2 = 0.5, 0.35
                self.areas["human"] = "left"
            elif y_human < 0: 
                y_tray = -0.45
                y_serve1, y_serve2 = -0.5, -0.35
                self.areas["human"] = "right"
            else: 
                y_tray = 0
                y_serve1, y_serve2 = -0.1, +0.1
                self.areas["human"] = "center"
            sim.setObjectPosition(self.tray, -1, [0.8, y_tray, 0.84])
            sim.setObjectOrientation(self.tray, -1, self.tray_ori)
            self.positions["human"] = [1.1, y_human, 0]
            self.positions["tray"] = [0.8, y_tray, 0.84]
            sim.setObjectPose(self.serve_1, sim.handle_world, [0.75, y_serve1, 0.88, 0,0,0,1])
            sim.setObjectPose(self.serve_2, sim.handle_world, [0.75, y_serve2, 0.88, 0,0,0,1])
            all_positions = list(itertools.product(x_coords, y_coords))
            random.shuffle(all_positions)
            selected_positions = all_positions[:len(self.colors)]

            for color, (x, y) in zip(self.colors, selected_positions):
                block = sim.getObject(f"/{color}_block")
                sensor = sim.getObject(f"/{color}_block_sensor")
                sim.setObjectPose(block, -1, [x, y, z_coord, 1, 0, 0, 0])
                if y > 0.25: area = "left" 
                elif y < -0.25: area = "right"
                else: area = "center"
                self.blocks[color] = {
                    "object": block,
                    "sensor": sensor,
                    "position": (x, y, z_coord),
                }
                self.areas[color] = area
                self.positions[color] = (x, y, z_coord)

                selected_blocks = random.sample(self.colors, self.block_nums)
                stacking_choice = random.choice(self.stacking_options)
                self.type = stacking_choice
                random.shuffle(selected_blocks)
                index = 0
                self.stacks = []
                for size in stacking_choice:
                    self.stacks.append(selected_blocks[index:index+size])
                    index += size
        # print(self.stacks)
        if len(self.stacks) == 1:
            self.task_orders = f"The task is to stack the selected {self.block_nums} blocks into one pile on the tray right in front of human, where the pile is : {format_stack(self.stacks[0])}."
        else:
            stack_descriptions = " and ".join(f"one pile is {format_stack(stack)}" for stack in self.stacks)
            self.task_orders = f"The task is to stack the selected {self.block_nums} blocks in two piles on the tray right in front of human, where {stack_descriptions}."
        self.task_des = f"""\n
### Task Information: \n
## Task Description: There are several blocks with different color at the table: {self.colors} There is a tray is at the same area where human stands, in which there are two points on it: serve_point_1 and serve_point_2. \n
## Task Goal: {self.task_orders} The block at each pile's bottom should be put at serve points.

## Task related objects: red_block, yellow_block, green_block, blue_block, purple_block, black_block
## Task related points: serve_point_1, serve_point_2, overlap_point_1, overlap_point_2
""" 
        self.task_des_visual = f"""\n
# Based on the vision analyzation, the human is standing in front of the {self.areas["human"]} area, the red_block is at the {self.areas["red"]} area, the green_block is at the {self.areas["green"]} area, the blue_block is at the {self.areas["blue"]} area, the yellow_block is at the {self.areas["yellow"]} area, the purple block is at the {self.areas["purple"]} area, and the black block is at the {self.areas["black"]} area.
"""
        self.config = {"objects": self.positions, "goal": self.stacks, "area":self.areas, "task_des": self.task_des, "task_des_visual": self.task_des_visual, "type": self.type}
    def check_success(self):
        for stack in self.stacks:
            detect_1 = sim.checkProximitySensor(self.serve_1_sensor, self.blocks[stack[0]]['object'])[0]
            detect_2 = sim.checkProximitySensor(self.serve_2_sensor, self.blocks[stack[0]]['object'])[0]
            if (not detect_1) and (not detect_2):
                print(f"Stacking error: the {stack[0]} block is not at the serve point." )
                return False
            if len(stack) > 1: 
                for i in range(len(stack) - 1):
                    lower_block = stack[i]
                    upper_block = stack[i + 1]
                    sensor = self.blocks[lower_block]["sensor"]
                    result = sim.checkProximitySensor(sensor, self.blocks[upper_block]["object"])[0]
                    if not result:
                        print(f"Stacking error: the {upper_block} block is not detected over the {lower_block} block.")
                        return False
        return True
        
    def get_image_from_eyes(self, index):
        head.set_pose_target(NicolPose([0.75, 0.0, 1.1], [0, 0, 0, 0]))
        time.sleep(2)
        self.image = nicol.head_.get_left_eye_camera_img(True, prefix=f"{self.name}_{index}")
        head.set_pose_target(NicolPose([0.75, 0.0, 1.4], [0, 0, 0, 0]))
        return self.image

def create_task(task_name, config=None):
    if task_name == "ServeWater":
        return ServeWaterTask()
    elif task_name == "ServeFruit":
        return ServeFruitTask()
    elif task_name == "StackBlock":
        return StackBlocksTask(config="complex")
    else:
        raise ValueError(f"Unknown task '{task_name}'. Available: ServeWater, ServeFruit, StackBlock")

def generate_visual_description(task, directions):
    if task.name == "ServeWater":
        return f"""
Based on the vision analyzation, the human is standing in the {directions['human face']} area, with serve point also in this area, the blue cup is at the {directions['blue cup']} area, the yellow cup is at the {directions['yellow cup']} area, the brown box is at the {directions['brown box']} area, with the store_point in the central, and the water is inside the blue cup at the present.
        """
    elif task.name == "ServeFruit":
        return f"""
Based on the vision analyzation, the human is standing in the {directions['human face']} area, with the red bowl at the {directions['red bowl']} area, the apple at the {directions['apple']} area, and the banana at the {directions['banana']} area.
        """
    elif task.name.startswith("StackBlocks"):
        return f"""
Based on the vision analyzation, the human is standing in the {directions['human face']} area, the red block is at the {directions['red cube']} area, the green block is at the {directions['green cube']} area, the blue block is at the {directions['blue cube']} area, the yellow block is at the {directions['yellow cube']} area, the purple block is at the {directions['purple cube']} area, and the black block is at the {directions['black cube']} area.
        """