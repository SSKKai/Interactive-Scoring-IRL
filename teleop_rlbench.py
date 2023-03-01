import math
import os
import time
import torch
import numpy as np
from argparse import ArgumentParser
from human_feedback import correct_action
from utils import KeyboardObserver, TrajectoriesDataset, loop_sleep
from custom_env import CustomEnv
from pyquaternion import Quaternion
import argparse

config = {
    'task': "PushButton",  #
    'static_env': False,  #
    'headless_env': False,  #
    'save_demos': True,  #
    'learn_reward_frequency': 100,  #
    'episodes': 3,  #
    'sequence_len': 150,  #
    'obs_type': "LowDimension"  # LowDimension WristCameraRGB
}
config = argparse.Namespace(**config)

state_list = []

env = CustomEnv(config)
keyboard_obs = KeyboardObserver()
env.reset()
gripper_open = 0.9
time.sleep(5)
print("Go!")
episodes_count = 0
first_flag = 0
while episodes_count < config.episodes:
    start_time = time.time()
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open])

    if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
        action = correct_action(keyboard_obs, action)
        gripper_open = action[-1]
    next_state, reward, done, state = env.step(action)
    ee_pose = np.array([getattr(state, 'gripper_pose')[:3]])
    target_pose = np.array([getattr(state, 'task_low_dim_state')])
    distance = np.sqrt((target_pose[0, 0] - ee_pose[0, 0]) ** 2 + (target_pose[0, 1] - ee_pose[0, 1]) ** 2 + (
                target_pose[0, 2] - ee_pose[0, 2]) ** 2)
    touch_force = env.task._task.robot.gripper.get_touch_sensor_forces()
    x, y, z, qx, qy, qz, qw = env.task._robot.arm.get_tip().get_pose()

    wp_x, wp_y, wp_z = env.task._task.get_low_dim_state()[21:24]

    xy_distance = np.sqrt((x - wp_x) ** 2 + (y - wp_y) ** 2)
    z_distance = np.sqrt((z - wp_z) ** 2)

    # # PushButton rotation distance
    arm_rot = Quaternion(qw, qx, qy, qz)
    arm_rot = arm_rot*Quaternion(0.7071,0,-0.7071,0)
    tar_rot = Quaternion(0,0,1,0)
    trans_rot = arm_rot.inverse*tar_rot
    std_vec = [0,0,1]
    arm_vec = trans_rot.rotate(std_vec)
    rot_distance = np.arccos(np.clip(np.dot(arm_vec, std_vec), -1.0, 1.0))
    print(reward)

    # # CloseMicrowave rotation distance
    # arm_rot = Quaternion(qw, qx, qy, qz)
    # tar_rot = Quaternion(target_pose[0,56], target_pose[0,53],target_pose[0,54],target_pose[0,55])
    #
    # dx, dy, dz = tar_rot.rotate([1, 0, 0])
    # vx, vy, vz = env.task._robot.gripper.get_velocity()[0]
    # angle = np.arccos(np.clip(np.dot([dx, dy], [vx/np.sqrt(vx**2+vy**2), vy/np.sqrt(vx**2+vy**2)]), -1.0, 1.0))
    # velocity = np.sqrt(vx**2+vy**2)*math.cos(angle)
    # q_tran = Quaternion(0.0, -0.0, -0.7071067811865475, -0.7071067811865475)
    # rot_distance = Quaternion.absolute_distance(arm_rot, tar_rot*q_tran)
    # print(rot_distance)

    # print(xy_distance, z_distance)
    #
    # state_list.append(target_pose)

    if keyboard_obs.reset_button:
        env.reset()
        gripper_open = 0.9
        keyboard_obs.reset()
    # elif done:
    #     env.reset()
    #     gripper_open = 0.9
    #     episodes_count += 1
    #     first_flag = 0
    #     keyboard_obs.reset()
    #     done = False
    else:
        loop_sleep(start_time)

ll_prime = [state_list[0], state_list[-1]]
ll_prime = np.array(ll_prime)
ll_prime = ll_prime.squeeze(axis=1)

target_pose = target_pose.squeeze(axis=0)
q_list = []
for i in range(0,len(target_pose)-3):
    if abs(1-(target_pose[i]**2+target_pose[i+1]**2+target_pose[i+2]**2+target_pose[i+3]**2)) <0.001:
        print(i)
        q = [target_pose[i],target_pose[i+1],target_pose[i+2],target_pose[i+3]]
        q_list.append(q)

