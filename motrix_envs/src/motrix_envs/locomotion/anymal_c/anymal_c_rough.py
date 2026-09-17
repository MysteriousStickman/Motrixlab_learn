# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import gymnasium as gym
import motrixsim as mtx
import numpy as np


from motrix_envs import registry
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState
from .common import generate_repeating_array

from .cfg import AnymalCRoughEnvCfg


@registry.env("anymal_c_navigation_rough", "np")
class AnymalCEnv(NpEnv):
    _cfg: AnymalCRoughEnvCfg

    def __init__(self, cfg: AnymalCRoughEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self.training_level = 0
        self._body = self._model.get_body(cfg.asset.body_name)

        self.foot_geoms = [self._model.get_geom(foot_name) for foot_name in cfg.asset.foot_names]
        self.anymalC_init_height = 0.56
        # Get target marker body
        self._target_marker_body = self._model.get_body("target_marker")

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)  # 动作空间 12维度 每个维度范围[-1, 1]
        # Observation space: linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12) + last_actions(12) +
        # commands(3) + position_error(2) + heading_error(1) + distance(1) + reached_flag(1) + stop_ready_flag(1) = 54
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(74,), dtype=np.float32)
        self._num_dof_pos = self._model.num_dof_pos  # 19  7(body:x,y,z+quat) + 12
        self._num_dof_vel = self._model.num_dof_vel  # 18  6(3个线速度+角速度) + 12
        self._num_action = self._model.num_actuators  # 12

        self._update_dof_pos()
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )
        
        
        self._init_buffer()
        self._init_contact_check()
        self.reset_counter = 0

        self.grid_size = (800, 800)  # nx, ny
        self.hfield = self._model.get_geom("floor_rough").hfield
        self.nx, self.ny = self.grid_size
        self.grid_len_x = (self.hfield.bound[3] - self.hfield.bound[0]) / self.nx
        self.grid_len_y = (self.hfield.bound[4] - self.hfield.bound[1]) / self.ny
        self.terrain_heights = self._generate_terrain_height_map(self.grid_size)
    
    def _update_dof_pos(self):
        self._init_dof_pos = self._model.compute_init_dof_pos()  # 根据xml定义关节顺序与body初始设定,同时根据cfg的各关节初始设定得来
        geom_floor_pos = self._model.get_geom("floor" if self.training_level == 0 else "floor_rough").local_pose[:3]
        geom_floor_pos[2] += self.anymalC_init_height + 0.02
        self._init_dof_pos[:3] = geom_floor_pos


    def _generate_terrain_height_map(self, grid_size: tuple[int, int] = (80, 80)) -> np.ndarray:
        nx, ny = grid_size
        hfield = self.hfield

        # 80 * 80 网格
        idx_x, idx_y = np.meshgrid(
            np.arange(nx),
            np.arange(ny),
            indexing="ij",
        )

        # 映射到 heightmap
        # hfield.ncol : 1024
        # hfield.nrow : 1024
        hfield_idx_x = (
            hfield.ncol * idx_x // nx
        ).astype(int)

        hfield_idx_y = (
            hfield.ncol * idx_y // ny
        ).astype(int)

        # 查询每个网格的地形高度
        heights = np.array(
            [
                hfield.get(iy, ix)
                for ix, iy in zip(
                    hfield_idx_x.flatten(),
                    hfield_idx_y.flatten(),
                )
            ],
            dtype=np.float32,
        )
        heights += self.anymalC_init_height + 0.02

        return heights.reshape(nx, ny)
    

    def _get_terrain_height(self, x, y):
        # 计算复杂地形实际高度
        idx_x = np.clip(np.floor(
            (x - self.hfield.bound[0]) / self.grid_len_x),
            0, self.nx - 1,
        ).astype(int)

        idx_y = np.clip(np.floor(
            (self.hfield.bound[4] - y) / self.grid_len_y),
            0, 
            self.ny - 1

        ).astype(int)
        terrain_height = self.terrain_heights[idx_x, idx_y]
        return terrain_height
    
    def _get_terrain_slope(self, x, y):
        """计算指定(x,y)点处的地形坡度 (dh/dx, dh/dy)"""
        dx = self.grid_len_x
        dy = self.grid_len_y
        
        # 前后差分
        h_plus_x = self._get_terrain_height(x + dx, y)
        h_minus_x = self._get_terrain_height(x - dx, y)
        dh_dx = (h_plus_x - h_minus_x) / (2 * dx)
        
        h_plus_y = self._get_terrain_height(x, y + dy)
        h_minus_y = self._get_terrain_height(x, y - dy)
        dh_dy = (h_plus_y - h_minus_y) / (2 * dy)
        
        return dh_dx, dh_dy


    def _generate_init_offsets(self, init_height: float, grid_size: tuple[int, int] = (10, 10)) -> np.ndarray:
        """Generate initialization offsets for rough terrain training."""
        hfield = self._model.get_geom("floor_rough").hfield
        nx, ny = grid_size

        # Vectorized grid generation
        idx_x, idx_y = np.meshgrid(np.arange(nx), np.arange(ny, dtype=np.float32), indexing="ij")
        idx_x, idx_y = idx_x.flatten(), idx_y.flatten()

        # Map to heightfield indices and get heights
        hfield_idx_x = (hfield.ncol * idx_x // nx).astype(int)
        hfield_idx_y = (hfield.nrow * idx_y // ny).astype(int)
        heights = np.array([hfield.get(iy, ix) for ix, iy in zip(hfield_idx_x, hfield_idx_y)])

        # Calculate positions
        grid_len_x = (hfield.bound[3] - hfield.bound[0]) / nx
        grid_len_y = (hfield.bound[4] - hfield.bound[1]) / ny
        center_offset = np.array([grid_len_x / 2, grid_len_y / 2])
        
        offsets = np.column_stack(
            [
                hfield.bound[0] + grid_len_x * idx_x + center_offset[0],
                hfield.bound[4] - grid_len_y * idx_y - center_offset[1],
                heights + init_height,
            ]
        ).astype(np.float32)
        return offsets


    def _init_buffer(self):
        """
        主要是根据cfg中的关节位置，初始化机器人关节位置
        """
        cfg = self._cfg

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        # PD parameters controlled by kp and kv in XML

        # Normalization coefficients
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel], dtype=np.float32
        )

        # Set default joint angles
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle

        self._init_dof_pos[-self._num_action :] = self.default_angles
    
    def _init_contact_check(self):
        cfg = self._cfg
        self.ground = []
        ground_name = cfg.asset.ground_plane_name if self.training_level == 0 else cfg.asset.ground_rough_name
        for geom_name in self.model.geom_names:
            if geom_name is not None and geom_name in ground_name:
                self.ground.append(self._model.get_geom_index(geom_name))
        
        self.termination_contact = None
        for ground_index in self.ground:
            for base_name in cfg.asset.terminate_after_contacts_on:
                if self.termination_contact is None:
                    self.termination_contact = np.array(
                        [[self._model.get_geom_index(base_name), ground_index]], dtype=np.uint32
                    )
                else:
                    self.termination_contact = np.append(
                        self.termination_contact,
                        np.array(
                            [[self._model.get_geom_index(base_name), ground_index]],
                            dtype=np.uint32,
                        ),
                        axis=0,
                    )

        self.num_termination_check = self.termination_contact.shape[0]

        self.foot_contact_check = None
        for ground_index in self.ground:
            for foot_name in cfg.asset.foot_names:
                if self.foot_contact_check is None:
                    self.foot_contact_check = np.array(
                        [[self._model.get_geom_index(foot_name), ground_index]], dtype=np.uint32
                    )
                else:
                    self.foot_contact_check = np.append(
                        self.foot_contact_check,
                        np.array(
                            [[self._model.get_geom_index(foot_name), ground_index]],
                            dtype=np.uint32,
                        ),
                        axis=0,
                    )
        self.foot_check_num = self.foot_contact_check.shape[0]
        

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 实际动作空间
        # Save current action for incremental control
        if "current_actions" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions

        # Position control mode: directly input target angles
        actions_scaled = actions * self._cfg.control_config.action_scale
        state.data.actuator_ctrls = self.default_angles + actions_scaled
        return state

    def update_state(self, state: NpEnvState):
        data = state.data
        # update obs
        obs, obs_info = self._get_obs(data, state.info)
        desired_vel_xy = obs_info["desired_vel_xy"]
        velocity_commands = obs_info["velocity_commands"]
        
        state.info["foot_heighs"] = obs_info["foot_heighs"]
        state.info["foot_slope"] = obs_info["foot_slope"]
        state.info["foot_contact"] = obs_info["foot_contact"]
        state.info["feet_air_time"] = obs_info["feet_air_time"]
        state.info["foot_pose"] = obs_info["foot_pose"]

        pose_commands = state.info["pose_commands"]
        state.info["desired_vel_xy"] = desired_vel_xy
        state.info["contacts"] = obs_info["foot_contact"]
        # training_level: 0 or 1 可以都要
        state.info["feet_air_time"] = obs_info["feet_air_time"]
        state.info["last_feet_air_time"] = obs_info["last_feet_air_time"]


        # Update target position marker
        self._update_target_marker(data, pose_commands)

        # Update arrow visualization (no physical effect)
        base_lin_vel = self.get_local_linvel(data)
        pose = self._body.get_pose(data)
        
        root_pos = pose[:, :3]
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel_xy)
        
        # update termination conditions
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated

        # update reward
        reward_dict = self._compute_reward(data, state.info, velocity_commands)
        
        reward_dict_scaled = {k: self.cfg.reward_config.scales.get(k, 0.0) * v for k, v in reward_dict.items()}
        reward = sum(reward_dict_scaled.values())
        reward = np.clip(reward, 0.0, 10000.0)
        rwd = np.where(terminated, np.array(0.0), reward)
        average_reward = np.average(rwd)
        if average_reward > 2.0 and self.training_level == 0: 
            print(f"average_reward: {average_reward}, learning_level: 0 -> 1   !!! ")
            self.training_level = 1 
            self._update_dof_pos()
            self._init_contact_check()

            
        # 用于调试新增奖励 缩放系数
        # if state.info["steps"][0] % 40 == 0:
        #     own_reward_keys = ["feet_air_time_reward", "base_height_reward", "lateral_penalty"]
        #     reward_dict_info = {k: np.mean(v) for k, v in reward_dict.items()}
        #     reward_dict_info = {k: self.cfg.reward_config.scales.get(k, 0.0) * v for k, v in reward_dict_info.items()}
        #     print(f"average_reward: {average_reward}, ---------------, training_level:{self.training_level}")
        #     print(f"steps[0]: {state.info['steps'][0]}")
            
        #     own_reward_info = {k: v for k, v in reward_dict_info.items() if k in own_reward_keys}
        #     other_reward_info = {k: v for k, v in reward_dict_info.items() if k not in own_reward_keys}
        #     values = list(other_reward_info.values())
        #     print(f"own_reward_info: {own_reward_info}")
        #     print('-----------')
        #     print(f"max_other: {max(values)}, --- average: {sum(values) / len(values)}, -----  min: {min(values)}")


        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        return state

     

    def update_feet_air_time(self, info: dict):
        last_feet_air_time = info["last_feet_air_time"]
        feet_air_time = info["feet_air_time"]
        feet_air_time += self.cfg.ctrl_dt
        contacts_idx = info["contacts"] & (feet_air_time > 0)
        last_feet_air_time[contacts_idx] = feet_air_time[contacts_idx]
        feet_air_time *= ~info["contacts"]
        return feet_air_time, last_feet_air_time


    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
    
    def _get_robot_state(self, data: mtx.SceneData):
        pose = self._body.get_pose(data)


        root_quat = pose[:, 3:7]

        base_lin_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)

        joint_vel = self._body.get_joint_dof_vel(data)

        projected_gravity = self._compute_projected_gravity(root_quat)
        return {
            "base_lin_vel": base_lin_vel,
            "gyro": gyro,
            "joint_vel": joint_vel,
            "projected_gravity": projected_gravity,
        }
    


    def _get_target_state(self, data: mtx.SceneData, info: dict):
        pose = self._body.get_pose(data)

        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]

        robot_position = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)

        pose_commands = info["pose_commands"]

        position_error = pose_commands[:, :2] - robot_position

        heading_diff = pose_commands[:, 2] - robot_heading

        heading_diff = np.where(
            heading_diff > np.pi,
            heading_diff - 2 * np.pi,
            heading_diff,
        )
        heading_diff = np.where(
            heading_diff < -np.pi,
            heading_diff + 2 * np.pi,
            heading_diff,
        )

        distance_to_target = np.linalg.norm(
            position_error,
            axis=1,
        )

        reached_position = distance_to_target < 0.3
        reached_heading = np.abs(heading_diff) < np.deg2rad(15)

        reached_all = reached_position & reached_heading
        return  {
            "root_pos": root_pos,
            "root_quat": root_quat,
            "position_error": position_error,
            "heading_diff": heading_diff,
            "distance": distance_to_target,
            "reached_position": reached_position,
            "reached_heading": reached_heading,
            "reached_all": reached_all,
        }
        
    def _get_foot_height(self, foot_poses):
        # 获取机器人脚下地形高度
        
        heights = self._get_terrain_height(foot_poses[:, :, 0], foot_poses[:, :, 1])  # num_envs, 4
        return heights
    
    def _get_foot_slope(self, foot_poses):
        dh_dx, dh_dy = self._get_terrain_slope(foot_poses[:, :, 0], foot_poses[:, :, 1])  # num_envs, 4     num_envs, 4
        return np.stack([dh_dx, dh_dy], 2)  # num_envs, 4, 2

    
        

    def _get_obs(self, data: mtx.SceneData, info: dict):  
        obs_info = {}
        target = self._get_target_state(data, info)         
        position_error = target["position_error"]
        heading_diff = target["heading_diff"]
        distance_to_target = target["distance"]
        reached_position = target["reached_position"]
        reached_all = target["reached_all"]

        root = self._get_robot_state(data)
        base_lin_vel = root["base_lin_vel"]
        gyro = root["gyro"]
        projected_gravity = root["projected_gravity"]
        joint_vel = root["joint_vel"]


        joint_pos_rel = self._body.get_joint_dof_pos(data) - self.default_angles

        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        desired_yaw_rate = np.where(np.abs(heading_diff) < np.deg2rad(8), 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)
        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)

        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = info["current_actions"]

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)
        reached_flag = reached_all.astype(np.float32)
        stop_ready_flag = (reached_all & (np.abs(gyro[:, 2]) < 5e-2)).astype(np.float32)

        # 机器人脚下地形高度: 决定下一步抬多高
        foot_poses = np.stack([foot_geom.get_pose(data) for foot_geom in self.foot_geoms], axis=1)  # num_envs, 4, 7
        foot_heighs = self._get_foot_height(foot_poses)  # num_envs 4

        # 机器人脚下地形坡度：决定身体姿态和步幅
        foot_slope = self._get_foot_slope(foot_poses)  # num_envs, 4, 2

        # 足端接触状态： 决定步状时序
        cquerys = self._model.get_contact_query(data)
        foot_contact = cquerys.is_colliding(self.foot_contact_check)
        try:
            foot_contact = foot_contact.reshape((data.shape[0], self.foot_check_num))  # num_envs, 4
        except:
            foot_contact = 0
            breakpoint()
            # breakpoint()
            temp = 1

        # 足端离地时间： 学会合理控制步状周期
        feet_air_time, last_feet_air_time = self.update_feet_air_time(info)  # num_envs, 4
        
        obs = np.hstack(
            [
                noisy_linvel,                           # 3   线速度
                noisy_gyro,                             # 3   角速度
                projected_gravity,                      # 3   重力投影
                noisy_joint_angle,                      # 12  关节位置
                noisy_joint_vel,                        # 12  关节速度
                last_actions,                           # 12  上一时刻动作
                command_normalized,                     # 3   速度命令
                position_error_normalized,              # 2   位置误差
                heading_error_normalized[:, None],      # 1   朝向误差
                distance_normalized[:, None],           # 1   距离
                reached_flag[:, None],                  # 1   是否到达
                stop_ready_flag[:, None],               # 1   是否停稳
                foot_heighs,                            # 4   脚下地形高度
                foot_slope.reshape(data.shape[0], -1), # 8   脚下地形坡度
                foot_contact,                           # 4   足端接触状态
                feet_air_time                           # 4   足端离地时间
            ]
        )

        assert obs.shape == (data.shape[0], 74)
        obs_info["desired_vel_xy"] = desired_vel_xy
        obs_info["velocity_commands"] = velocity_commands
        obs_info["foot_heighs"] = foot_heighs
        obs_info["foot_slope"] = foot_slope
        obs_info["foot_contact"] = foot_contact
        obs_info["feet_air_time"] = feet_air_time
        obs_info["last_feet_air_time"] = last_feet_air_time
        obs_info["foot_pose"] = foot_poses
        return obs, obs_info

    

    def _update_heading_arrows(
        self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray
    ):
        """
        Update arrow positions (using DOF to control freejoint, no physical effect)
        robot_pos: [num_envs, 3] - Robot position
        desired_vel_xy: [num_envs, 2] - Desired linear velocity (ground coordinates)
        base_lin_vel_xy: [num_envs, 2] - Actual linear velocity (ground coordinates)
        """

        arrow_height = 0.76  # Arrow height (base=0.56 + 0.2)
        cur_yaw = np.where(
            np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3,
            np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
            0.0,
        )
        robot_arrow_pos = robot_pos.copy()
        if self.training_level == 0:
            robot_arrow_pos[:, 2] = arrow_height
        else:
            heights = self._get_terrain_height(robot_arrow_pos[:, 0], robot_arrow_pos[:, 1])
            robot_arrow_pos[:, 2] = heights + arrow_height
        
        robot_arrow_quat = quaternion.from_euler(0, 0, cur_yaw)
        mocap = self._model.get_body("robot_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, robot_arrow_quat], axis=1))

        des_yaw = np.where(
            np.linalg.norm(desired_vel_xy, axis=1) > 1e-6, np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]), 0.0
        )
        desired_arrow_quat = quaternion.from_euler(0, 0, des_yaw)
        mocap = self._model.get_body("desired_heading_arrow").mocap
        mocap.set_pose(data, np.concatenate([robot_arrow_pos, desired_arrow_quat], axis=1))


    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        """
        Velocity tracking reward mechanism
        velocity_commands: [num_envs, 3] - (vx, vy, vyaw)
        """
        target = self._get_target_state(data, info)         
        distance_to_target = target["distance"]
        reached_all = target["reached_all"]

        # Calculate termination condition penalties
        termination_mask = self._get_termination_masks(data)

        termination_penalty = np.where(
            termination_mask["terminated"],
            -20.0,
            0.0,
        )

        root = self._get_robot_state(data)
        base_lin_vel = root["base_lin_vel"]
        gyro = root["gyro"]
        projected_gravity = root["projected_gravity"]
        joint_vel = root["joint_vel"]


        # 1. Linear velocity tracking reward
        # 线速度跟踪奖励：鼓励机器人按照期望的线速度运动，减少误差
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)  # 机器人线速度误差
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)  # tracking_sigma = 0.25

        # 2. Angular velocity tracking reward / heading error penalty (mixed strategy)
        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])  # 机器人角速度误差
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)


        # One-time reward for first time reaching position
        info["ever_reached"] = info.get("ever_reached", np.zeros(self._num_envs, dtype=bool))
        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)
        # 首次到达奖励一次大的
        arrival_bonus = np.where(first_time_reach, 10.0, 0.0)

        # Distance approach reward: incentivize getting closer to target
        # Use historical minimum distance to calculate progress
        # 距离接近奖励：鼓励机器人接近目标位置
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # Reward 5 points for every 1 meter closer

        # 3. Orientation stability reward (penalize deviation from normal standing posture)
        # When standing normally, projected_gravity ≈ [0, 0, -1]
        # 重力投影：正常情况下是:[0, 0, -1]
        # 重力投影各分量平方和，最小值为0。此项目为负奖励，鼓励机器人保持平稳姿态

        orientation_penalty = (
            np.square(projected_gravity[:, 0])
            + np.square(projected_gravity[:, 1])
            + np.square(projected_gravity[:, 2] + 1.0)
        )

        # Arrival and stop determination (reward bonus)
        # 机器人平稳，不左右晃动
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)  # 机器人线速度大小
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05  # Relax to 0.05 rad/s ≈ 2.86°/s  机器人角速度:[num_envs, 3]   
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)  # 平稳到达奖励
        # 线速度、角速度越小，奖励越大，鼓励机器人停稳
        stop_base = 2 * (0.8 * np.exp(-((speed_xy / 0.2) ** 2)) + 1.2 * np.exp(-((np.abs(gyro[:, 2]) / 0.1) ** 4)))  
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

        # 4. Z-axis linear velocity penalty
        # Z轴线速度惩罚，鼓励机器人保持稳定的高度，避免跳跃或下落
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])

        # 5. XY-axis angular velocity penalty
        # XY轴角速度惩罚，鼓励机器人保持稳定的姿态，避免翻滚或旋转
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)

        # 6. Torque penalty
        # 力矩惩罚：节能、动作平滑、保护硬件
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)

        # 7. Joint velocity penalty
        # 关节速度惩罚
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        # 8. Action change penalty
        # 动作变化惩罚：鼓励动作平滑，避免过度抖动
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # Combined reward
        # After reaching: stop all positive rewards, only keep stop reward and penalties

        # 复杂地形专有奖励
        # 步态奖励
        feet_air_time_reward = self._reward_feet_air_time(data, info)

        # 地形跟随惩罚
        base_height_reward = self._reward_base_height(data, info)

        # 惩罚横向穿越陡坡的行为，鼓励沿坡度梯度方向运动
        lateral_penalty = self._reward_slope_direction_consistency(data, info)
        
        

        reward_dict_reached = {
            "stop_bonus": stop_bonus,
            "arrival_bonus": arrival_bonus,

            "lin_vel_z_penalty": lin_vel_z_penalty if self.training_level == 0 else np.zeros_like(lin_vel_z_penalty),
            "ang_vel_xy_penalty": ang_vel_xy_penalty,
            "orientation_penalty": orientation_penalty,
            "torque_penalty": torque_penalty,
            "dof_vel_penalty": dof_vel_penalty,
            "action_rate_penalty": action_rate_penalty,
            "termination_penalty": termination_penalty,
            "feet_air_time_reward": feet_air_time_reward,
            "base_height_reward": base_height_reward if self.training_level == 1 else np.zeros_like(base_height_reward),
            "lateral_penalty": lateral_penalty if self.training_level == 1 else np.zeros_like(lateral_penalty)
        }

        reward_dict_not_reached = {
            "tracking_lin_vel": tracking_lin_vel,
            "tracking_ang_vel": tracking_ang_vel,
            "approach_reward": approach_reward,

            "lin_vel_z_penalty": lin_vel_z_penalty if self.training_level == 0 else np.zeros_like(lin_vel_z_penalty),
            "ang_vel_xy_penalty": ang_vel_xy_penalty,
            "orientation_penalty": orientation_penalty,
            "torque_penalty": torque_penalty,
            "dof_vel_penalty": dof_vel_penalty,
            "action_rate_penalty": action_rate_penalty,
            "termination_penalty": termination_penalty,
            "feet_air_time_reward": feet_air_time_reward,
            "base_height_reward": base_height_reward if self.training_level == 1 else np.zeros_like(base_height_reward),
            "lateral_penalty": lateral_penalty if self.training_level == 1 else np.zeros_like(lateral_penalty)
        }

        reward_dict = {
            k: np.where(
                reached_all, 
                reward_dict_reached.get(k, 0.0),
                reward_dict_not_reached.get(k, 0.0)
            ) for k in set(reward_dict_reached) | set(reward_dict_not_reached)
        }

        return reward_dict
    
    def _reward_feet_air_time(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """
        导航任务自适应步态腾空时间奖励
        """
        last_air_time = info["last_feet_air_time"]       # 触地瞬间保存的腾空时间
        first_contact = (last_air_time > 0.0) * info["contacts"]
        # 第一次触地时触发奖励
        rew_airTime = np.sum((last_air_time - 0.5) * first_contact, axis=1)
        reached_all = self._get_target_state(data, info)["reached_all"]
        rew_airTime *= ~reached_all
        return rew_airTime
    
    def _reward_base_height(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        target_height = 0.5
        pose = self._body.get_pose(data)
        rebot_terrain_height = self._get_terrain_height(pose[:, 0], pose[:, 1])
        terrain_clearance = pose[:, 2] - rebot_terrain_height
        terrain_follow_penalty = np.square(terrain_clearance - target_height) 
        return terrain_follow_penalty

    def _reward_slope_direction_consistency(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """
        惩罚横向穿越陡坡的行为，鼓励沿坡度梯度方向运动
        """
        slopes = info["foot_slope"]  # [N, 4, 2]
        mean_slope = np.mean(slopes, axis=1)  # [N, 2]
        slope_magnitude = np.linalg.norm(mean_slope, axis=1)  # [N]
        
        # 只在有明显坡度时才施加此奖励
        significant_slope_mask = slope_magnitude > 0.1
        
        # 获取机器人本地线速度 xy
        root = self._get_robot_state(data)
        vel_xy = root["base_lin_vel"][:, :2]  # [N, 2]
        vel_norm = np.linalg.norm(vel_xy, axis=1, keepdims=True) + 1e-8
        vel_dir = vel_xy / vel_norm
        
        # 坡度梯度方向
        slope_dir = mean_slope / (slope_magnitude[:, None] + 1e-8)
        
        # 速度与坡度梯度的点积
        # 正相关 = 沿坡上下; 负相关/零 = 横切
        dot_product = np.sum(vel_dir * slope_dir, axis=1)
        
        # 横切惩罚: 当坡度大且dot_product接近0时惩罚
        lateral_penalty = significant_slope_mask * (1.0 - np.abs(dot_product)) * slope_magnitude
        
        return lateral_penalty  # 返回负值作为惩罚

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """
        Update position and orientation of target marker
        """
        num_envs = data.shape[0]
        arrow_pos = np.column_stack([pose_commands[:, 0], pose_commands[:, 1], np.full((num_envs, 1), 3.0)])
        if self.training_level == 1:
            heights = self._get_terrain_height(arrow_pos[:, 0], arrow_pos[:, 1])
            arrow_pos[:, 2] = heights + 0.76
 
        arrow_quat = quaternion.from_euler(0, 0, pose_commands[:, 2])
        mocap = self._model.get_body("target_marker").mocap
        mocap.set_pose(data, np.concatenate([arrow_pos, arrow_quat], axis=1))


    def _get_termination_masks(self, data: mtx.SceneData):
        dof_vel = self._body.get_joint_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel  # 速度过大
        # Extreme velocity/NaN/Inf protection
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)  # 仿真崩掉

        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_contact)
        termination_check = termination_check.reshape((self._num_envs, self.num_termination_check))
        base_contact = termination_check.any(axis=1)  # 机器人倒地

        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip = tilt_angle > np.deg2rad(75)  # 机器人过度倾斜

        terminated = (
            vel_overflow
            | vel_extreme
            | base_contact
            | side_flip
        )

        return {
            "vel_overflow": vel_overflow,
            "vel_extreme": vel_extreme,
            "base_contact": base_contact,
            "side_flip": side_flip,
            "terminated": terminated,
        }


    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        mask = self._get_termination_masks(data)

        return state.replace(terminated=mask["terminated"])

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        cfg: AnymalCRoughEnvCfg = self._cfg
        num_envs = data.shape[0]

        # ---------------------------随机化重置初始命令----------------------------
        # First generate robot initial position (in world coordinates)
        # 命令初始位置随机范围：[-10, 10]，[-10, 10]
        pos_range = cfg.init_state.pos_randomization_range
        robot_offset_init_x = np.random.uniform(
            pos_range[0],
            pos_range[2],  # x_min, x_max
            num_envs,
        ).astype(np.float32)
        robot_offset_init_y = np.random.uniform(
            pos_range[1],
            pos_range[3],  # y_min, y_max
            num_envs,
        ).astype(np.float32)
        robot_offset_pos = np.stack([robot_offset_init_x, robot_offset_init_y], axis=1)  # [num_envs, 2]

        

        # ---------------------------重置机器人初始状态：随机化位置、初始化朝向、关节---------------------------
        # ---------------------------重置机器人初始速度：全为0---------------------------
        # Set initial state - avoid adding noise to quaternion
        # 
        init_dof_pos = np.tile(self._init_dof_pos, (*data.shape, 1))  # [num_envs, 19] 7(body) + 12(joint)
        init_dof_vel = np.tile(self._init_dof_vel, (*data.shape, 1))  # [num_envs, 18] 6(body) + 12(joint)

        # Generate target position: offset relative to robot initial position
        # pose_command_range now represents offset range relative to robot
        # 命令初始位置xy施加偏移，偏移范围为[-5, 5]
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2], high=cfg.commands.pose_command_range[3:5], size=(num_envs, 2)
        ).astype(np.float32)
        target_positions = init_dof_pos[:, :2] + target_offset  # Target position in world coordinates

        # Generate target heading (absolute heading, random in horizontal direction)
        # 命令朝向施加偏移，偏移范围为[-pi, pi]
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2], high=cfg.commands.pose_command_range[5], size=(num_envs, 1)
        ).astype(np.float32)

        pose_commands = np.concatenate([target_positions, target_headings], axis=1)
 
        # Create noise - do not add noise to quaternion
        noise_pos = np.zeros((*data.shape, self._num_dof_pos), dtype=np.float32)

        # Base position (DOF 0-2): use the generated random initial position
        noise_pos[:, 0] = robot_offset_init_x - cfg.init_state.pos[0]  # Offset from default position
        noise_pos[:, 1] = robot_offset_init_y - cfg.init_state.pos[1]
        # No noise on Z axis, maintain fixed height to avoid falling feeling

        # All velocities set to 0, ensure completely stationary
        noise_vel = np.zeros((*data.shape, self._num_dof_vel), dtype=np.float32)

        # 机器人随机初始位置
        dof_pos = init_dof_pos  #  + noise_pos
        # 机器人初c速度全为0
        dof_vel = init_dof_vel + noise_vel


        if self.training_level == 1:
            dof_pos[:, :2] += robot_offset_pos
            x_final = dof_pos[:, 0]
            y_final = dof_pos[:, 1]
            dof_pos[:, 2] = self._get_terrain_height(x_final, y_final)


        
        
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        # Update target position marker
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros(
                (num_envs, self._num_action),
                dtype=np.float32,
            ),
            "current_actions": np.zeros(
                (num_envs, self._num_action),
                dtype=np.float32,
            ),
            "ever_reached": np.zeros(
                num_envs,
                dtype=bool,
            ),
            "min_distance": np.full(
                num_envs,
                np.inf,
                dtype=np.float32,
            ),
            "feet_air_time": np.zeros(
                (num_envs, self.foot_check_num),
                dtype=np.float32,
            ),
            "last_feet_air_time": np.zeros(
                (num_envs, self.foot_check_num),
                dtype=np.float32,
            ),
            "contacts": np.zeros((num_envs, self.foot_check_num), dtype=np.bool),
            "foot_heighs": np.zeros((num_envs, self.foot_check_num), dtype=np.float32),
            "foot_slope": np.zeros((num_envs, self.foot_check_num, 2), dtype=np.float32),
            "foot_contact": np.ones((num_envs, self.foot_check_num), dtype=np.bool), 
            "foot_pose": np.ones((num_envs, self.foot_check_num, 7), dtype=np.float32),  
        }


        self._update_target_marker(
            data,
            pose_commands,
        )
        obs, obs_info = self._get_obs(
            data,
            info,
        )

        info["desired_vel_xy"] = obs_info["desired_vel_xy"]
        
        return obs, info

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return quaternion.rotate_vector(quat, gravity)
