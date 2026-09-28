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

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math import quaternion
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection01EnvCfg



@registry.env("vbot_navigation_section01", "np")
class VBotSection01Env(NpEnv):
    """
    VBot在Section001地形上的导航任务
    继承自NpEnv，使用VBotSection001EnvCfg配置
    """
    _cfg: VBotSection01EnvCfg
    ROUGH_Y_RANGE = (-1.65, 1.85)
    PIT_Y_RANGE = (-1.5, 1.5)
    TERRAIN_SCAN_OFFSETS = np.asarray([0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60], dtype=np.float32)
    
    
    def __init__(self, cfg: VBotSection01EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        self.section_idx = 1
        self.target_positions1 = np.array([[0.0, -0.6], [0.0, 1.4], [0.0, 3.4], [0.0, 5.4], [0.0, 8.0]], dtype=np.float32)
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self.num_foot_check = 4 
        self._joint_noise_scale = 0.03
        self.position_threshold = 0.2
        self._waypoint_lin_kp = float(getattr(cfg.commands, "waypoint_lin_kp", 0.8))
        self._waypoint_ang_kp = float(getattr(cfg.commands, "waypoint_ang_kp", 1.0))
        self._vel_low = np.asarray(getattr(cfg.commands, "vel_limit", ((-0.35, -0.16, -0.45), (0.55, 0.16, 0.45)))[0], dtype=np.float32)
        self._vel_high = np.asarray(getattr(cfg.commands, "vel_limit", ((-0.35, -0.16, -0.45), (0.55, 0.16, 0.45)))[1], dtype=np.float32)
        self._rough_vel_low = np.asarray(getattr(cfg.commands, "rough_vel_limit", ((-0.16, -0.10, -0.85), (0.82, 0.20, 0.65)))[0], dtype=np.float32)
        self._rough_vel_high = np.asarray(getattr(cfg.commands, "rough_vel_limit", ((-0.16, -0.10, -0.35), (0.82, 0.20, 0.65)))[1], dtype=np.float32)
        self._command_smooth_alpha = float(np.clip(cfg.ctrl_dt / (0.25 + cfg.ctrl_dt), 0.0, 1.0))
        wait_seconds = max(float(getattr(cfg.commands, "waypoint_wait_seconds", 0.5)), 0.0)
        self._waypoint_wait_steps = 0 if wait_seconds <= 0.0 else max(1, int(round(wait_seconds / max(cfg.ctrl_dt, 1e-6))))

        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：67维（55 + 12维接触力）
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos  # 36
        self._num_dof_vel = self._model.num_dof_vel  # 33
        self._num_action = self._model.num_actuators  # 12
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        self._init_termination_contact()
        self._init_undesired_contact()
        
        # 初始位置生成参数：从配置文件读取
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)  # 从配置读取
        self.spawn_range = 0.1  # 随机生成范围：±0.1m（0.2m×0.2m区域）
    
        # 导航统计计数器
        self.navigation_stats_step = 0
    

    def _init_undesired_contact(self):
        geom_names = list(getattr(self._cfg.asset, "undesired_contacts_on", []))
        ground_prefixes = ["C1_", "C2_", "C3_"]
        ground_geoms = []
        for geom_name in self._model.geom_names:
            if geom_name is not None and any(prefix in geom_name for prefix in ground_prefixes):
                ground_geoms.append(self._model.get_geom_index(geom_name))
        ground_geoms = list(dict.fromkeys(ground_geoms))

        pairs = []
        for geom_name in geom_names:
            try:
                geom_idx = self._model.get_geom_index(geom_name)
            except BaseException:
                continue
            for ground_idx in ground_geoms:
                pairs.append([geom_idx, ground_idx])
        self.undesired_contact = np.asarray(pairs, dtype=np.uint32) if pairs else np.zeros((0, 2), dtype=np.uint32)
        self.num_undesired_contact_check = int(self.undesired_contact.shape[0])
    

    def _get_undesired_contacts(self, data: mtx.SceneData) -> np.ndarray:
        if self.num_undesired_contact_check <= 0:
            return np.zeros((data.shape[0],), dtype=np.float32)
        cquerys = self._model.get_contact_query(data)
        contact = cquerys.is_colliding(self.undesired_contact)
        contact = contact.reshape((data.shape[0], self.num_undesired_contact_check))
        return contact.any(axis=1).astype(np.float32)
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        
        # self._init_dof_pos[-self._num_action:] = self.default_angles
        # self._init_dof_pos[10:22] = self.default_angles
        self.action_filter_alpha = 0.3
    
    def _rough_zone_mask(self, base_xy: np.ndarray) -> np.ndarray:
        return np.logical_and(base_xy[:, 1] >= self.ROUGH_Y_RANGE[0], base_xy[:, 1] <= self.ROUGH_Y_RANGE[1])
    
    def _command_limits(self, base_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = np.tile(self._vel_low, (base_xy.shape[0], 1))
        high = np.tile(self._vel_high, (base_xy.shape[0], 1))
        rough = self._rough_zone_mask(base_xy)
        if np.any(rough):
            low[rough] = self._rough_vel_low
            high[rough] = self._rough_vel_high
        return low, high
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        arrow_init_height = self._cfg.init_state.pos[2] + 0.5 
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
     

    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info["current_actions"]
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"]

        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 80.0   # 位置增益
        kv = 6.0    # 速度增益
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        return torques
    
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading
    
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(np.pi / 2)
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        # print(all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end])
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
       
    
    def _update_heading_arrows(self, data: mtx.SceneData, commands: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        
        robot_pose = self._body.get_pose(data)
        robot_pos = robot_pose[:, :3]
        root_quat = robot_pose[:, 3:7]

        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        base_lin_vel_xy = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        robot_yaw = self._quat_to_yaw(root_quat)

        command_body = np.zeros((num_envs, 3), dtype=np.float32)
        command_body[:, :2] = commands[:, :2]
        command_world = Quaternion.rotate_vector(root_quat, command_body)

        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = robot_yaw[env_idx]
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头

            des_v = command_world[env_idx, :2]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = robot_yaw[env_idx]
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    @staticmethod
    def _quat_to_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
        x = quat_xyzw[:, 0]
        y = quat_xyzw[:, 1]
        z = quat_xyzw[:, 2]
        w = quat_xyzw[:, 3]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)
    
    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    

    def _update_waypoint(self, data, info):
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        base_xy = pose[:, :2]
        num_envs = data.shape[0]

        target_idx = info["target_idx"]
        commands_smoothed = info["commands_smoothed"]
        is_finish = info["is_finish"]
        targets = info["waypoints"][np.arange(num_envs), target_idx]


        delta_world_xy = targets - base_xy
        dist = np.linalg.norm(delta_world_xy, axis=1)
        info["reached_waypoint"].fill(False)

        reached = np.logical_and(~is_finish, dist <= self.position_threshold)
        # print(reached, '------', dist)
        if np.any(reached):
            info["reached_waypoint"][reached] = True
            last_idx = info["num_waypoints"] - 1
            finished = np.logical_and(reached, target_idx >= last_idx)
            advance = np.logical_and(reached, ~finished)
            target_idx[advance] += 1
            is_finish[finished] = True
            targets = info["waypoints"][np.arange(num_envs), target_idx]
            delta_world_xy = targets - base_xy
            dist = np.linalg.norm(delta_world_xy, axis=1)



        command_raw = np.zeros((num_envs, 3), dtype=np.float32)
        active = np.logical_not(is_finish)
        if np.any(active):
            delta_world = np.zeros((num_envs, 3), dtype=np.float32)
            delta_world[:, :2] = delta_world_xy
            delta_body = Quaternion.rotate_inverse(root_quat, delta_world)[:, :2]
            current_yaw = self._quat_to_yaw(root_quat)
            desired_yaw = np.arctan2(delta_world_xy[:, 1], delta_world_xy[:, 0]).astype(np.float32)
            yaw_error = self._wrap_to_pi(desired_yaw - current_yaw)
            command_raw[:, 0] = self._waypoint_lin_kp * delta_body[:, 0]
            command_raw[:, 1] = self._waypoint_lin_kp * delta_body[:, 1]
            command_raw[:, 2] = self._waypoint_ang_kp * yaw_error
            low, high = self._command_limits(base_xy)
            command_raw = np.clip(command_raw, low, high).astype(np.float32)
            command_raw[~active] = 0.0

        commands_smoothed += self._command_smooth_alpha * (command_raw - commands_smoothed)
        low, high = self._command_limits(base_xy)
        commands_smoothed = np.clip(commands_smoothed, low, high).astype(np.float32)
        commands_smoothed[is_finish] = 0.0


        info["target_idx"] = target_idx
        info["commands_smoothed"] = commands_smoothed
        info["commands"] = commands_smoothed.copy()
        info["is_finish"] = is_finish
        info["target_xy"] = targets.astype(np.float32)
        info["distance_to_waypoint"] = dist.astype(np.float32)




    def update_state(self, state: NpEnvState) -> NpEnvState:
        """
        更新环境状态，计算观测、奖励和终止条件
        """
        data = state.data
        info = state.info
        
        
        # Update history-dependent sensor state once per physics step.
        foot_contact = self._get_foot_check(data)
        air_time_before_contact = info["feet_air_time"].copy()
        info["first_contact"] = np.logical_and(air_time_before_contact > 0.0, foot_contact)
        info["air_time_before_contact"] = air_time_before_contact
        info["feet_air_time"] = np.where(foot_contact, 0.0, air_time_before_contact + self._cfg.ctrl_dt)
        info["foot_contact"] = foot_contact
           
        self._update_waypoint(data, info)
        # Evaluate the transition using the waypoint that produced the action.


        # The returned observation must describe the target for the next action.
        self._update_target_marker(data, info["waypoints"][np.arange(data.shape[0]), info["target_idx"]])

        self._update_heading_arrows(
            data,
            info["commands"],
        )
        
        # 计算终止条件
        physical_failure = self._compute_terminated(data)
        pose = self._body.get_pose(data)
        root_xy = pose[:, :2]
        stuck = self._update_stuck_state(
            info,
            root_xy,
        )
        failure = physical_failure | stuck

        # 更新奖励、观测
        reward_dict = self._compute_reward(
            data, info, failure
        )
        obs = self._get_obs(data, info)
        done = failure | info["is_finish"]


        reward_dict_scaled = {k: self.cfg.reward_config.scales.get(k, 0.0) * v for k, v in reward_dict.items()}
        reward = sum(reward_dict_scaled.values())
        reward = np.clip(reward, -100.0, 10000.0)
        info["reward_dict_scaled"] = reward_dict_scaled
        state.obs = obs
        state.reward = reward
        state.terminated = done
        
        return state

    def _get_terrain_scan(self, base_xy: np.ndarray) -> np.ndarray:
        """Compact section01 terrain preview, analogous to IsaacLab height_scan.

        The yellow rough section is a known route band in this scene. These
        forward samples tell the policy when a raised/uneven entrance is coming
        before the feet collide with it, which is the missing signal that tends
        to produce tiny reactive steps.
        """
        sample_y = base_xy[:, 1:2] + self.TERRAIN_SCAN_OFFSETS.reshape(1, -1)
        y_min, y_max = self.ROUGH_Y_RANGE
        inside = np.logical_and(sample_y >= y_min, sample_y <= y_max)
        entry_edge = np.clip(1.0 - np.abs(sample_y - y_min) / 0.6, 0.0, 1.0)
        exit_edge = np.clip(1.0 - np.abs(sample_y - y_max) / 0.6, 0.0, 1.0)
        edge_signal = 0.5 * np.maximum(entry_edge, exit_edge)
        scan = np.where(inside, 1.0, edge_signal)
        return scan.astype(np.float32)


    def _get_obs(self, data: mtx.SceneData, info: dict):
        # 获取基础状态
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        base_lin_vel_world = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        local_lin_vel = Quaternion.rotate_inverse(root_quat, base_lin_vel_world)
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        local_gravity = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles
        

        # 足端触地状态
        foot_contact = self._get_foot_check(data)

        terrain_scan = self._get_terrain_scan(pose[:, :2])
        obs_base = np.hstack(
            [
                local_lin_vel * self._cfg.normalization.lin_vel,
                gyro * self._cfg.normalization.ang_vel,
                local_gravity,
                joint_pos_rel * self._cfg.normalization.dof_pos,
                joint_vel * self._cfg.normalization.dof_vel,
                info["current_actions"],
                info["commands"] * self.commands_scale,
            ]
        )
        assert obs_base.shape == (data.shape[0], 48)
        terrain_scan = self._get_terrain_scan(pose[:, :2])
        obs = np.concatenate([obs_base, foot_contact, terrain_scan], axis=1)

        assert obs.shape == (data.shape[0], 60)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    

    def _update_stuck_state(
        self,
        info: dict,
        root_xy: np.ndarray,
    ) -> np.ndarray:
        
        num_envs = root_xy.shape[0]  
        target_idx = info["target_idx"]     
        wait_steps = info["waypoint_wait_steps_left"]
        is_finish = info["is_finish"]
        
        targets = info["waypoints"][np.arange(num_envs), target_idx]
        delta_world_xy = targets - root_xy
        dist = np.linalg.norm(delta_world_xy, axis=1)
        reached = np.logical_and(~is_finish, np.logical_and(wait_steps <= 0, dist <= self.position_threshold))

        displacement = np.linalg.norm(root_xy - info["stuck_reference_pos"], axis=1)
        should_move = (
            np.linalg.norm(info["commands"][:, :2], axis=1)
            > self.cfg.stuck_command_threshold
        ) & ~reached
        


        made_progress = displacement >= self.cfg.stuck_min_displacement
        reset_window = made_progress | ~should_move
        info["stuck_reference_pos"][reset_window] = root_xy[reset_window]
        info["stuck_steps"][reset_window] = 0
        info["stuck_steps"][should_move & ~made_progress] += 1

        timeout_steps = max(
            1, int(round(self.cfg.stuck_timeout_seconds / self.cfg.ctrl_dt))
        )
        return info["stuck_steps"] >= timeout_steps
    
    
    def _init_termination_contact(self):
        base_geom_names = list(getattr(self._cfg.asset, "terminate_after_contacts_on", []))
        ground_prefixes = ["C1_", "C2_", "C3_"]
        ground_geoms = []
        for geom_name in self._model.geom_names:
            if geom_name is not None and any(prefix in geom_name for prefix in ground_prefixes):
                ground_geoms.append(self._model.get_geom_index(geom_name))
        ground_geoms = list(dict.fromkeys(ground_geoms))

        pairs = []
        for base_geom_name in base_geom_names:
            try:
                base_idx = self._model.get_geom_index(base_geom_name)
            except BaseException:
                continue
            for ground_idx in ground_geoms:
                pairs.append([base_idx, ground_idx])
        self.termination_contact = np.asarray(pairs, dtype=np.uint32) if pairs else np.zeros((0, 2), dtype=np.uint32)
        self.num_termination_check = int(self.termination_contact.shape[0])
        if self.num_termination_check == 0:
            print("[Warning] 未找到任何终止接触geom，基座接触检测将回退到contact sensors。")


    def _get_base_contacts(self, data: mtx.SceneData) -> np.ndarray:
        if self.num_termination_check > 0:
            cquerys = self._model.get_contact_query(data)
            contact = cquerys.is_colliding(self.termination_contact)
            contact = contact.reshape((data.shape[0], self.num_termination_check)) 
            return contact.any(axis=1)
        return self._get_base_contact(data)

    def _get_base_contact(self, data: mtx.SceneData) -> NpEnvState:
        """
        重写终止条件，与locomotion stairs完全一致
        """
        # 基座接触地面终止（使用传感器）
        try:
            base_contact_value = self._model.get_sensor_value(self.cfg.asset.ground_subtree.format(self.section_idx), data)  # num_envs, 3  x y z
            base_contact = (np.linalg.norm(base_contact_value, axis=-1) > 0.01)  # z轴判断倒地  x y轴判断碰撞   shape: (num_envs, )
        except Exception as e:
            print(f"[Warning] 无法读取base_contact传感器: {e}")
            base_contact = np.zeros(self._num_envs, dtype=bool)
        
        return base_contact
    
    def _compute_terminated(self, data: mtx.SceneData) -> NpEnvState:
        dof_vel = self._body.get_joint_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)  # 仿真崩掉

        # 机身发生碰撞、倒地
        base_contact = self._get_base_contacts(data)

        # 
        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = Quaternion.rotate_inverse(root_quat, self.gravity_vec)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip = tilt_angle > np.deg2rad(50)  # 机器人过度倾斜

        # undesired_contacts = self._get_undesired_contacts(data)

        terminated = (
            vel_extreme
            | base_contact
            | side_flip
            # | (undesired_contacts > 0.5)
            ) 
        return terminated


    def _get_foot_check(self, data) -> NpEnvState:
        """
        """
        try:
            foot_contact = []
            for foot_name in self.cfg.asset.foot_names:
                
                self._model.num_sensors
                foot_contact_value = self._model.get_sensor_value(foot_name.format(self.section_idx), data)  # num_envs, 3: x, y, z
                single_foot_contact = (np.linalg.norm(foot_contact_value, axis=-1) > 0.01)
                foot_contact.append(single_foot_contact)
            foot_contact = np.vstack(foot_contact).T  # nums_envs, 4


        except Exception as e:
            print(f"[Warning] 无法读取foot_contact传感器: {e}")
            foot_contact = np.zeros((self._num_envs, 4), dtype=bool)
        
        return foot_contact
    
    def _pit_zone_mask(self, y: np.ndarray) -> np.ndarray:
        return np.logical_and(y >= self.PIT_Y_RANGE[0], y <= self.PIT_Y_RANGE[1])
    
     
    def _compute_reward(self, data: mtx.SceneData, info: dict, termination) -> np.ndarray:
        """
        导航任务奖励计算
        """
        cfg = self._cfg
        num_envs = data.shape[0]
        target_idx = info["target_idx"]
        commands = info["commands"]
        termination_penalty = np.where(
            termination,
            1.0,
            0.0,
        )

        # 获取基础状态
        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles


        # 传感器数据
        base_lin_vel = root_vel[:, :3]  # 世界坐标系线速度
        local_base_lin_vel_body = Quaternion.rotate_inverse(root_quat, base_lin_vel)
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        body_speed_xy = np.linalg.norm(local_base_lin_vel_body[:, :2], axis=1)
        
        
        # 导航目标
        pose_commands = info["waypoints"][np.arange(num_envs), target_idx]
        robot_position = root_pos[:, :2]
        target_position = pose_commands[:, :2]
        
        # 计算位置误差
        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)
        if "min_distance" not in info:
            info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # Reward 5 points for every 1 meter closer


        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])

        # 5. XY-axis angular velocity penalty
        # XY轴角速度惩罚，鼓励机器人保持稳定的姿态，避免翻滚或旋转
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)


        # 线速度跟踪奖励：鼓励机器人按照期望的线速度运动，减少误差
        lin_vel_error = np.sum(np.square(commands[:, :2] - local_base_lin_vel_body[:, :2]), axis=1)  # 机器人线速度误差
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)  # tracking_sigma = 0.25
        ang_vel_error = np.square(commands[:, 2] - gyro[:, 2])
        tracking_ang_vel_reward = np.exp(-ang_vel_error / 0.25)

        # 参考速度惩罚，未到目标点且低于命令速度，受到此惩罚
        command_speed_xy = np.linalg.norm(commands[:, :2], axis=1)
        speed_deficit = np.clip(command_speed_xy - body_speed_xy, 0.0, None)
        active_move = (command_speed_xy > 0.05).astype(np.float32)


        target_rel_world = np.zeros((data.shape[0], 3), dtype=np.float32)
        target_rel_world[:, :2] = target_position - root_pos[:, :2]
        target_rel_body = Quaternion.rotate_inverse(root_quat, target_rel_world)[:, :2]
        target_dir_body = target_rel_body / (np.linalg.norm(target_rel_body, axis=1, keepdims=True) + 1e-6)
        tracking_goal_vel = np.minimum(np.sum(target_dir_body * local_base_lin_vel_body[:, :2], axis=1), commands[:, 0])
        tracking_goal_vel = tracking_goal_vel / (np.abs(commands[:, 0]) + 1e-5)
        tracking_goal_vel = np.clip(tracking_goal_vel, -1.0, 1.0) * active_move
        
        
        # 力矩惩罚：节能、动作平滑、保护硬件
        torque_penalty = np.sum(np.square(data.actuator_ctrls), axis=1)


        # 动作变化惩罚：鼓励动作平滑，避免过度抖动
        action_diff = info["current_actions"] - info["last_actions"]
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # 步态奖励
        feet_air_time_reward = np.sum(
            (info["air_time_before_contact"] - 0.5) * info["first_contact"],
            axis=1,
        )

        # 坑洼抬脚：坑洼段内鼓励 2~3 脚触地（=1~2 脚抬起跨越坑洼），
        # 抑制全触地滑行(4)与全腾空跳跃(0)；仅在坑洼区且移动时生效。
        n_contact = np.sum(info["foot_contact"].astype(np.int32), axis=1)
        in_pit = self._pit_zone_mask(root_pos[:, 1]).astype(np.float32)
        swing_ok = np.logical_and(n_contact >= 2, n_contact <= 3).astype(np.float32)

        arrival_bonus = np.where(info["reached_waypoint"], 1.0, 0.0)

        # 到达最终奖励更大的
        is_finish = info["is_finish"]
        arrival_final_bonus = np.where(is_finish, 1.0, 0.0)

        
        reward_dict = {
            "is_finish": arrival_final_bonus,  # 最终到达奖励
            "arrival_bonus": arrival_bonus,    # 到达各waypoint奖励

            "ang_vel_xy_penalty": ang_vel_xy_penalty,  # xy角速度惩罚，鼓g机器人保持稳定姿态
            "distance_to_target": approach_reward,  # 接近目标奖励
            "tracking_lin_vel_reward": tracking_lin_vel,   # 线速度跟踪奖励
            "tracking_ang_vel_reward": tracking_ang_vel_reward,  # 角速度跟踪奖励
            "tracking_goal_vel": tracking_goal_vel,
            "feet_air_time_reward": feet_air_time_reward,  # 步态奖励

            "torque_penalty": torque_penalty,    # 力矩惩罚
            "action_rate_penalty": action_rate_penalty,  # 动作剧烈变化惩罚
            "termination_penalty": termination_penalty,  # 终止惩罚
            "anti_stall": speed_deficit * active_move,  # 速度跟踪惩罚
            "pit_swing": swing_ok * in_pit * active_move  # S1路段坑洼地段抬脚奖励
        }
        # 计算总奖励
        return reward_dict
    

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg: VBotSection01EnvCfg = self._cfg
        num_envs = data.shape[0]
        
        # 在高台中央小范围内随机生成位置
        # X, Y: 在spawn_center周围 ±spawn_range 范围内随机
        random_xy = np.random.uniform(
            low=-self.spawn_range,
            high=self.spawn_range,
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_xy  # [num_envs, 2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)  # 使用配置的高度
        
        
        # 组合XYZ坐标
        robot_init_pos = robot_init_xy  # [num_envs, 2]
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])  # [num_envs, 3]
        
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))
        
        # 设置 base 的 XYZ位置（DOF 3-5）
        dof_pos[:, 3:6] = robot_init_xyz  # [x, y, z] 随机生成的位置


        # 按照waypoint给指令
        target_positions1 = np.tile(self.target_positions1, (num_envs, 1, 1))
        target_idx = np.zeros(num_envs, dtype=np.int32)  # (num_envs, )
        # pose_commands = 
        

        dof_pos[:, self._base_quat_start:self._base_quat_end] = np.array([0, 0, 0.7071, 0.7071], dtype=np.float32)
        dof_pos[:, self._robot_arrow_dof_start+3:self._robot_arrow_dof_end] = np.array([0, 0, 0.7071, 0.7071], dtype=np.float32)
        dof_pos[:, self._desired_arrow_dof_start+3:self._desired_arrow_dof_end] = np.array([0, 0, 0.7071, 0.7071], dtype=np.float32) 
        
        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        joint_noise = np.random.uniform(
            -self._joint_noise_scale, self._joint_noise_scale, size=(num_envs, self._num_action)
        ).astype(np.float32)
        joint_dof_pos = self.default_angles + joint_noise
        self._body.set_dof_pos(data, joint_dof_pos, include_floatingbase=False)
        self._body.set_dof_vel(data, np.zeros((num_envs, self._num_action), dtype=np.float32), include_floatingbase=False)
        self._model.forward_kinematic(data)
        
        # 更新目标位置标记
        self._update_target_marker(data, target_positions1[np.arange(num_envs), target_idx])
        reset_root_xy = self._body.get_pose(data)[:, :2].copy()
        
        info = {
            "commands": np.zeros((num_envs, 3), dtype=np.float32),
            "commands_smoothed": np.zeros((num_envs, 3), dtype=np.float32),
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            # 新增：与locomotion一致的字段
            "last_dof_vel": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "air_time_before_contact": np.zeros((num_envs, 4), dtype=np.float32),
            "feet_air_time": np.zeros(
                (num_envs, self.num_foot_check),
                dtype=np.float32,
            ),
            "num_waypoints": self.target_positions1.shape[0],
            "waypoints": target_positions1,
            "target_idx": target_idx,
            "is_finish": np.zeros(num_envs, dtype=np.bool_),
            "reached_waypoint": np.zeros(num_envs, dtype=np.bool_),
            "first_contact": np.zeros(
                (num_envs, self.num_foot_check), dtype=np.bool_
            ),
            "min_distance": np.full(
                num_envs,
                np.inf,
                dtype=np.float32,
            ),
            "stuck_reference_pos": reset_root_xy,
            "stuck_steps": np.zeros(num_envs, dtype=np.int32),
            "waypoint_wait_steps_left": np.zeros((num_envs,), dtype=np.int32),
            
            "reward_dict": {},
            
        }

        # Seed contact history from the reset pose so standing feet are not
        # reported as fresh landings on the first transition.
        info["foot_contact"] = self._get_foot_check(data)
        obs = self._get_obs(data, info)
        return obs, info
    
