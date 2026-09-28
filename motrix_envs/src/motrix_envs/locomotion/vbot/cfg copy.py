# # Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

# import os
# from dataclasses import dataclass, field

# from motrix_envs import registry
# from motrix_envs.base import EnvCfg

# model_file = os.path.dirname(__file__) + "/xmls/scene.xml"

# @dataclass
# class NoiseConfig:
#     level: float = 1.0
#     scale_joint_angle: float = 0.03
#     scale_joint_vel: float = 1.5
#     scale_gyro: float = 0.2
#     scale_gravity: float = 0.05
#     scale_linvel: float = 0.1

# @dataclass
# class ControlConfig:
#     # stiffness[N*m/rad] 使用XML中kp参数，仅作记录
#     # damping[N*m*s/rad] 使用XML中kv参数，仅作记录
#     action_scale = 0.25  # 平地navigation使用0.25
#     # torque_limit[N*m] 使用XML forcerange参数

# @dataclass
# class InitState:
#     # the initial position of the robot in the world frame
#     pos = [0.0, 0.0, 0.5]  
    
#     # 位置随机化范围 [x_min, y_min, x_max, y_max]
#     pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # 在ground上随机分散20m x 20m范围

#     # the default angles for all joints. key = joint name, value = target angle [rad]
#     # 使用locomotion的关节角度配置
#     default_joint_angles = {
#         "FR_hip_joint": -0.0,     # 右前髋关节
#         "FR_thigh_joint": 0.9,    # 右前大腿
#         "FR_calf_joint": -1.8,    # 右前小腿
#         "FL_hip_joint": 0.0,      # 左前髋关节
#         "FL_thigh_joint": 0.9,    # 左前大腿
#         "FL_calf_joint": -1.8,    # 左前小腿
#         "RR_hip_joint": -0.0,     # 右后髋关节
#         "RR_thigh_joint": 0.9,    # 右后大腿
#         "RR_calf_joint": -1.8,    # 右后小腿
#         "RL_hip_joint": 0.0,      # 左后髋关节
#         "RL_thigh_joint": 0.9,    # 左后大腿
#         "RL_calf_joint": -1.8,    # 左后小腿
#     }

# @dataclass
# class Commands:
#     # 目标位置相对于机器人初始位置的偏移范围 [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
#     # dx/dy: 相对机器人初始位置的偏移（米）
#     # yaw: 目标绝对朝向（弧度），水平方向随机
#     pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]

# @dataclass
# class Normalization:
#     lin_vel = 2.0
#     ang_vel = 0.25
#     dof_pos = 1.0
#     dof_vel = 0.05

# @dataclass
# class Asset:
#     body_name = "base"
#     foot_names = ["FR", "FL", "RR", "RL"]
#     terminate_after_contacts_on = ["collision_middle_box", "collision_head_box"]
#     ground_subtree = "C_"  # 地形根节点，用于subtree接触检测
   
# @dataclass
# class Sensor:
#     base_linvel = "base_linvel"
#     base_gyro = "base_gyro"
#     feet = ["FR", "FL", "RR", "RL"]  # 足部接触力传感器名称

# @dataclass
# class RewardConfig:
#     scales: dict[str, float] = field(
#         default_factory=lambda: {
#             # ===== 导航任务核心奖励 =====
#             "position_tracking": 2.0,      # 位置误差奖励（提高10倍）
#             "fine_position_tracking": 2.0,  # 精细位置奖励（提高10倍）
#             "heading_tracking": 1.0,        # 朝向跟踪奖励（新增）
#             "forward_velocity": 0.5,        # 前进速度奖励（鼓励朝目标移动）
            
#             # ===== Locomotion稳定性奖励（保持但降低权重） =====
#             "orientation": -0.05,           # 姿态稳定（降低权重）
#             "lin_vel_z": -0.5,              # 垂直速度惩罚
#             "ang_vel_xy": -0.05,            # XY轴角速度惩罚
#             "torques": -1e-5,               # 扭矩惩罚
#             "dof_vel": -5e-5,               # 关节速度惩罚
#             "dof_acc": -2.5e-7,             # 关节加速度惩罚
#             "action_rate": -0.01,           # 动作变化率惩罚
            
#             # ===== 终止惩罚 =====
#             "termination": -200.0,          # 终止惩罚
#         }
#     )

# @registry.envcfg("vbot_navigation_flat")
# @dataclass
# class VBotEnvCfg(EnvCfg):
#     model_file: str = model_file
#     reset_noise_scale: float = 0.01
#     max_episode_seconds: float = 10
#     max_episode_steps: int = 1000
#     sim_dt: float = 0.01    # 仿真步长 10ms = 100Hz
#     ctrl_dt: float = 0.01
#     reset_yaw_scale: float = 0.1
#     max_dof_vel: float = 100.0  # 最大关节速度阈值，训练初期给予更大容忍度

#     noise_config: NoiseConfig = field(default_factory=NoiseConfig)
#     control_config: ControlConfig = field(default_factory=ControlConfig)
#     reward_config: RewardConfig = field(default_factory=RewardConfig)
#     init_state: InitState = field(default_factory=InitState)
#     commands: Commands = field(default_factory=Commands)
#     normalization: Normalization = field(default_factory=Normalization)
#     asset: Asset = field(default_factory=Asset)
#     sensor: Sensor = field(default_factory=Sensor)


# @registry.envcfg("vbot_navigation_stairs")
# @dataclass
# class VBotStairsEnvCfg(VBotEnvCfg):
#     """VBot在楼梯地形上的导航配置，继承flat配置"""
#     model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs.xml"
#     max_episode_seconds: float = 20.0  # 增加到20秒，给更多时间学习转向
#     max_episode_steps: int = 2000
    
#     @dataclass
#     class ControlConfig:
#         action_scale = 0.25  # 楼梯navigation使用0.2，足够转向但比平地更谨慎
    
#     control_config: ControlConfig = field(default_factory=ControlConfig)


# @registry.envcfg("VBotStairsMultiTarget-v0")
# @dataclass
# class VBotStairsMultiTargetEnvCfg(VBotStairsEnvCfg):
#     """VBot楼梯多目标导航配置，继承单目标配置"""
#     max_episode_seconds: float = 60.0  # 多目标需要更长时间
#     max_episode_steps: int = 6000


# @registry.envcfg("vbot_navigation_stairs_obstacles")
# @dataclass
# class VBotStairsObstaclesEnvCfg(VBotStairsEnvCfg):
#     """VBot楼梯地形带障碍球的导航配置"""
#     model_file: str = os.path.dirname(__file__) + "/xmls/scene_stairs_obstacles.xml"
#     max_episode_seconds: float = 20.0
#     max_episode_steps: int = 2000

# @registry.envcfg("vbot_navigation_long_course")
# @dataclass
# class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
#     """VBot三段地形完整导航配置（比赛任务）- 使用world.xml统一地图"""
#     # 使用scene_world.xml作为完整的三段地形地图（集成了world.xml）
#     model_file: str = os.path.dirname(__file__) + "/xmls/scene_world.xml"
#     max_episode_seconds: float = 60.0  # 优化：减少到60秒，加快训练速度
#     max_episode_steps: int = 6000  # 对应60秒 @ 100Hz
    
#     @dataclass
#     class InitState:
#         # 起始位置：section01的中心位置
#         pos = [0.0, 0.0, 1.8]  # 高台中心，高度1.8m
#         pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # 小范围随机±0.5m
        
#         default_joint_angles = {
#             "FR_hip_joint": -0.0,
#             "FR_thigh_joint": 0.9,
#             "FR_calf_joint": -1.8,
#             "FL_hip_joint": 0.0,
#             "FL_thigh_joint": 0.9,
#             "FL_calf_joint": -1.8,
#             "RR_hip_joint": -0.0,
#             "RR_thigh_joint": 0.9,
#             "RR_calf_joint": -1.8,
#             "RL_hip_joint": 0.0,
#             "RL_thigh_joint": 0.9,
#             "RL_calf_joint": -1.8,
#         }
    
#     @dataclass
#     class Commands:
#         # 目标范围：覆盖整个30米路线（section01:0-10m, section02:10-20m, section03:20-30m）
#         pose_command_range = [-3.0, 20.0, -3.14, 3.0, 32.0, 3.14]
    
#     @dataclass
#     class ControlConfig:
#         action_scale = 0.25  # 与stairs保持一致
    
#     init_state: InitState = field(default_factory=InitState)
#     commands: Commands = field(default_factory=Commands)
#     control_config: ControlConfig = field(default_factory=ControlConfig)



# # ============================================================================
# # Section011 越障导航：三阶段训练 pipeline 配置
# #   s1：速度指令跟踪（密集奖励）+ 简化语义地形 + 关一切课程·起跑线全宽出生 + 6 穿越门稠密奖励 → 从零训练
# #   s2：速度指令跟踪 + 真实地形高度感知 + EMA 地形难度课程 → warmstart 自 s1
# #   s3：端到端（去人工速度指令：目标相对位置观测 + position-progress 奖励）+ 由近及远距离课程 → warmstart 自 s2
# # 三阶段观测维度恒为 68、网络同构 → 逐级 --checkpoint warmstart 兼容。
# # 正式注册名 vbot_navigation_section011 / _eval 即 s3 逻辑。
# # ============================================================================
# @dataclass
# class VBotSection011EnvCfg(VBotStairsEnvCfg):
#     """Section011 越障导航基础配置（默认开关组合 = s1：指令跟踪 + 简化语义地形 + 关一切课程）。"""
#     model_file: str = os.path.dirname(__file__) + "/xmls/scene_section011.xml"
#     max_episode_seconds: float = 40.0
#     max_episode_steps: int = 4000
#     eval_mode: bool = False
#     reach_threshold: float = 0.45
#     # 出生范围：集中在起跑平地中心 (x≈0, y≈-2.4)、统一朝 +y，保证每个 episode 任务一致。
#     spawn_x_range: tuple = (-0.5, 0.5)
#     spawn_y_range: tuple = (-2.9, -2.0)

#     # —— 阶段开关（s1/s2/s3 子类按阶段组合）——
#     use_e2e_goal_obs: bool = False        # True=端到端：观测用目标相对位置替代速度指令（s3）
#     use_real_terrain_scan: bool = False   # True=真实地形高度采样 TerrainSampler（s2/s3）
#     use_terrain_curriculum: bool = False  # True=EMA 地形难度课程：难段入口前出生混合（s2）
#     use_dist_curriculum: bool = False     # True=由近及远距离课程（仅 warmstart 后的 s3 启用）

#     # —— 端到端观测缩放（与速度指令量级相近，warmstart 友好）——
#     goal_pos_obs_clip: float = 3.0
#     goal_pos_obs_scale: float = 0.5
#     goal_yaw_obs_scale: float = 0.5
#     # 端到端模式 anti_stall 的参考前进速度（未到终点却低于此速度受罚）
#     e2e_ref_speed: float = 0.4

#     # —— 地形难度课程参数（s2）——
#     curriculum_p_hard_init: float = 0.15      # 初始难段出生比例
#     curriculum_p_hard_max: float = 0.5        # 难段出生比例上限
#     curriculum_success_target: float = 0.6    # EMA 成功率目标（达标→提高难段比例）
#     curriculum_ema_alpha: float = 0.02        # 成功率 EMA 平滑系数
#     # 反滑行：单脚连续触地超此秒数开始惩罚（防止策略收敛到贴地滑行局部最优）
#     stance_overstay_sec: float = 0.4

#     # —— 由近及远距离课程参数（s3）——
#     # 出生 y 初始落在平台上 [6.8, 7.3]（距终点中心 7.8 大于到达阈值 0.45，不会出生即判到达），
#     # EMA 达标后下界逐步后退：平台 → 坡顶 → 上坡 → 坑洼 → 平地 → 起跑线 -2.9。
#     dist_spawn_y_hi: float = 7.3
#     dist_spawn_y_lo_init: float = 6.8
#     dist_spawn_y_far: float = -2.9
#     dist_curriculum_step: float = 0.4         # 每档后退增量
#     # 推进一档后 EMA×此系数，强制在新出生点重新达标才再推下一档；
#     # 取值过大会让课程推进超过策略学习速度、导致整体退化，宁慢勿崩。
#     dist_curriculum_ema_decay: float = 0.6

#     @dataclass
#     class InitState:
#         pos = [0.0, -3.0, 0.5]
#         default_joint_angles = {
#             "FR_hip_joint": -0.0,
#             "FR_thigh_joint": 0.9,
#             "FR_calf_joint": -1.8,
#             "FL_hip_joint": 0.0,
#             "FL_thigh_joint": 0.9,
#             "FL_calf_joint": -1.8,
#             "RR_hip_joint": -0.0,
#             "RR_thigh_joint": 0.9,
#             "RR_calf_joint": -1.8,
#             "RL_hip_joint": 0.0,
#             "RL_thigh_joint": 0.9,
#             "RL_calf_joint": -1.8,
#         }

#     @dataclass
#     class Commands:
#         waypoint_lin_kp: float = 0.8
#         waypoint_ang_kp: float = 1.0

#     @dataclass
#     class ControlConfig:
#         action_scale: float = 0.45
#         stiffness: float = 80.0
#         damping: float = 6.0

#     @dataclass
#     class NavRewardConfig:
#         tracking_sigma: float = 0.2
#         feet_air_time_target: float = 0.45
#         scales: dict = field(default_factory=dict)

#     init_state: InitState = field(default_factory=InitState)
#     commands: Commands = field(default_factory=Commands)
#     control_config: ControlConfig = field(default_factory=ControlConfig)
#     reward_config: NavRewardConfig = field(default_factory=NavRewardConfig)


# @registry.envcfg("vbot_navigation_section011_s1")
# @dataclass
# class VBotSection011S1EnvCfg(VBotSection011EnvCfg):
#     """[阶段1 · 从零训练] 速度指令跟踪（密集奖励，上层按航路点生成 commands）+ 简化语义地形
#     + 关一切课程·起跑线全宽均匀出生 (x∈[-0.5,0.5], y∈[-2.9,-2.0]) + 6 固定穿越门稠密奖励
#     （每过一门 +8）从起跑线一路牵引到平台。

#     刻意不用任何「刷近出生」课程：从零阶段若让出生点靠近终点，训练成功率会虚高
#     （平整段滑行即可「到达」），而 eval 从起跑线出发的真实成功率为 0、坑洼区得不到训练；
#     训练与评估保持同口径（都从起跑线走全程）才能保证指标真实。"""
#     use_e2e_goal_obs: bool = False        # 速度指令跟踪（密集奖励，从零可训）
#     use_real_terrain_scan: bool = False   # 简化语义地形，先专注「会走 + 全程导航」
#     use_dist_curriculum: bool = False
#     use_terrain_curriculum: bool = False


# @registry.envcfg("vbot_navigation_section011_s1_eval")
# @dataclass
# class VBotSection011S1EvalEnvCfg(VBotSection011S1EnvCfg):
#     """阶段1 评估/回放：固定起跑线全程出生、关一切课程，口径与真实任务一致（简化语义地形）。"""
#     eval_mode: bool = True
#     render_spacing: float = 0.0


# @registry.envcfg("vbot_navigation_section011_s2")
# @dataclass
# class VBotSection011S2EnvCfg(VBotSection011EnvCfg):
#     """[阶段2 · 真实地形感知，warmstart 自 s1] 仍为速度指令跟踪，但 terrain_scan 换成从仿真几何
#     实采的真实高度差（米），策略须自行解读地形。叠加 EMA 地形难度课程：起跑线全宽出生
#     + 按 p_hard 概率混入难段【入口前】出生（仍需穿越难段才算成功，不刷近终点）。"""
#     use_e2e_goal_obs: bool = False        # 仍速度指令跟踪
#     use_real_terrain_scan: bool = True    # 引入真实地形高度感知（本阶段新难度）
#     use_dist_curriculum: bool = False
#     use_terrain_curriculum: bool = True   # EMA 地形难度课程


# @registry.envcfg("vbot_navigation_section011_s2_eval")
# @dataclass
# class VBotSection011S2EvalEnvCfg(VBotSection011S2EnvCfg):
#     """阶段2 评估/回放：固定起跑线全程出生、关一切课程（真实地形感知）。"""
#     eval_mode: bool = True
#     use_terrain_curriculum: bool = False
#     render_spacing: float = 0.0


# @registry.envcfg("vbot_navigation_section011")
# @registry.envcfg("vbot_navigation_section011_s3")
# @dataclass
# class VBotSection011S3EnvCfg(VBotSection011EnvCfg):
#     """[阶段3 · 端到端，warmstart 自 s2，正式交付形态] 去掉人工速度指令：观测改目标相对位置
#     goal_delta_body + 朝向误差 yaw_error，奖励改 position-progress 驱动，由策略自主决定速度。
#     配「由近及远」距离课程：warmstart 后策略已会走，课程帮稀疏的无指令导航拿到第一次成功，
#     并随 EMA 达标把出生点逐步拉回起跑线；评估始终用固定起跑线全程口径。"""
#     use_e2e_goal_obs: bool = True
#     use_real_terrain_scan: bool = True
#     use_dist_curriculum: bool = True
#     use_terrain_curriculum: bool = False
#     # 距离课程的 EMA 达标阈值取 0.7（比地形课程更谨慎，达标才拉远出生下界）
#     curriculum_success_target: float = 0.7


# @registry.envcfg("vbot_navigation_section011_eval")
# @registry.envcfg("vbot_navigation_section011_s3_eval")
# @dataclass
# class VBotSection011S3EvalEnvCfg(VBotSection011S3EnvCfg):
#     """阶段3 评估/回放：固定起跑线全程出生、关一切课程，口径同真实任务（无速度指令自主导航）。"""
#     eval_mode: bool = True
#     use_dist_curriculum: bool = False
#     render_spacing: float = 0.0

