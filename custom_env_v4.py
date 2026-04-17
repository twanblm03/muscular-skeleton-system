import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium import spaces
from gymnasium.spaces import MultiBinary
# DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 1.5,
    # "lookat": np.array((0.0, 0.0, 1.15)),
    # "elevation": -20.0,
}

class RobExampleEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 250,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "system_model.xml")
        
        # 读取XML文件中actuator标签下的muscle数量
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        actuator = root.find(".//actuator")
        if actuator is not None:
            self.muscle_count = len(actuator.findall("muscle"))
        else:
            raise ValueError("No actuator tag found in XML file")

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            # shape=(15 + self.muscle_count + self.muscle_count,),  # 4(qpos) + 3(qvel) + muscle_count(muscle_length) + 3(forearm_tip_pos) + 3(vec_to_target)
            shape=(16,),
            # shape=(6,),
            dtype=np.float64
        )


        MujocoEnv.__init__(
            self,
            xml_path,
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # self.action_space = Box(
        #     low=0,
        #     high=1, 
        #     shape=(4,),  # 4个肌肉的激活水平
        #     dtype=np.float64
        # )
        self.action_space = MultiBinary(self.muscle_count)  

        self.prev_dist = float('inf')  # 添加距离历史记录
        self.no_progress_count = 0     # 添加计数器

        self.prev_action = np.zeros(self.muscle_count)  # 添加属性记录上一次动作

        self.stability_threshold = 0.06  # 稳定性阈值
        self.required_stable_steps = 50  # 需要保持稳定的步数

        self.max_dist = 0.36 * 2  # 最大距离

        self.cumulative_changes = 0.0  # 添加累积变化量
        self.episode_steps = 0  # 添加步数计数器


    def step(self, a):
        vec = self.get_body_com("forearm_tip") - self.get_body_com("target")
        curr_dist = np.linalg.norm(vec)

        # 距离奖励：使用指数函数使奖励更平滑
        # sigma = self.max_dist / 6
        # reward_dist = np.exp(-(curr_dist**2) / (2 * sigma**2))
        reward_dist = - (curr_dist / self.max_dist)

        # 动作变化惩罚：鼓励保持稳定的动作
        # action_change = -0.1 * np.sum(np.abs(a - self.prev_action))
        # action_change = -0.1 * np.sqrt(np.sum(np.abs(a - self.prev_action)))
        if np.any(self.prev_action != 0):  # 检查是否为第一步
            immediate_change = (np.sqrt(np.sum(np.abs(a - self.prev_action))) / np.sqrt(self.muscle_count))

            self.cumulative_changes += immediate_change
            self.episode_steps += 1
            ave_his_act_change = - 0.2 * self.cumulative_changes / self.episode_steps

            action_change = - 0.9 * immediate_change
        else:
            action_change = 0.0  # 第一步不施加惩罚

            ave_his_act_change = 0.0
        

        # 动作数量惩罚：鼓励使用更少的肌肉
        action_sparsity = - 0.7 * (np.sum(a) / self.muscle_count)

        # 稳定性奖励：在目标点附近保持稳定
        stability_reward = 0.0
        if curr_dist < self.stability_threshold:
            self.stable_count += 1
            stability_reward = 0.3 * (self.stable_count / self.required_stable_steps)
        else:
            self.stable_count = 0

        # reward = reward_dist + action_change + action_sparsity + progress_reward
        # reward = reward_dist + action_change + stability_reward
        # reward = reward_dist + action_change + action_sparsity + ave_his_act_change + stability_reward
        reward = reward_dist + action_change + action_sparsity
        # reward = reward_dist

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        self.prev_action = a.copy()

        # 放宽完成条件
        # done = curr_dist < 0.02  # 增加容差范围
        done = (curr_dist < self.stability_threshold and 
                self.stable_count >= self.required_stable_steps)

        # 如果长时间没有进展，提前终止
        if abs(curr_dist - self.prev_dist) < 1e-4:
            self.no_progress_count += 1
        else:
            self.no_progress_count = 0
        self.prev_dist = curr_dist

        truncated = self.no_progress_count > self.required_stable_steps  # 增加允许尝试的次数

        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            truncated,
            dict(
                reward_dist=reward_dist,
                action_change=action_change,
                action_sparsity=action_sparsity,
                ave_his_act_change=ave_his_act_change,
                stability_reward=stability_reward
            ),
        )

    def reset_model(self, target_pos=None):
        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=-0.1, high=0.1, size=self.model.nq
        # )
        qpos = self.init_qpos.copy()

        # while True:
        #     self.goal = np.array([
        #         self.np_random.uniform(low=-0.3, high=0.3),  # x
        #         self.np_random.uniform(low=-0.3, high=0.3),  # y
        #         self.np_random.uniform(low=-0.4, high=-0.01) # z
        #     ])
        #     if np.linalg.norm(self.goal[:2]) < 0.3:  # 检查水平面内的距离
        #         break

        if target_pos is not None:
            # Use provided target position
            self.goal = target_pos.copy()
        else:
            # 生成目标点
            while True:
                # 在圆柱体内生成随机点
                radius_xy = self.np_random.uniform(0, 0.3)  # 水平面内的半径
                angle = self.np_random.uniform(0, 2 * np.pi)  # 水平面内的角度
                z = self.np_random.uniform(-0.22, -0.098)  # 限制z坐标范围

                self.goal = np.array([
                    radius_xy * np.cos(angle),  # x
                    radius_xy * np.sin(angle),  # y
                    z                           # z
                ])

                # 确保点在球面上（到球心的距离为0.3）
                center = np.array([0, 0, -0.06])  # 球心
                vec_to_center = self.goal - center
                dist_to_center = np.linalg.norm(vec_to_center)

                if dist_to_center <= 0.3:  # 如果点在球内
                    # 将点投影到球面上
                    self.goal = center + vec_to_center / dist_to_center * 0.3
                    # 检查z坐标是否仍在允许范围内
                    if -0.22 <= self.goal[2] <= -0.098:
                        break

        # center = np.array([0, 0, -0.06])
        # final_dist = np.linalg.norm(self.goal - center)
        # print(f"Final distance to center: {final_dist:.4f}")

        qpos[-3:] = self.goal
        # qvel = self.init_qvel + self.np_random.uniform(
        #     low=-0.1, high=0.1, size=self.model.nv
        # )
        qvel = self.init_qvel.copy()
        qvel[-3:] = 0
        # print("Resetting environment with goal:", self.goal)
        self.set_state(qpos, qvel)

        self.prev_action = np.zeros(self.muscle_count)
        self.prev_dist = float('inf')
        self.no_progress_count = 0
        self.stable_count = 0  # 重置稳定计数器

        self.cumulative_changes = 0.0
        self.episode_steps = 0

        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos.flat[:4]  # 只取球关节的4个自由度
        qvel = self.data.qvel.flat[:3]  # 只取球关节的3个速度
        actuator_length = self.data.actuator_length  
        actuator_activation = self.data.ctrl
        tip_pos = self.get_body_com("forearm_tip")      
        vec_to_target = tip_pos - self.get_body_com("target")
        target_pos = self.get_body_com("target")

        # # Debug prints
        # print(f"\nObservation components:")
        # print(f"qpos (joint angles): {qpos}, shape: {np.array(qpos).shape}")
        # print(f"qvel (joint velocities): {qvel}, shape: {np.array(qvel).shape}")
        # print(f"actuator_length: {actuator_length}, shape: {actuator_length.shape}")
        # print(f"tip_pos: {tip_pos}, shape: {tip_pos.shape}")
        # print(f"vec_to_target: {vec_to_target}, shape: {vec_to_target.shape}")

        obs = np.concatenate([
            qpos,          # 球关节角度 (4)
            qvel,          # 球关节速度 (3)
            # actuator_length,         # 肌肉长度 (n)
            # actuator_activation,      # 肌肉激活水平 (n)
            tip_pos,                 # forearm_tip位置 (3)
            vec_to_target,           # 相对距离 (3)
            target_pos 
        ])
        # print(f"Total observation shape: {obs.shape}\n")
        # print(f"Target position: {self.get_body_com("target")}")
        return obs