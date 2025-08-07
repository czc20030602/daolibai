import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import time
from stable_baselines3 import PPO

class InvertedPendulumEnv(gym.Env):
    """
    只支持渲染的倒立摆环境，用于测试阶段。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, xml_path, render_mode='human'):
        super().__init__()
        # 加载 MuJoCo 模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.render_mode = render_mode

        # 定义空间（与训练时保持一致）
        high = np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1000.0, 1000.0, shape=(1,), dtype=np.float32)

        self.max_steps = 500  # 每回合最大步数
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 重设状态
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.qpos[1] = self.np_random.uniform(-0.05, 0.05)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # 应用动作并仿真
        force = np.clip(action[0], -1000.0, 1000.0)
        self.data.ctrl[0] = force
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        obs = self._get_obs()
        x, x_dot, theta, theta_dot = obs
        reward = 1.0 - (theta**2 + 0.1 * x**2)
        done = bool(abs(theta) > np.pi/2 or self.current_step >= self.max_steps)
        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode != 'human':
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        time.sleep(0.01)  # 控制帧率，约 20 FPS

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        return np.array([
            self.data.qpos[0],
            self.data.qvel[0],
            self.data.qpos[1],
            self.data.qvel[1]
        ], dtype=np.float32)

if __name__ == '__main__':
    # 模型与 MJCF 文件路径
    xml_path = 'inverted_pendulum.xml'
    model_path = 'ppo_inverted_pendulum.zip'

    # 加载训练好的 PPO 模型
    model = PPO.load(model_path)

    # 创建评估环境（只渲染阶段）
    env = InvertedPendulumEnv(xml_path, render_mode='human')

    num_episodes = 20
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            env.render()

        print(f"测试回合 {ep} 完成，累计奖励：{ep_reward:.2f}")

    env.close()
