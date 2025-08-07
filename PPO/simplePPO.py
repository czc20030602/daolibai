import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import time
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

class InvertedPendulumEnv(gym.Env):
    """
    自定义的倒立摆 Gymnasium 环境，支持训练/评估阶段渲染控制
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        high = np.array([np.inf, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1000.0, 1000.0, shape=(1,), dtype=np.float32)

        self.max_steps = 500
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.qpos[0] = self.np_random.uniform(-0.5, 0.5)
        # self.data.qpos[1] = 0.03
        self.data.qpos[1] = self.np_random.uniform(-0.05, 0.05)
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        force = np.clip(action[0], -1000.0, 1000.0)
        self.data.ctrl[0] = force
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        obs = self._get_obs()
        x, x_dot, theta, theta_dot = obs
        reward = 1.0 - ((theta**2) + 0.1*x**2)
        done = bool(abs(theta) > np.pi/3 or self.current_step >= self.max_steps)
        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode != 'human':
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        time.sleep(0.01)  # 控制渲染速度，大约 20 FPS

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        return np.array([self.data.qpos[0], self.data.qvel[0], self.data.qpos[1], self.data.qvel[1]], dtype=np.float32)

if __name__ == '__main__':
    xml_path = 'inverted_pendulum.xml'

    # ----------------- 训练阶段 -----------------
    # 使用 Monitor + VecMonitor 来记录并打印每个回合的奖励
    def make_env():
        base_env = InvertedPendulumEnv(xml_path)
        return Monitor(base_env, allow_early_resets=True)

    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env)

    model = PPO('MlpPolicy', train_env, verbose=1)
    model.learn(total_timesteps=500_000)
    model.save('ppo_inverted_pendulum')
    train_env.close()

    # ----------------- 评估阶段 -----------------
    num_episodes = 10
    max_steps_per_ep = 500
    eval_env = InvertedPendulumEnv(xml_path, render_mode='human')
    try:
        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            ep_reward = 0.0
            for step in range(max_steps_per_ep):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                ep_reward += reward
                eval_env.render()
                if done:
                    print(f"评估回合 {ep+1} 结束, 累计奖励: {ep_reward:.2f}")
                    break
    finally:
        eval_env.close()
