
import time
import mujoco
import mujoco.viewer
import threading
from threading import Thread
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# 分离线性化和平稳仿真的初始角度
init_theta_lin = 0.0  # 用于 AB 估计的初始角度（rad）
init_theta_sim = 0.05  # 用于仿真测试的初始角度（rad）

# 创建线程锁，防止并发访问 mj_data
locker = threading.Lock()

# 数据记录列表
time_list = []
cart_pos_list = []
pole_angle_list = []
cart_vel_list = []
pole_angle_vel_list = []
force_list = []

# 倒立摆 MJCF 模型
mjcf = """
<mujoco model="inverted_pendulum">
    <compiler angle="radian" coordinate="local"/>
    <option timestep="0.01" gravity="0 0 -9.81"/>

    <default>
        <joint limited="false" damping="0"/>
        <geom type="box" rgba="0.8 0.6 0.4 1"/>
    </default>

    <worldbody>
        <geom name="ground" type="plane" pos="0 0 0" size="10 10 0.1" rgba="0.2 0.2 0.2 1"/>
        <body name="cart" pos="0 0 0.1">
            <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.4 0.4"/>
            <joint name="slider" type="slide" axis="1 0 0"/>
            <geom size="0.2 0.1 0.1"/>

            <body name="pole" pos="0 0 0.1">
                <inertial mass="0.2" pos="0 0 0.4" diaginertia="0.01 0.01 0.0001"/>
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom size="0.02 0.02 0.4" pos="0 0 0.4"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="slider" ctrlrange="-1000 1000"/>
    </actuator>
</mujoco>
"""

# 保存 MJCF 为临时文件
with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
    f.write(mjcf)
    xml_path = f.name

# 加载模型和数据
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# 设置仿真时间步长
mj_model.opt.timestep = 0.01

# —— 用于仿真测试的初始角度 —— 
mj_data.qpos[1] = init_theta_sim
mujoco.mj_forward(mj_model, mj_data)

# 启动被动式 viewer
viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

def get_state(mj_data):
    return np.array([
        mj_data.qpos[0],
        mj_data.qvel[0],
        mj_data.qpos[1],
        mj_data.qvel[1]
    ])

def set_state(mj_data, x):
    mj_data.qpos[0] = x[0]
    mj_data.qvel[0] = x[1]
    mj_data.qpos[1] = x[2]
    mj_data.qvel[1] = x[3]
    mujoco.mj_forward(mj_model, mj_data)

def estimate_AB(mj_model, mj_data, x0, u0, dt=1e-4):
    n = len(x0)
    m = len(u0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    def get_dx(x, u):
        set_state(mj_data, x)
        mj_data.ctrl[:] = u
        mujoco.mj_forward(mj_model, mj_data)
        x_before = get_state(mj_data)
        mujoco.mj_step(mj_model, mj_data)
        x_after = get_state(mj_data)
        return (x_after - x_before) / mj_model.opt.timestep

    # 估计 A 矩阵
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = dt
        A[:, i] = (get_dx(x0 + dx, u0) - get_dx(x0 - dx, u0)) / (2 * dt)

    # 估计 B 矩阵
    for i in range(m):
        du = np.zeros(m)
        du[i] = dt
        B[:, i] = (get_dx(x0, u0 + du) - get_dx(x0, u0 - du)) / (2 * dt)

    return A, B

# —— 用于 AB 估计的线性化平衡点 —— 
x0 = np.array([0.0, 0.0, init_theta_lin, 0.0])
u0 = np.array([0.0])

# 估计 A, B
A, B = estimate_AB(mj_model, mj_data, x0, u0)
print("Estimated A:\\n", A)
print("Estimated B:\\n", B)

# LQR 设计
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P

# 设置 LQR 权重并计算增益
Q = np.diag([1, 1, 10, 10])
R = np.array([[0.001]])
K = lqr(A, B, Q, R)
print("LQR Gain K:", K)

mj_data.qpos[1] = init_theta_sim
mujoco.mj_forward(mj_model, mj_data)

# 仿真控制线程
def SimulationThread():
    while viewer.is_running():
        t0 = time.perf_counter()
        locker.acquire()

        x = get_state(mj_data)
        u = -K @ x
        mj_data.ctrl[0] = float(u)

        # 记录数据
        time_list.append(mj_data.time)
        cart_pos_list.append(x[0])
        cart_vel_list.append(x[1])
        pole_angle_list.append(x[2])
        pole_angle_vel_list.append(x[3])
        force_list.append(float(u))

        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        dt = mj_model.opt.timestep - (time.perf_counter() - t0)
        if dt > 0:
            time.sleep(dt)

# Viewer 同步线程
def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(0.02)

if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread    = Thread(target=SimulationThread)
    viewer_thread.start()
    sim_thread.start()
    viewer_thread.join()
    sim_thread.join()

    # 垂直拼接绘图
    fig, axs = plt.subplots(5, 1, figsize=(10,12), sharex=True)
    axs[0].plot(time_list, cart_pos_list);       axs[0].set_ylabel("Cart Pos (m)");      axs[0].grid(True)
    axs[1].plot(time_list, cart_vel_list);       axs[1].set_ylabel("Cart Vel (m/s)");   axs[1].grid(True)
    axs[2].plot(time_list, pole_angle_list);     axs[2].set_ylabel("Pole Angle (rad)"); axs[2].grid(True)
    axs[3].plot(time_list, pole_angle_vel_list); axs[3].set_ylabel("Pole Vel (rad/s)"); axs[3].grid(True)
    axs[4].plot(time_list, force_list);          axs[4].set_ylabel("Force (N)");        axs[4].set_xlabel("Time (s)"); axs[4].grid(True)
    plt.suptitle("Inverted Pendulum LQR Control (init_sim=%.3f, init_lin=%.3f)" % (init_theta_sim, init_theta_lin))
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()