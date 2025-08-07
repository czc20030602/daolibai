import time
import mujoco
import mujoco.viewer
import threading
from threading import Thread
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# 创建线程锁，防止并发访问 mj_data
locker = threading.Lock()

# 数据记录列表
time_list = []
cart_pos_list = []
cart_vel_list = []
pole_angle_list = []
pole_angle_vel_list = []
force_list = []

# 倒立摆 MJCF 模型（含质量与惯性）
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

# 设置非垂直初始角度
mj_data.qpos[1] = 0.1  # 弧度

# 启动 viewer
viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# LQR 增益矩阵
# K = np.array([-31.6227766 ,  -58.76085072 ,-397.99538636 ,-130.60647878])   # 这里的 K 是通过 mujoco仿真建模LQR 方法计算得到的增益矩阵
# K = np.array([-31.6228 , -59.4429 ,-404.3242 ,-135.2326])  # 这里的 K 是通过 物理分析建模LQR 方法计算得到的增益矩阵
K = np.array([-250/7, -200/7, -60274/525, -2776/105])   # 这里的 K 是通过 物理分析建模+极点配置 方法计算得到的增益矩阵


# 仿真线程：控制倒立摆
def SimulationThread():
    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()

        # 获取状态量
        x = np.array([
            mj_data.qpos[0],  # 小车位置
            mj_data.qvel[0],  # 小车速度
            mj_data.qpos[1],  # 杆角度
            mj_data.qvel[1],  # 杆角速度
        ])

        # 控制输入
        u = -K @ x
        mj_data.ctrl[0] = u

        # 记录数据
        time_list.append(mj_data.time)
        cart_pos_list.append(x[0])
        cart_vel_list.append(x[1])
        pole_angle_list.append(x[2])
        pole_angle_vel_list.append(x[3])
        force_list.append(u)

        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        # 控制频率
        dt = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if dt > 0:
            time.sleep(dt)

# Viewer 同步线程
def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(0.02)

# 主程序
if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    viewer_thread.start()
    sim_thread.start()
    viewer_thread.join()
    sim_thread.join()

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time_list, cart_pos_list, label="Cart Position")
    plt.plot(time_list, cart_vel_list, label="Cart Velocity")
    plt.legend()
    plt.ylabel("Cart State")

    plt.subplot(3, 1, 2)
    plt.plot(time_list, pole_angle_list, label="Pole Angle")
    plt.plot(time_list, pole_angle_vel_list, label="Pole Angular Velocity")
    plt.legend()
    plt.ylabel("Pole State")

    plt.subplot(3, 1, 3)
    plt.plot(time_list, force_list, label="Control Force")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Force")

    plt.suptitle("Inverted Pendulum with LQR Control")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

