import time
import mujoco
import mujoco.viewer
import threading
from threading import Thread
import tempfile
import matplotlib.pyplot as plt

# 创建线程锁
locker = threading.Lock()

# 数据记录列表
time_list = []
cart_pos_list = []
pole_angle_list = []
pole_angle_vel_list = []
force_list = []
target_angle_list = []

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
# 设置初始角度
mj_data.qpos[1] = 0.01

# 启动 viewer
viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# 上层 PID（位置控制 -> 输出目标角度）
pos_kp = 0.005
pos_kd = 0.02
pos_ki = 0.00001
pos_integral_error = 0.0

# 下层 PID（角度控制 -> 输出控制力）
ang_kp = 100.0
ang_kd = 50.0
ang_ki = 50.0
# ang_kp = 34000.0
# ang_kd = 100.0
# ang_ki = 2500.0
ang_integral_error = 0.0

# 仿真线程
def SimulationThread():
    global mj_data, mj_model, pos_integral_error, ang_integral_error
    target_cart_pos = 1.0  # 目标位置

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        # 当前状态
        x = mj_data.qpos[0]
        theta = mj_data.qpos[1]
        theta_dot = mj_data.qvel[1]

        # 上层 PID：位置控制 -> 输出目标角度
        pos_error = target_cart_pos - x
        pos_integral_error += pos_error * mj_model.opt.timestep
        target_theta = pos_kp * pos_error + pos_kd * (-mj_data.qvel[0]) + pos_ki * pos_integral_error

        # 下层 PID：角度控制 -> 输出控制力
        ang_error = theta - target_theta
        ang_integral_error += ang_error * mj_model.opt.timestep
        force = ang_kp * ang_error + ang_kd * theta_dot + ang_ki * ang_integral_error

        mj_data.ctrl[0] = force

        # 记录数据
        time_list.append(mj_data.time)
        cart_pos_list.append(x)
        pole_angle_list.append(theta)
        pole_angle_vel_list.append(theta_dot)
        force_list.append(force)
        target_angle_list.append(target_theta)

        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# viewer 线程
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

    # 绘图（垂直拼接）
    fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(time_list, cart_pos_list, color="blue")
    axs[0].set_ylabel("Cart Pos (m)")
    axs[0].grid(True)

    axs[1].plot(time_list, pole_angle_list, color="green")
    axs[1].set_ylabel("Pole Angle (rad)")
    axs[1].grid(True)

    axs[2].plot(time_list, target_angle_list, color="purple")
    axs[2].set_ylabel("Target Angle (rad)")
    axs[2].grid(True)

    axs[3].plot(time_list, pole_angle_vel_list, color="orange")
    axs[3].set_ylabel("Pole Vel (rad/s)")
    axs[3].grid(True)

    axs[4].plot(time_list, force_list, color="red")
    axs[4].set_ylabel("Force (N)")
    axs[4].set_xlabel("Time (s)")
    axs[4].grid(True)

    plt.suptitle("Inverted Pendulum - Two-Level PID Control")
    plt.tight_layout()
    plt.show()
