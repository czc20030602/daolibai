import pygame
import time
import mujoco
import mujoco.viewer
import threading
from threading import Thread
import tempfile
import matplotlib.pyplot as plt

# 初始化 pygame
pygame.init()
pygame.display.set_mode((100, 100))

locker = threading.Lock()

# 仿真数据记录
time_list = []
cart_pos_list = []
xd_list = []            # 导纳输出目标位置
pole_angle_list = []
pole_angle_vel_list = []
force_list = []
target_angle_list = []
keyboard_force_list = []

# 导纳参数
M_a = 1.0    # 虚拟质量
B_a = 10.0    # 虚拟阻尼
K_a = 0.0    # 虚拟刚度

# 键盘力设置
keyboard_force = 0.0
fixed_force = 10.0

# MuJoCo 模型
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
        <motor joint="slider" ctrlrange="-10000 10000"/>
    </actuator>
</mujoco>
"""

# 写入临时文件并加载
with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
    f.write(mjcf)
    xml_path = f.name

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data  = mujoco.MjData(mj_model)
dt_sim = 0.001
mj_model.opt.timestep = dt_sim
mj_data.qpos[1] = 0.0  # 初始角度

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# PID 参数
# 上层位置 PID

pos_kp = 0.73
pos_kd = 0.45
pos_ki = 0.01

ang_kp = 400.0
ang_kd = 130.0
ang_ki = 0.0

# pos_kp = 0.03
# pos_kd = 0.067
# pos_ki = 0.00001

# # 下层角度 PID
# ang_kp = 300.0
# ang_kd = 0.0
# ang_ki = 0.0

# 键盘线程：更新 keyboard_force
def KeyboardThread():
    global keyboard_force
    while viewer.is_running():
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            keyboard_force = -fixed_force
        elif keys[pygame.K_RIGHT]:
            keyboard_force = fixed_force
        else:
            keyboard_force = 0.0
        time.sleep(0.005)

# 仿真线程
def SimulationThread():
    # 导纳状态
    xd = 0.0
    xd_dot = 0.0
    # PID 积分项
    pos_int = 0.0
    ang_int = 0.0

    start_time = time.time()
    sim_duration = 30.0

    while viewer.is_running():
        # 自动停止
        if time.time() - start_time > sim_duration:
            viewer.close()
            break

        t0 = time.perf_counter()
        locker.acquire()

        # 物理状态
        x        = mj_data.qpos[0]
        theta    = mj_data.qpos[1]
        theta_dot= mj_data.qvel[1]

        # —— 导纳计算，得到动态目标位置 xd —— 
        xdd = (keyboard_force - B_a*xd_dot - K_a*xd) / M_a
        xd_dot += xdd * dt_sim
        xd     += xd_dot * dt_sim

        # 上层位置 PID，目标 = xd
        pos_err = xd - x
        pos_int += pos_err * dt_sim
        target_theta = (pos_kp * pos_err 
                        + pos_kd * (-mj_data.qvel[0]) 
                        + pos_ki * pos_int)

        # 下层角度 PID，目标 = target_theta
        ang_err = theta - target_theta
        ang_int += ang_err * dt_sim
        force = (ang_kp * ang_err 
                 + ang_kd * theta_dot 
                 + ang_ki * ang_int)+ keyboard_force

        mj_data.ctrl[0] = force

        # 记录数据
        time_list.append(mj_data.time)
        cart_pos_list.append(x)
        xd_list.append(xd)
        pole_angle_list.append(theta)
        pole_angle_vel_list.append(theta_dot)
        force_list.append(force)
        target_angle_list.append(target_theta)
        keyboard_force_list.append(keyboard_force)

        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        dt = dt_sim - (time.perf_counter() - t0)
        if dt > 0: time.sleep(dt)

# Viewer 同步
def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(0.02)

if __name__ == "__main__":
    t_k = Thread(target=KeyboardThread)
    t_s = Thread(target=SimulationThread)
    t_v = Thread(target=PhysicsViewerThread)
    t_k.start(); t_s.start(); t_v.start()
    t_k.join(); t_s.join(); t_v.join()
    pygame.quit()

    # 绘图：8 子图
    fig, axs = plt.subplots(8, 1, figsize=(10, 16), sharex=True)

    fontsize_label = 6
    fontsize_tick = 6
    fontsize_title = 10

    axs[0].plot(time_list, cart_pos_list)
    axs[0].set_ylabel("Cart Pos (m)", fontsize=fontsize_label)
    axs[0].tick_params(axis='both', labelsize=fontsize_tick)

    axs[1].plot(time_list, xd_list)
    axs[1].set_ylabel("xd (m)", fontsize=fontsize_label)
    axs[1].tick_params(axis='both', labelsize=fontsize_tick)

    axs[2].plot(time_list, keyboard_force_list)
    axs[2].set_ylabel("F_keyboard (N)", fontsize=fontsize_label)
    axs[2].tick_params(axis='both', labelsize=fontsize_tick)

    axs[3].plot(time_list, pole_angle_list)
    axs[3].set_ylabel("Pole Angle (rad)", fontsize=fontsize_label)
    axs[3].tick_params(axis='both', labelsize=fontsize_tick)

    axs[4].plot(time_list, target_angle_list)
    axs[4].set_ylabel("Theta_target (rad)", fontsize=fontsize_label)
    axs[4].tick_params(axis='both', labelsize=fontsize_tick)

    axs[5].plot(time_list, pole_angle_vel_list)
    axs[5].set_ylabel("Pole Vel (rad/s)", fontsize=fontsize_label)
    axs[5].tick_params(axis='both', labelsize=fontsize_tick)

    axs[6].plot(time_list, force_list)
    axs[6].set_ylabel("Control Force (N)", fontsize=fontsize_label)
    axs[6].tick_params(axis='both', labelsize=fontsize_tick)

    axs[7].plot(time_list, [t - a for t, a in zip(target_angle_list, pole_angle_list)], color="magenta")
    axs[7].set_ylabel("Angle Error (rad)", fontsize=fontsize_label)
    axs[7].set_xlabel("Time (s)", fontsize=fontsize_label)
    axs[7].tick_params(axis='both', labelsize=fontsize_tick)
    axs[7].grid(True)

    plt.suptitle(
        f"Admittance→PositionPID→AnglePID\n"
        f"Admit(M={M_a},B={B_a},K={K_a}), PosPID(Kp,Ki,Kd)=({pos_kp},{pos_ki},{pos_kd}), AnglePID(Kp,Ki,Kd)=({ang_kp},{ang_ki},{ang_kd})",
        fontsize=fontsize_title, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

