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

# 创建线程锁
locker = threading.Lock()

# 数据记录列表
time_list = []
cart_pos_list = []
pole_angle_list = []
pole_angle_vel_list = []
force_list = []
target_angle_list = []
target_pos_list = []

# 键盘控制目标位置
target_cart_pos = 1.0
keyboard_step = 0.005  # 每次按键调整目标位置的步长

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
        <motor joint="slider" ctrlrange="-10000 10000"/>
    </actuator>
</mujoco>
"""

with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
    f.write(mjcf)
    xml_path = f.name
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data  = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.0005

mj_data.qpos[1] = 0.00  # 初始角度

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# 串级 PID 参数
# pos_kp = 0.1
# pos_kd = 0.005
# pos_ki = 0.0000
pos_kp = 0.73
pos_kd = 0.45
pos_ki = 0.01

ang_kp = 400.0
ang_kd = 130.0
ang_ki = 0.0


# 记录仿真开始时间
sim_start_time = time.time()
sim_duration = 5.0  # seconds

def KeyboardThread():
    global target_cart_pos
    while viewer.is_running():
        # 自动停止
        if time.time() - sim_start_time > sim_duration:
            viewer.close()
            break
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            target_cart_pos -= keyboard_step
        if keys[pygame.K_RIGHT]:
            target_cart_pos += keyboard_step
        time.sleep(0.01)

def SimulationThread():
    global target_cart_pos
    pos_integral_error = 0.0
    ang_integral_error = 0.0

    while viewer.is_running():
        # 自动停止
        if time.time() - sim_start_time > sim_duration:
            viewer.close()
            break

        t0 = time.perf_counter()
        locker.acquire()

        # 读取状态
        x         = mj_data.qpos[0]
        theta     = mj_data.qpos[1]
        theta_dot = mj_data.qvel[1]

        # 上层 PID：位置 -> 目标摆角
        pos_error = target_cart_pos - x
        pos_integral_error += pos_error * mj_model.opt.timestep
        target_theta = (pos_kp * pos_error
                        + pos_kd * (-mj_data.qvel[0])
                        + pos_ki * pos_integral_error)

        # 下层 PID：角度 -> 控制力
        # target_theta=0.02
        ang_error = theta - target_theta
        ang_integral_error += ang_error * mj_model.opt.timestep
        force = (ang_kp * ang_error
                 + ang_kd * theta_dot
                 + ang_ki * ang_integral_error)

        mj_data.ctrl[0] = force

        # 记录数据
        time_list.append(mj_data.time)
        cart_pos_list.append(x)
        target_pos_list.append(target_cart_pos)
        pole_angle_list.append(theta)
        pole_angle_vel_list.append(theta_dot)
        force_list.append(force)
        target_angle_list.append(target_theta)

        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        dt = mj_model.opt.timestep - (time.perf_counter() - t0)
        if dt > 0:
            time.sleep(dt)

def PhysicsViewerThread():
    while viewer.is_running():
        # 自动停止
        if time.time() - sim_start_time > sim_duration:
            viewer.close()
            break
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(0.02)

if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread    = Thread(target=SimulationThread)
    key_thread    = Thread(target=KeyboardThread)
    viewer_thread.start()
    sim_thread.start()
    key_thread.start()
    viewer_thread.join()
    sim_thread.join()
    key_thread.join()
    pygame.quit()

    # 绘图（7 子图）
    fig, axs = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    axs[0].plot(time_list, cart_pos_list);       axs[0].set_ylabel("Cart Pos (m)");      axs[0].grid(True)
    axs[1].plot(time_list, target_pos_list);     axs[1].set_ylabel("Target Pos (m)");    axs[1].grid(True)
    axs[2].plot(time_list, pole_angle_list);     axs[2].set_ylabel("Pole Angle (rad)");  axs[2].grid(True)
    axs[3].plot(time_list, target_angle_list);   axs[3].set_ylabel("Target Angle (rad)");axs[3].grid(True)
    axs[4].plot(time_list, [t - a for t, a in zip(target_angle_list, pole_angle_list)], color="magenta")
    axs[4].set_ylabel("Angle Error (rad)");axs[4].grid(True)
    axs[5].plot(time_list, pole_angle_vel_list); axs[5].set_ylabel("Pole Vel (rad/s)");   axs[5].grid(True)
    axs[6].plot(time_list, force_list);          axs[6].set_ylabel("Force (N)");         axs[6].set_xlabel("Time (s)"); axs[6].grid(True)

    pid_text = (
        f"Pos PID: Kp={pos_kp}, Ki={pos_ki}, Kd={pos_kd}    "
        f"Ang PID: Kp={ang_kp}, Ki={ang_ki}, Kd={ang_kd}"
    )
    fig.suptitle("Inverted Pendulum Two-Level PID with Keyboard Target Control\n" + pid_text,
                 fontsize=12, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

