import time
import mujoco
import mujoco.viewer
import threading
from threading import Thread
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pygame

# 初始化 pygame
pygame.init()
pygame.display.set_mode((500, 500))

locker = threading.Lock()

# 记录数据
time_list = []
cart_pos_list = []
cart_vel_list = []
pole_angle_list = []
pole_angle_vel_list = []
force_list = []

# 键盘控制力
keyboard_force = 0.0
fixed_force = 50.0  # 固定的人工干预力

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

with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
    f.write(mjcf)
    xml_path = f.name

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.01
mj_data.qpos[1] = 0.1  # 初始角度

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# LQR 增益
K = np.array([-250/7, -200/7, -60274/525, -2776/105])

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
        time.sleep(0.01)

def SimulationThread():
    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()
        x = np.array([
            mj_data.qpos[0],
            mj_data.qvel[0],
            mj_data.qpos[1],
            mj_data.qvel[1],
        ])
        u = -K @ x + keyboard_force
        mj_data.ctrl[0] = u
        time_list.append(mj_data.time)
        cart_pos_list.append(x[0])
        cart_vel_list.append(x[1])
        pole_angle_list.append(x[2])
        pole_angle_vel_list.append(x[3])
        force_list.append(u)
        mujoco.mj_step(mj_model, mj_data)
        locker.release()
        dt = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if dt > 0:
            time.sleep(dt)

def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(0.02)

if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    key_thread = Thread(target=KeyboardThread)
    viewer_thread.start()
    sim_thread.start()
    key_thread.start()
    viewer_thread.join()
    sim_thread.join()
    key_thread.join()
    pygame.quit()

    # 画图
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

    plt.suptitle("Inverted Pendulum with LQR + Keyboard Control")
    plt.grid(True)
    plt.tight_layout()
    plt.show()