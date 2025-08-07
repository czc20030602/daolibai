import mujoco
import mujoco.viewer
import glfw
import numpy as np
import threading
import time

# ---- MJCF 定义倒立摆 ----
mjcf = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.8"/>
  <worldbody>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.2 0.1 0.05" rgba="0 0.8 0 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.02" rgba="0.8 0 0 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-20 20" gear="10"/>
  </actuator>
</mujoco>
"""

# ---- 初始化 ----
model = mujoco.MjModel.from_xml_string(mjcf)
data = mujoco.MjData(model)
data.qpos[1] = 0.26  # 初始角度不是垂直（15°）

# ---- 控制参数 ----
ctrl = 0.0
ctrl_lock = threading.Lock()

def key_callback(window, key, scancode, action, mods):
    global ctrl
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_LEFT:
            with ctrl_lock:
                ctrl = -1.0
        elif key == glfw.KEY_RIGHT:
            with ctrl_lock:
                ctrl = 1.0
    elif action == glfw.RELEASE:
        if key in [glfw.KEY_LEFT, glfw.KEY_RIGHT]:
            with ctrl_lock:
                ctrl = 0.0

# ---- 可视化线程 ----
def visual_thread():
    glfw.init()
    window = glfw.create_window(800, 600, "Inverted Pendulum", None, None)
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    renderer = mujoco.Renderer(model)
    while not glfw.window_should_close(window):
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(),
                               mujoco.MjvPerturb(), mujoco.MjvCamera(),
                               mujoco.MjvScene(), mujoco.MjvPerturb())
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=mujoco.MjvCamera())
        renderer.render()
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.01)
    glfw.terminate()

# ---- 物理仿真线程 ----
def sim_loop():
    while True:
        with ctrl_lock:
            data.ctrl[0] = ctrl
        mujoco.mj_step(model, data)
        time.sleep(0.01)

# ---- 启动线程 ----
t_vis = threading.Thread(target=visual_thread)
t_sim = threading.Thread(target=sim_loop)
t_vis.start()
t_sim.start()
