# run_car_glfw.py
import mujoco          # 官方 MuJoCo Python 绑定
import glfw            # 窗口和 OpenGL 上下文管理
import numpy as np

# 1) 加载模型
model = mujoco.MjModel.from_xml_path("simple_car.xml")
data  = mujoco.MjData(model)

# 2) 初始化 GLFW 并创建窗口
if not glfw.init():
    raise RuntimeError("glfw 初始化失败")
window = glfw.create_window(800, 600, "MuJoCo Simple Car", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("glfw 窗口创建失败")
glfw.make_context_current(window)

# 3) 创建 MuJoCo 渲染上下文、场景和摄像机设置
cam    = mujoco.MjvCamera()                      # 摄像机
opts   = mujoco.MjvOption()                      # 渲染选项（纹理、光照…）
scene  = mujoco.MjvScene(model, maxgeom=1000)    # 场景容器
ctx    = mujoco.GLContext(model)                 # OpenGL 上下文

# 4) 主循环：物理步进 -> 更新场景 -> 渲染 -> GLFW 刷新
while not glfw.window_should_close(window):
    # —— 控制：前后轮都给满正向油门
    data.ctrl[0] = 1.0   # motor_front
    data.ctrl[1] = 1.0   # motor_back

    # —— 物理模拟一步
    mujoco.mj_step(model, data)

    # —— 更新 MuJoCo 场景（内存拷贝）
    viewport = glfw.get_framebuffer_size(window)  # (width, height)
    mujoco.mjv_updateScene(
        model, data, opts, cam, scene
    )

    # —— 调用底层渲染器（OpenGL）
    mujoco.mjr_render(
        viewport, scene, ctx
    )

    # —— 交换帧缓冲 & 处理事件
    glfw.swap_buffers(window)
    glfw.poll_events()

# 5) 清理
glfw.terminate()
