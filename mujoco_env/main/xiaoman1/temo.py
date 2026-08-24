import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'

import mujoco
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=240, width=320)
renderer.update_scene(data)
image = renderer.render()
print("EGL 渲染成功! 图像 shape:", image.shape)