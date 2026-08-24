import mujoco
import mujoco.viewer
import numpy as np
from thermal_sensor import ThermalSensor
import time


def main():
    # 加载环境模型
    model_path = "../robot_model/env/env.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 创建RGB渲染器用于对比 (width, height)
    from mujoco.renderer import Renderer
    rgb_renderer = Renderer(model, width=240, height=320)
    
    # 创建热传感器（使用env.xml中的thermal_cam）
    thermal_sensor = ThermalSensor(
        model=model,
        data=data,
        cam_name="thermal_cam",
        width=240,
        height=320,
        enable_thermal_blur=True,
        blur_kernel_size=7,
        blur_sigma=0.4,
        enable_distance_attenuation=True,
        attenuation_coefficient=0.05,
        enable_noise=True,
        noise_stddev=0.3
    )
    
    # 设置容器内液体温度（随机且分层，便于任务排序）
    obj_base_temp = float(np.random.uniform(0.0, 60.0))
    object_temps = np.array([obj_base_temp, obj_base_temp + 20.0, obj_base_temp + 40.0], dtype=np.float64)
    np.random.shuffle(object_temps)
    thermal_sensor.set_liquid_temperature("beaker1", float(object_temps[0]), glass_conductivity=0.5)
    
    # 加热设备
    thermal_sensor.set_body_temperature("bench", 35.0)  # 桌面稍热
    
    # 测试管架
    thermal_sensor.set_body_temperature("target_place_table_1", 45.0)
    thermal_sensor.set_body_temperature("target_place_table_2", 45.0)
    thermal_sensor.set_body_temperature("target_place_table_3", 45.0)
    
    thermal_sensor.set_liquid_temperature("graduated_cylinder", float(object_temps[1]), glass_conductivity=0.5)
    thermal_sensor.set_liquid_temperature("erlenmeyer_flask", float(object_temps[2]), glass_conductivity=0.5)
    
    # 消防设备（常温）
    thermal_sensor.set_body_temperature("fire_extinguisher", 20.0)
    
    # 废物箱
    thermal_sensor.set_body_temperature("waste_bin", 28.0)
    
    print("热传感器初始化完成！")
    print("按任意键退出")
    print("\n当前物体温度设置（液体→玻璃热传导）：")
    print(f"  beaker1: 液体{object_temps[0]:.1f}°C")
    print(f"  graduated_cylinder: 液体{object_temps[1]:.1f}°C")
    print(f"  erlenmeyer_flask: 液体{object_temps[2]:.1f}°C")
    print("  target_place_table_1: 45°C")
    print("  target_place_table_2: 45°C")
    print("  target_place_table_3: 45°C")
    print("  其他物体: 20-35°C")
    
    # 只渲染一帧用于测试
    mujoco.mj_step(model, data)
    
    # 渲染RGB图像
    rgb_renderer.update_scene(data, camera=thermal_sensor.camera_id)
    rgb_image = rgb_renderer.render()
    
    # 渲染热成像（启用调试）
    print("\n开始渲染热成像...")
    temperature_map, body_id_map = thermal_sensor.render_thermal_image(debug=True)
    
    # 生成灰度热成像图（0-100度映射到0-255）
    grayscale_thermal = thermal_sensor.temperature_to_grayscale(temperature_map, 0.0, 100.0)
    
    # 生成伪彩色热成像图（用于可视化对比）
    thermal_color = thermal_sensor.temperature_to_color(temperature_map, 0.0, 100.0)
    
    # 拼接图像（左到右：灰度热成像、伪彩色热成像、RGB灰度、RGB彩色）
    import cv2
    
    # RGB转灰度
    rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # 融合图：RGB灰度 * 0.65 + 热成像灰度 * 0.35
    blended = thermal_sensor.blend_with_rgb(grayscale_thermal, rgb_image, thermal_weight=0.85)
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)

    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    thermal_bgr = cv2.cvtColor(thermal_color, cv2.COLOR_RGB2BGR)
    grayscale_bgr = cv2.cvtColor(grayscale_thermal, cv2.COLOR_GRAY2BGR)
    rgb_gray_bgr = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([grayscale_bgr, thermal_bgr, rgb_gray_bgr, rgb_bgr, blended_bgr])

    # 添加文字标签
    cv2.putText(combined, "Thermal Gray", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(combined, "Thermal Color", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(combined, "RGB Gray", (490, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(combined, "RGB", (730, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(combined, "Blended(IR35+G65)", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 显示温度统计
    valid_temps = temperature_map[temperature_map > 0]
    if len(valid_temps) > 0:
        print(f"\n温度统计 - 最小: {valid_temps.min():.1f}°C, "
              f"最大: {valid_temps.max():.1f}°C, "
              f"平均: {valid_temps.mean():.1f}°C")
    
    # 保存图像
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"thermal_test_{timestamp}.png"
    cv2.imwrite(filename, combined)
    print(f"测试图像已保存: {filename}")
    
    # 显示并等待按键
    cv2.imshow("Thermal Sensor Test", combined)
    print("\n按任意键退出...")
    cv2.waitKey(0)
    
    thermal_sensor.close_all_windows()
    print("程序结束")


if __name__ == "__main__":
    main()
