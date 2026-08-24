import mujoco
from mujoco import renderer
import numpy as np
import cv2


class ThermalSensor:
    def __init__(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        cam_name: str,
        width: int = 640,
        height: int = 480,
        body_temperatures: dict = None,
        ambient_temperature: float = 25.0,
        enable_distance_attenuation: bool = True,
        attenuation_coefficient: float = 0.1,
        enable_noise: bool = True,
        noise_stddev: float = 0.5,
        enable_internal_gradient: bool = True,
        edge_temperature_ratio: float = 0.7,
        enable_thermal_blur: bool = True,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.5
    ):
        """
        基于深度渲染器的红外热传感器（使用geom_id缓冲）。
        
        参数:
            model: MuJoCo 模型
            data: MuJoCo 数据
            cam_name: 相机名称
            width: 图像宽度
            height: 图像高度
            body_temperatures: body_id -> 温度(°C) 的字典，默认所有物体25°C
            ambient_temperature: 环境温度(°C)
            enable_distance_attenuation: 是否启用距离衰减
            attenuation_coefficient: 衰减系数（越大衰减越快）
            enable_noise: 是否启用噪声
            noise_stddev: 噪声标准差(°C)
            enable_internal_gradient: 是否启用物体内部温度梯度
            edge_temperature_ratio: 边缘温度相对中心温度的比例(0-1)
            enable_thermal_blur: 是否启用热扩散模糊（模拟物体-环境热交换）
            blur_kernel_size: 模糊核大小（奇数）
            blur_sigma: 高斯模糊标准差
        """
        self.model = model
        self.data = data
        self.cam_name = cam_name
        self.camera_id = self.model.cam(cam_name).id
        self.width = width
        self.height = height
        self.ambient_temperature = ambient_temperature
        self.enable_distance_attenuation = enable_distance_attenuation
        self.attenuation_coefficient = attenuation_coefficient
        self.enable_noise = enable_noise
        self.noise_stddev = noise_stddev
        self.enable_internal_gradient = enable_internal_gradient
        self.edge_temperature_ratio = edge_temperature_ratio
        self.enable_thermal_blur = enable_thermal_blur
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.cuda_postprocess_enabled = self._check_cuda_postprocess_available()
        self._cuda_mode_logged = False
        
        self.renderer = renderer.Renderer(model, height=height, width=width)
        self._scene_cache_valid = False
        self._scene_cache_camera_id = None
        
        if body_temperatures is None:
            self.body_temperatures = {i: ambient_temperature for i in range(model.nbody)}
        else:
            self.body_temperatures = body_temperatures
        
        self.geom_temperatures = {}
        self.liquid_temperatures = {}
        self._geom_temp_lut_cache = np.full(self.model.ngeom, self.ambient_temperature, dtype=np.float32)
        self._geom_temp_lut_cache_valid = False
        self._cuda_filters = {}
        self._cuda_morph_filters = {}
        self._cuda_gmat_u8 = None
        self._cuda_gmat_f32 = None
        self._cuda_rgb_gmat = None
        self._cuda_gray_gmat = None
        self._cuda_thermal_gmat = None

    def _check_cuda_postprocess_available(self) -> bool:
        """
        检查 OpenCV CUDA 后处理能力是否可用。

        这里只加速热成像后处理，不涉及 MuJoCo 本身渲染。
        因此判定条件是：
        1. cv2.cuda 命名空间存在
        2. OpenCV 能看到至少 1 个 CUDA 设备
        3. 需要的基础算子存在
        """
        if not hasattr(cv2, "cuda"):
            return False
        try:
            device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            return False
        if device_count <= 0:
            return False
        required_attrs = (
            "GpuMat",
            "createGaussianFilter",
            "cvtColor",
            "addWeighted",
        )
        return all(hasattr(cv2.cuda, attr) for attr in required_attrs)

    def _invalidate_temperature_cache(self) -> None:
        """
        温度配置变化后使 geom 温度 LUT 失效。
        """
        self._geom_temp_lut_cache_valid = False

    def _rebuild_geom_temperature_lut(self) -> None:
        """
        重建 geom -> temperature 的查找表。

        当 enable_internal_gradient=False 时，每个 geom 的温度仅由：
        1. geom 显式温度
        2. body 温度
        3. ambient 温度
        决定，因此可以缓存，避免每帧重复计算。
        """
        lut = np.full(self.model.ngeom, self.ambient_temperature, dtype=np.float32)
        for geom_id in range(self.model.ngeom):
            if geom_id in self.geom_temperatures:
                lut[geom_id] = float(self.geom_temperatures[geom_id])
            else:
                body_id = int(self.model.geom_bodyid[geom_id])
                lut[geom_id] = float(self.body_temperatures.get(body_id, self.ambient_temperature))
        self._geom_temp_lut_cache = lut
        self._geom_temp_lut_cache_valid = True

    def _get_geom_temperature_lut(self) -> np.ndarray:
        """
        获取 geom 温度 LUT。

        说明：
        - 若启用了内部温度梯度，则温度依赖几何位置/姿态，不做静态缓存。
        - 若未启用内部温度梯度，则 LUT 可安全缓存。
        """
        if self.enable_internal_gradient:
            lut = np.full(self.model.ngeom, self.ambient_temperature, dtype=np.float32)
            for geom_id in range(self.model.ngeom):
                body_id = int(self.model.geom_bodyid[geom_id])
                lut[geom_id] = float(self._compute_internal_temperature(body_id, geom_id, (0, 0)))
            return lut

        if not self._geom_temp_lut_cache_valid:
            self._rebuild_geom_temperature_lut()
        return self._geom_temp_lut_cache

    def _log_cuda_mode_once(self) -> None:
        """
        仅打印一次当前热成像后处理使用的计算后端。
        """
        if self._cuda_mode_logged:
            return
        mode = "CUDA" if self.cuda_postprocess_enabled else "CPU"
        print(f"✓ Thermal postprocess backend: {mode}")
        self._cuda_mode_logged = True

    def _get_cuda_gaussian_filter(self, cv_type: int, kernel_size: int, sigma: float):
        """
        复用 CUDA Gaussian filter，避免每帧重新创建。
        """
        key = (cv_type, kernel_size, float(sigma))
        gaussian = self._cuda_filters.get(key)
        if gaussian is None:
            gaussian = cv2.cuda.createGaussianFilter(
                cv_type,
                cv_type,
                (kernel_size, kernel_size),
                sigma,
            )
            self._cuda_filters[key] = gaussian
        return gaussian

    def _get_cuda_morph_filter(self, op: int, cv_type: int, kernel: np.ndarray):
        """
        复用 CUDA 形态学 filter，避免每帧重新创建。
        """
        kernel_key = (op, cv_type, kernel.shape, kernel.dtype.str, kernel.tobytes())
        morph = self._cuda_morph_filters.get(kernel_key)
        if morph is None:
            morph = cv2.cuda.createMorphologyFilter(op, cv_type, kernel)
            self._cuda_morph_filters[kernel_key] = morph
        return morph

    def _gaussian_blur_u8(self, image_u8: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        """
        对 uint8 图像做高斯模糊，优先走 CUDA，失败则回退 CPU。
        """
        if self.cuda_postprocess_enabled:
            try:
                if self._cuda_gmat_u8 is None:
                    self._cuda_gmat_u8 = cv2.cuda_GpuMat()
                self._cuda_gmat_u8.upload(image_u8)
                gaussian = self._get_cuda_gaussian_filter(
                    self._cuda_gmat_u8.type(),
                    kernel_size,
                    sigma,
                )
                return gaussian.apply(self._cuda_gmat_u8).download()
            except Exception:
                self.cuda_postprocess_enabled = False
        return cv2.GaussianBlur(image_u8, (kernel_size, kernel_size), sigma)

    def _gaussian_blur_f32(self, image_f32: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        """
        对 float32 图像做高斯模糊，优先走 CUDA，失败则回退 CPU。
        """
        src_f32 = np.ascontiguousarray(image_f32, dtype=np.float32)
        if self.cuda_postprocess_enabled:
            try:
                if self._cuda_gmat_f32 is None:
                    self._cuda_gmat_f32 = cv2.cuda_GpuMat()
                self._cuda_gmat_f32.upload(src_f32)
                gaussian = self._get_cuda_gaussian_filter(
                    self._cuda_gmat_f32.type(),
                    kernel_size,
                    sigma,
                )
                return gaussian.apply(self._cuda_gmat_f32).download()
            except Exception:
                self.cuda_postprocess_enabled = False
        return cv2.GaussianBlur(src_f32, (kernel_size, kernel_size), sigma)

    def _dilate_u8(self, image_u8: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        对 uint8 掩码做膨胀，优先走 CUDA，失败则回退 CPU。
        """
        src_u8 = np.ascontiguousarray(image_u8, dtype=np.uint8)
        if self.cuda_postprocess_enabled:
            try:
                if self._cuda_gmat_u8 is None:
                    self._cuda_gmat_u8 = cv2.cuda_GpuMat()
                self._cuda_gmat_u8.upload(src_u8)
                morph = self._get_cuda_morph_filter(
                    cv2.MORPH_DILATE,
                    self._cuda_gmat_u8.type(),
                    kernel,
                )
                return morph.apply(self._cuda_gmat_u8).download()
            except Exception:
                self.cuda_postprocess_enabled = False
        return cv2.dilate(src_u8, kernel, iterations=1)

    def _rgb_to_gray_u8(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        RGB 转灰度，优先走 CUDA，失败则回退 CPU。
        """
        if self.cuda_postprocess_enabled:
            try:
                if self._cuda_rgb_gmat is None:
                    self._cuda_rgb_gmat = cv2.cuda_GpuMat()
                self._cuda_rgb_gmat.upload(rgb_image)
                self._cuda_gray_gmat = cv2.cuda.cvtColor(self._cuda_rgb_gmat, cv2.COLOR_RGB2GRAY)
                return self._cuda_gray_gmat.download()
            except Exception:
                self.cuda_postprocess_enabled = False
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    def _blend_gray_u8(self, rgb_gray: np.ndarray, thermal_gray: np.ndarray, thermal_weight: float) -> np.ndarray:
        """
        灰度融合，公式为：
          I_blend = (1 - alpha) * I_rgb + alpha * I_thermal
        其中 alpha = thermal_weight。

        优先走 CUDA，失败则回退 CPU。
        """
        rgb_gray_u8 = np.ascontiguousarray(rgb_gray, dtype=np.uint8)
        thermal_gray_u8 = np.ascontiguousarray(thermal_gray, dtype=np.uint8)
        if self.cuda_postprocess_enabled:
            try:
                if self._cuda_gray_gmat is None:
                    self._cuda_gray_gmat = cv2.cuda_GpuMat()
                if self._cuda_thermal_gmat is None:
                    self._cuda_thermal_gmat = cv2.cuda_GpuMat()
                self._cuda_gray_gmat.upload(rgb_gray_u8)
                self._cuda_thermal_gmat.upload(thermal_gray_u8)
                blended = cv2.cuda.addWeighted(
                    self._cuda_gray_gmat,
                    1.0 - thermal_weight,
                    self._cuda_thermal_gmat,
                    thermal_weight,
                    0.0,
                )
                return blended.download()
            except Exception:
                self.cuda_postprocess_enabled = False
        blended = ((1.0 - thermal_weight) * rgb_gray_u8.astype(np.float32)
                   + thermal_weight * thermal_gray_u8.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)
        
    def set_body_temperature(self, body_name: str, temperature: float):
        """设置指定物体的温度"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.body_temperatures[body_id] = temperature
            self._invalidate_temperature_cache()
        except:
            print(f"Warning: Body '{body_name}' not found in model")
    
    def set_body_temperature_by_id(self, body_id: int, temperature: float):
        """通过body_id设置温度"""
        if 0 <= body_id < self.model.nbody:
            self.body_temperatures[body_id] = temperature
            self._invalidate_temperature_cache()
        else:
            print(f"Warning: Invalid body_id {body_id}")
    
    def set_geom_temperature(self, geom_name: str, temperature: float):
        """设置指定geom的温度（优先级高于body温度）"""
        try:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.geom_temperatures[geom_id] = temperature
            self._invalidate_temperature_cache()
        except:
            print(f"Warning: Geom '{geom_name}' not found in model")
    
    def set_liquid_temperature(self, body_name: str, liquid_temp: float, glass_conductivity: float = 0.5):
        """
        设置容器内液体温度，自动计算玻璃温度
        
        参数:
            body_name: 容器body名称（如"beaker1"）
            liquid_temp: 液体温度(°C)
            glass_conductivity: 玻璃热传导系数(0-1)，越大玻璃温度越接近液体温度
        """
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.liquid_temperatures[body_id] = liquid_temp
            
            glass_temp = self.ambient_temperature + (liquid_temp - self.ambient_temperature) * glass_conductivity
            
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == body_id:
                    geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    if geom_name and 'liquid' not in geom_name.lower():
                        self.geom_temperatures[geom_id] = glass_temp
            self._invalidate_temperature_cache()
                        
        except:
            print(f"Warning: Body '{body_name}' not found in model")
    
    def _compute_internal_temperature(self, body_id: int, geom_id: int, pixel_pos_2d: tuple) -> float:
        """
        计算物体内部某点的温度（基于到中心的距离）
        
        参数:
            body_id: 物体ID
            geom_id: 几何体ID
            pixel_pos_2d: 像素在图像中的2D位置 (u, v)
        
        返回:
            该点的温度
        """
        if geom_id in self.geom_temperatures:
            base_temp = self.geom_temperatures[geom_id]
        else:
            base_temp = self.body_temperatures.get(body_id, self.ambient_temperature)
        
        if not self.enable_internal_gradient:
            return base_temp
        
        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name and 'liquid' not in geom_name.lower() and body_id in self.liquid_temperatures:
            liquid_temp = self.liquid_temperatures[body_id]
            body_center_3d = self.data.xpos[body_id]
            geom_pos_3d = self.data.geom_xpos[geom_id]
            
            height_diff = geom_pos_3d[2] - body_center_3d[2]
            
            body_geoms = [gid for gid in range(self.model.ngeom) if self.model.geom_bodyid[gid] == body_id]
            max_height = 0.0
            min_height = 0.0
            for gid in body_geoms:
                gpos = self.data.geom_xpos[gid]
                h = gpos[2] - body_center_3d[2]
                max_height = max(max_height, h)
                min_height = min(min_height, h)
            
            height_range = max_height - min_height
            if height_range > 1e-6:
                normalized_height = (height_diff - min_height) / height_range
            else:
                normalized_height = 0.0
            
            bottom_temp = self.ambient_temperature + (liquid_temp - self.ambient_temperature) * 0.6
            top_temp = self.ambient_temperature + (liquid_temp - self.ambient_temperature) * 0.2
            
            temperature = bottom_temp + (top_temp - bottom_temp) * normalized_height
            
            return temperature
        
        body_geoms = [gid for gid in range(self.model.ngeom) if self.model.geom_bodyid[gid] == body_id]
        if len(body_geoms) == 0:
            return base_temp
        
        body_center_3d = self.data.xpos[body_id]
        geom_pos_3d = self.data.geom_xpos[geom_id]
        
        distance_to_center = np.linalg.norm(geom_pos_3d - body_center_3d)
        
        max_distance = 0.0
        for gid in body_geoms:
            gpos = self.data.geom_xpos[gid]
            dist = np.linalg.norm(gpos - body_center_3d)
            max_distance = max(max_distance, dist)
        
        if max_distance < 1e-6:
            normalized_distance = 0.0
        else:
            normalized_distance = distance_to_center / max_distance
        
        edge_temp = self.ambient_temperature + (base_temp - self.ambient_temperature) * self.edge_temperature_ratio
        
        temperature = base_temp - (base_temp - edge_temp) * normalized_distance
        
        return temperature
    
    def _apply_thermal_blur(self, temperature_map: np.ndarray, body_id_map: np.ndarray) -> np.ndarray:
        """
        热交换模糊：物体加热周围环境，环境像素温度随距离衰减
        
        物理模型：
        - 物体边缘温度向环境温度降低
        - 环境（world/背景）被物体加热，温度随距离衰减
        - 距离物体越近，环境温度越高
        
        参数:
            temperature_map: 温度图
            body_id_map: body_id分割图
        
        返回:
            应用热交换后的温度图
        """
        result = temperature_map.copy()
        
        object_mask = (body_id_map > 0).astype(np.uint8)
        background_mask = (body_id_map <= 0)
        
        if object_mask.sum() == 0:
            return result
        
        distance_to_object = cv2.distanceTransform(1 - object_mask, cv2.DIST_L2, 5)
        
        max_heat_distance = self.blur_kernel_size * 2
        heat_influence = np.exp(-distance_to_object / max_heat_distance)
        heat_influence = np.clip(heat_influence, 0.0, 1.0)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated_object_mask = self._dilate_u8(object_mask, kernel)
        object_edge = dilated_object_mask - object_mask
        object_edge = object_edge.astype(bool)
        
        edge_temp_map = np.zeros_like(temperature_map)
        edge_temp_map[object_edge] = result[object_edge]
        
        blurred_edge_temp = self._gaussian_blur_f32(edge_temp_map, self.blur_kernel_size, 0.0)
        blurred_edge_temp = np.where(blurred_edge_temp > 0, blurred_edge_temp, result[object_edge].mean() if object_edge.sum() > 0 else self.ambient_temperature)
        
        heated_env_temp = self.ambient_temperature + (blurred_edge_temp - self.ambient_temperature) * heat_influence * self.blur_sigma
        
        result[background_mask] = heated_env_temp[background_mask]
        
        return result
    
    def render_thermal_image(self, debug=False) -> tuple[np.ndarray, np.ndarray]:
        """
        渲染热成像图像（使用MuJoCo的segmentation功能）

        返回:
            temperature_map: 温度图 (height, width)，单位°C
            body_id_map: body_id图 (height, width)，-1表示背景
        """
        self.renderer.update_scene(self.data, camera=self.camera_id)
        self._scene_cache_valid = True
        self._scene_cache_camera_id = self.camera_id

        self.renderer.enable_segmentation_rendering()
        seg_image = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        self.renderer.enable_depth_rendering()
        depth_image = self.renderer.render()
        self.renderer.disable_depth_rendering()

        temperature_map = np.full((self.height, self.width), self.ambient_temperature, dtype=np.float32)
        body_id_map = np.full((self.height, self.width), -1, dtype=np.int32)

        if debug:
            print(f"分割图像形状: {seg_image.shape}, dtype: {seg_image.dtype}")
            print(f"深度图像形状: {depth_image.shape}, 范围: [{depth_image.min():.3f}, {depth_image.max():.3f}]")
            print(f"分割ID范围: [{seg_image.min()}, {seg_image.max()}]")
            unique_seg_ids = np.unique(seg_image)
            print(f"唯一的seg_id数量: {len(unique_seg_ids)}, 值: {unique_seg_ids[:20]}")

        geom_id_image = seg_image[:, :, 0].astype(np.int32)
        valid_mask = (geom_id_image >= 0) & (geom_id_image < self.model.ngeom)

        if not valid_mask.any():
            return temperature_map, body_id_map

        # --- body_id_map（向量化） ---
        body_id_map[valid_mask] = self.model.geom_bodyid[geom_id_image[valid_mask]]

        # --- geom 温度 LUT ---
        # 当内部梯度关闭时，这里会复用缓存，避免每帧遍历 unique geom 重算。
        geom_temp_lut = self._get_geom_temperature_lut()

        # --- LUT 查表赋温度（向量化） ---
        temperature_map[valid_mask] = geom_temp_lut[geom_id_image[valid_mask]]

        # --- 距离衰减（向量化） ---
        if self.enable_distance_attenuation:
            dist_mask = valid_mask & (depth_image > 0)
            temp_diff = temperature_map[dist_mask] - self.ambient_temperature
            temperature_map[dist_mask] = (
                self.ambient_temperature
                + temp_diff * np.exp(-self.attenuation_coefficient * depth_image[dist_mask])
            )

        if self.enable_noise:
            noise = np.random.normal(0, self.noise_stddev, (self.height, self.width))
            temperature_map += noise

        if self.enable_thermal_blur:
            temperature_map = self._apply_thermal_blur(temperature_map, body_id_map)

        if debug:
            unique_bodies_found = set(body_id_map[body_id_map >= 0].flatten())
            print(f"检测到的body数量: {len(unique_bodies_found)}")
            for body_id in sorted(unique_bodies_found):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None:
                    body_name = f"body_{body_id}"
                object_temp = self.body_temperatures.get(body_id, self.ambient_temperature)
                mask = body_id_map == body_id
                measured_temps = temperature_map[mask]
                avg_measured = measured_temps.mean()
                avg_distance = depth_image[mask].mean() if mask.sum() > 0 else 0
                pixel_count = mask.sum()
                print(f"  Body {body_id} ({body_name}): 设定={object_temp:.1f}°C, "
                      f"测量平均={avg_measured:.1f}°C, 平均距离={avg_distance:.3f}m, 像素数={pixel_count}")

        return temperature_map, body_id_map
    
    def temperature_to_grayscale(self, temperature_map: np.ndarray,
                                temp_min: float = 0.0,
                                temp_max: float = 100.0) -> np.ndarray:
        """将温度图转换为灰度图（0-100度映射到0-255灰度）"""
        grayscale = np.clip(
            (temperature_map - temp_min) / (temp_max - temp_min) * 255,
            0,
            255
        ).astype(np.uint8)
        return grayscale
    
    def temperature_to_color(self, temperature_map: np.ndarray, 
                            temp_min: float = 15.0, 
                            temp_max: float = 100.0) -> np.ndarray:
        """将温度图转换为伪彩色热成像"""
        temp_normalized = np.clip(
            (temperature_map - temp_min) / (temp_max - temp_min) * 255, 
            0, 
            255
        ).astype(np.uint8)
        
        thermal_image = cv2.applyColorMap(temp_normalized, cv2.COLORMAP_JET)
        thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
        
        return thermal_image
    
    def render_rgb(self) -> np.ndarray:
        """从热成像相机同视角渲染 RGB 图像 (H, W, 3)，RGB 格式"""
        if (not self._scene_cache_valid) or (self._scene_cache_camera_id != self.camera_id):
            self.renderer.update_scene(self.data, camera=self.camera_id)
            self._scene_cache_valid = True
            self._scene_cache_camera_id = self.camera_id
        return self.renderer.render()

    def blend_with_rgb(self, thermal_gray: np.ndarray, rgb_image: np.ndarray,
                       thermal_weight: float = 0.65) -> np.ndarray:
        """
        将热成像灰度图与 RGB 灰度图融合。

        参数:
            thermal_gray: 热成像灰度图 (H, W)，uint8
            rgb_image: 原始 RGB 图像 (H, W, 3)
            thermal_weight: 热成像灰度权重（默认 0.35），RGB 灰度权重 = 1 - thermal_weight
        返回:
            融合灰度图 (H, W)，uint8
        """
        rgb_gray = self._rgb_to_gray_u8(rgb_image)
        return self._blend_gray_u8(rgb_gray, thermal_gray, thermal_weight)

    def postprocess_thermal_frame(
        self,
        temperature_map: np.ndarray,
        rgb_image: np.ndarray,
        temp_min: float = 0.0,
        temp_max: float = 100.0,
        blur_kernel_size: int = 25,
        blur_sigma: float = 5.0,
        thermal_weight: float = 0.85,
    ) -> np.ndarray:
        """
        热成像后处理统一入口。

        流程：
        1. 温度图 -> 灰度图
        2. 高斯模糊
        3. 与 RGB 灰度图融合

        其中 2 和 3 优先走 CUDA，以减少 CPU 后处理耗时。
        """
        self._log_cuda_mode_once()
        thermal_gray = self.temperature_to_grayscale(temperature_map, temp_min, temp_max)
        thermal_gray = self._gaussian_blur_u8(thermal_gray, blur_kernel_size, blur_sigma)
        return self.blend_with_rgb(thermal_gray, rgb_image, thermal_weight=thermal_weight)

    def display_thermal(self, thermal_image: np.ndarray, window_name: str = "Thermal Camera"):
        """显示热成像图像"""
        bgr = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr)
    
    @staticmethod
    def wait_for_key(delay: int = 1) -> int:
        """等待按键输入"""
        return cv2.waitKey(delay)
    
    @staticmethod
    def close_all_windows():
        """关闭所有OpenCV窗口"""
        cv2.destroyAllWindows()


# ── ROS2 发布器（可选依赖）──────────────────────────────────────────────────
_ros2_pub_available = False
_ros2_pub_error = ""
try:
    import rclpy
    from sensor_msgs.msg import Image as _Image
    from rclpy.qos import qos_profile_sensor_data
    from cv_bridge import CvBridge as _CvBridge
    _ros2_pub_available = True
except ImportError as _e:
    _ros2_pub_error = str(_e)


class ThermalImagePublisher:
    """
    将热成像融合图（灰度）发布到 ROS2 Image topic。

    发布 topic: /{topic}  (默认 /thermal_camera/image)
    编码格式: mono8
    """

    def __init__(
        self,
        topic: str = "/thermal_camera/image",
        node_name: str = "thermal_image_publisher",
    ) -> None:
        if not _ros2_pub_available:
            raise ImportError(
                f"ROS2 not available: {_ros2_pub_error}\n"
                "Please source your ROS2 environment:\n"
                "  source /opt/ros/humble/setup.bash"
            )
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._pub = self._node.create_publisher(_Image, topic, qos_profile_sensor_data)
        self._bridge = _CvBridge()
        self._topic = topic

    def publish(self, blended_gray: np.ndarray, stamp_sec: float | None = None) -> None:
        """
        发布融合灰度图。

        参数:
            blended_gray: uint8 灰度图 (H, W)
            stamp_sec: 时间戳（秒），None 则使用节点时钟
        """
        msg = self._bridge.cv2_to_imgmsg(blended_gray, encoding="mono8")
        if stamp_sec is None:
            msg.header.stamp = self._node.get_clock().now().to_msg()
        else:
            msg.header.stamp = rclpy.time.Time(seconds=stamp_sec).to_msg()
        msg.header.frame_id = "thermal_camera"
        self._pub.publish(msg)
        # 移除 spin_once：在高频显示循环中调用会导致回调重入错误
        # ROS2 处理由独立线程负责

    def shutdown(self) -> None:
        try:
            self._pub.destroy()
            self._node.destroy_node()
        except Exception:
            pass
