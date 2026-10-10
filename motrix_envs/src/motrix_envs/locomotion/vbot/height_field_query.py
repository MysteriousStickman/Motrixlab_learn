"""Static XML collision-surface heights, independent of motrixsim.

Requires numpy and mujoco. MuJoCo compiles PNG normalization, asset paths,
body transforms and attached models; downward rays also cover boxes/meshes.
This is privileged static-map perception, not a simulated onboard sensor.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def make_height_scan_grid(x_range, y_range, resolution):
    """
    按范围和间距生成机身水平坐标系的网格：+X 向前，+Y 向左。
    """
    spacing = np.asarray(resolution, dtype=np.float64)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 2)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("height scan resolution must be a positive scalar or XY pair.")
    axes = []
    for bounds, step in zip((x_range, y_range), spacing):
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] < bounds[0]:
            raise ValueError("height scan ranges must be finite (min, max) pairs with min <= max.")
        intervals = (bounds[1] - bounds[0]) / step
        count = int(round(intervals))
        if not np.isclose(intervals, count, rtol=0, atol=1e-6):
            raise ValueError("height scan range widths must be multiples of resolution.")
        axes.append(np.linspace(bounds[0], bounds[1], count + 1))
    # X 为外层、Y 为内层；每次观测都保持相同的点位排列顺序。
    x, y = np.meshgrid(*axes, indexing="ij")
    return np.stack((x.ravel(), y.ravel()), axis=-1).astype(np.float32)


class TerrainHeightQuery:
    """查世界坐标 XY 下方最高的静态碰撞面；查不到返回 NaN。

    使用独立的 MuJoCo 地形副本，不含机器人；地形如果会移动，此副本不会同步。
    """

    def __init__(self, model_file, terrain_prefixes=("C1_", "C2_", "C3_")):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("TerrainHeightQuery requires the 'mujoco' package.") from exc
        self._mj = mujoco
        scene_path = Path(model_file).resolve()
        scene = ET.parse(scene_path).getroot()

        # 场景 XML 的 model 声明指向子地形文件；attach 的前缀标识碰撞地形。
        assets = {node.get("name"): node.get("file") for node in scene.findall("./asset/model")}
        self._models = []
        self.ray_start_z = -np.inf
        for attachment in scene.findall("./worldbody/attach"):
            prefix = attachment.get("prefix", "")
            if not prefix.startswith(tuple(terrain_prefixes)):
                continue
            asset_file = assets.get(attachment.get("model"))
            if not asset_file:
                raise ValueError(f"Missing collision model for attachment {prefix!r}.")
            
            # 加载碰撞模型
            model = self._load_collision_model((scene_path.parent / asset_file).resolve())
            data = mujoco.MjData(model)
            # 计算各地形几何体的世界位置，供后面的射线检测使用。
            mujoco.mj_forward(model, data)
            selected = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            if not len(selected):
                continue
            model.geom_group[:] = 0
            model.geom_group[selected] = 5
            # 射线起点需高于所有地形。rbound 是几何体的保守包围球半径。
            tops = data.geom_xpos[selected, 2] + model.geom_rbound[selected]
            self.ray_start_z = max(self.ray_start_z, float(np.max(tops) + 1.0))
            self._models.append((model, data))
        if not self._models:
            raise ValueError(f"No collision terrain found for prefixes {terrain_prefixes}.")
        # mj_ray 仅检测第 5 组的碰撞体，避开非碰撞几何体。
        self._groups = np.asarray([0, 0, 0, 0, 0, 1], dtype=np.uint8)
        self._direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        self._geom_id = np.empty(1, dtype=np.int32)

    def _load_collision_model(self, path):
        """在内存中兼容地形 XML 和资源路径；不改磁盘上的 XML。"""
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        asset_dir = path.parent / (compiler.get("assetdir", "") if compiler is not None else "")
        assets = {}
        for element in root.findall("./asset/*"):
            filename = element.get("file")
            if filename:
                assets[filename] = (asset_dir / filename).read_bytes()
        if compiler is not None:
            compiler.attrib.pop("assetdir", None)
        for hfield in root.findall("./asset/hfield"):
            size = hfield.get("size")
            if size:
                parts = size.split()
                # Some MuJoCo releases reject zero hfield base thickness.
                # A tiny positive thickness leaves the top surface unchanged.
                if len(parts) == 4 and float(parts[3]) == 0.0:
                    parts[3] = "0.000001"
                    hfield.set("size", " ".join(parts))
        return self._mj.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode"), assets=assets
        )

    def query_batch(self, xy):
        """查询层：逐点射线查询世界坐标系 Z
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError("xy must contain finite coordinates with shape (..., 2).")
        result = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
        origin = np.asarray([0.0, 0.0, self.ray_start_z], dtype=np.float64)
        for i, point in enumerate(xy.reshape(-1, 2)):
            origin[:2] = point
            for model, data in self._models:
                # mj_ray 返回射线到首次碰撞的距离；未击中返回 -1。射线起点在所有地形之上，方向向下。
                distance = self._mj.mj_ray(
                    model, data, origin, self._direction,
                    self._groups, True, -1, self._geom_id,
                )
                if distance >= 0:
                    # 地形世界高度 = 射线起点高度 - 命中距离
                    height = self.ray_start_z - distance
                    result.flat[i] = np.fmax(result.flat[i], height)
        return result


    def get_height_scan(self, base_xy, yaw, offsets):
        """
        本体坐标 → 世界坐标 → 高度查询
        """
        base_xy = np.asarray(base_xy, dtype=np.float64)
        yaw = np.asarray(yaw, dtype=np.float64)
        offsets = np.asarray(offsets, dtype=np.float64)
        if base_xy.ndim != 2 or base_xy.shape[1] != 2:
            raise ValueError("base_xy must have shape (batch, 2).")
        if yaw.shape != (len(base_xy),) or offsets.ndim != 2 or offsets.shape[1] != 2:
            raise ValueError("yaw must be (batch,) and offsets must be (N, 2).")
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        dx, dy = offsets[:, 0][None, :], offsets[:, 1][None, :]
        world_xy = base_xy[:, None, :] + np.stack([c * dx - s * dy, s * dx + c * dy], axis=-1)
        return self.query_batch(world_xy)
