from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObstacleBox:
    path: str
    minimum: np.ndarray
    maximum: np.ndarray

    def contains(self, pos):
        pos = np.asarray(pos, dtype=float)
        return bool(np.all(pos >= self.minimum) and np.all(pos <= self.maximum))


class StaticWorldMap:
    """Static obstacle map backed by inflated USD world bounding boxes."""

    def __init__(self, boxes=None, voxel_size=0.2):
        self.voxel_size = float(voxel_size)
        self.boxes = list(boxes or [])

    @classmethod
    def from_stage(
        cls,
        stage,
        obstacle_paths,
        voxel_size=0.2,
        inflation_radius=0.1,
    ):
        from pxr import UsdGeom

        cache = UsdGeom.BBoxCache(
            0.0,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )

        boxes = []
        inflation = np.full(3, float(inflation_radius), dtype=float)

        for path in obstacle_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue

            bbox = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bbox.IsEmpty():
                continue

            minimum = np.array(bbox.GetMin(), dtype=float) - inflation
            maximum = np.array(bbox.GetMax(), dtype=float) + inflation
            boxes.append(ObstacleBox(str(path), minimum, maximum))

        return cls(boxes=boxes, voxel_size=voxel_size)

    def is_occupied(self, pos):
        pos = np.asarray(pos, dtype=float)
        return any(box.contains(pos) for box in self.boxes)

    def obstacle_count(self):
        return len(self.boxes)

    def describe(self):
        return [
            {
                "path": box.path,
                "min": tuple(float(v) for v in box.minimum),
                "max": tuple(float(v) for v in box.maximum),
            }
            for box in self.boxes
        ]


class VoxelWorldMap:
    """Static voxel occupancy grid in world coordinates."""

    DEFAULT_EXCLUDE_PREFIXES = (
        "/World/iris",
        "/World/gripper_mount",
    )
    DEFAULT_EXCLUDE_KEYWORDS = (
        "ground",
        "floor",
        "road",
        "plane",
    )

    def __init__(
        self,
        occupied,
        map_min=(-10.0, -10.0, -1.0),
        map_max=(10.0, 10.0, 6.0),
        voxel_size=0.2,
        source_boxes=None,
    ):
        self.occupied = np.asarray(occupied, dtype=bool)
        self.map_min = np.asarray(map_min, dtype=float)
        self.map_max = np.asarray(map_max, dtype=float)
        self.voxel_size = float(voxel_size)
        self.shape = self.occupied.shape
        self.source_boxes = list(source_boxes or [])

    @classmethod
    def from_stage(
        cls,
        stage,
        obstacle_paths=None,
        map_min=(-10.0, -10.0, -1.0),
        map_max=(10.0, 10.0, 6.0),
        voxel_size=0.2,
        inflation_radius=0.1,
        exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES,
        exclude_keywords=DEFAULT_EXCLUDE_KEYWORDS,
    ):
        boxes = _stage_obstacle_boxes(
            stage=stage,
            obstacle_paths=obstacle_paths,
            inflation_radius=inflation_radius,
            exclude_prefixes=exclude_prefixes,
            exclude_keywords=exclude_keywords,
        )

        map_min = np.asarray(map_min, dtype=float)
        map_max = np.asarray(map_max, dtype=float)
        voxel_size = float(voxel_size)
        shape = np.ceil((map_max - map_min) / voxel_size).astype(int)
        occupied = np.zeros(tuple(shape), dtype=bool)

        for box in boxes:
            start = np.floor((box.minimum - map_min) / voxel_size).astype(int)
            stop = np.ceil((box.maximum - map_min) / voxel_size).astype(int) - 1

            start = np.maximum(start, 0)
            stop = np.minimum(stop, shape - 1)
            if np.any(stop < start):
                continue

            occupied[
                start[0] : stop[0] + 1,
                start[1] : stop[1] + 1,
                start[2] : stop[2] + 1,
            ] = True

        return cls(
            occupied=occupied,
            map_min=map_min,
            map_max=map_max,
            voxel_size=voxel_size,
            source_boxes=boxes,
        )

    def world_to_index(self, pos):
        pos = np.asarray(pos, dtype=float)
        return np.floor((pos - self.map_min) / self.voxel_size).astype(int)

    def in_bounds(self, pos):
        pos = np.asarray(pos, dtype=float)
        return bool(np.all(pos >= self.map_min) and np.all(pos < self.map_max))

    def is_occupied(self, pos):
        if not self.in_bounds(pos):
            return True

        idx = self.world_to_index(pos)
        return bool(self.occupied[tuple(idx)])

    def debug_point(self, pos):
        pos = np.asarray(pos, dtype=float)
        in_bounds = self.in_bounds(pos)
        idx = self.world_to_index(pos) if in_bounds else None
        containing = [
            box.path
            for box in self.source_boxes
            if box.contains(pos)
        ]
        return {
            "pos": tuple(float(v) for v in pos),
            "in_bounds": in_bounds,
            "index": None if idx is None else tuple(int(v) for v in idx),
            "occupied": self.is_occupied(pos),
            "source_boxes_containing_point": containing[:10],
            "source_box_count": len(containing),
        }

    def obstacle_count(self):
        return len(self.source_boxes)

    def occupied_count(self):
        return int(np.count_nonzero(self.occupied))

    def occupancy_fraction(self):
        return float(self.occupied_count() / self.occupied.size)

    def describe(self):
        return {
            "map_min": tuple(float(v) for v in self.map_min),
            "map_max": tuple(float(v) for v in self.map_max),
            "voxel_size": self.voxel_size,
            "shape": tuple(int(v) for v in self.shape),
            "source_obstacles": self.obstacle_count(),
            "occupied_voxels": self.occupied_count(),
            "occupancy_fraction": self.occupancy_fraction(),
        }


def _stage_obstacle_boxes(
    *,
    stage,
    obstacle_paths,
    inflation_radius,
    exclude_prefixes,
    exclude_keywords,
):
    from pxr import UsdGeom

    cache = UsdGeom.BBoxCache(
        0.0,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )

    if obstacle_paths is None:
        prims = [
            prim
            for prim in stage.Traverse()
            if UsdGeom.Boundable(prim)
            and not _path_is_excluded(str(prim.GetPath()), exclude_prefixes, exclude_keywords)
        ]
    else:
        prims = []
        for path in obstacle_paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid() and not _path_is_excluded(str(prim.GetPath()), exclude_prefixes, exclude_keywords):
                prims.append(prim)

    boxes = []
    inflation = np.full(3, float(inflation_radius), dtype=float)
    for prim in prims:
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bbox.IsEmpty():
            continue

        boxes.append(
            ObstacleBox(
                path=str(prim.GetPath()),
                minimum=np.array(bbox.GetMin(), dtype=float) - inflation,
                maximum=np.array(bbox.GetMax(), dtype=float) + inflation,
            )
        )

    return boxes


def _path_is_excluded(path, exclude_prefixes, exclude_keywords):
    lower_path = path.lower()
    return (
        any(path == prefix or path.startswith(f"{prefix}/") for prefix in exclude_prefixes)
        or any(keyword.lower() in lower_path for keyword in exclude_keywords)
    )
