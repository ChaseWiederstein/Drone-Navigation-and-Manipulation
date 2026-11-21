

def create_scene_handles(
    *,
    trash_path,
    drone_body_path,
    rgb_camera_path,
    depth_camera_path,
    camera_resolution,
):
    from isaacsim.core.prims import RigidPrim
    from isaacsim.sensors.camera import Camera

    trash = RigidPrim(prim_paths_expr=trash_path, name="trash")
    drone_body = RigidPrim(prim_paths_expr=drone_body_path, name="drone_body")
    rgb_camera = Camera(
        prim_path=rgb_camera_path,
        name="rgb_camera",
        resolution=camera_resolution,
    )
    depth_camera = Camera(
        prim_path=depth_camera_path,
        name="depth_camera",
        resolution=camera_resolution,
    )
    return trash, drone_body, rgb_camera, depth_camera
