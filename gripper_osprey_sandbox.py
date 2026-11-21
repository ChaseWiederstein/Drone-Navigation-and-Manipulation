import math
from pathlib import Path

from isaacsim import SimulationApp


from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.utils.stage import create_new_stage


PHYSICS_DT = 1.0 / 60.0
OSPREY_MODEL_USD = Path(r"C:/robotics/UAM/osprey/new_model.usd")
IRIS_USD = Path(r"C:/robotics/UAM/Collected_litterRoad/SubUSDs/iris.usd")

PIVOT_SPACING_M = 0.10
IRIS_WORLD_POS = Gf.Vec3d(0.0, 0.0, 0.35)
GRIPPER_PIVOT_LOCAL_Y = 0
GRIPPER_PIVOT_LOCAL_Z = -0.0532
GRIPPER_VISUAL_CENTER_WORLD = Gf.Vec3d(0.0, 0.0, 0.2968)
GRIPPER_MOUNT_YAW_DEG = 90.0
FREEZE_FOR_ALIGNMENT = False
ANIMATE_FINGERS = True
ENABLE_GRIPPER_MESH_COLLISION = True
GRIPPER_COLLISION_APPROXIMATION = "convexHull"

LEFT_VISUAL_PRIMS = [
    "/World/drone/tn__098082_02_/tn__113108_00refsetMODEL_zKGfJ/tn__098036_01refsetMODEL_zKGfJ",
    "/World/drone/tn__098082_02_/tn__113108_00refsetMODEL_zKGfJ/tn__048215_09refsetMODEL_zKGfJ",
    "/World/drone/tn__098082_02_/tn__113108_00refsetMODEL_zKGfJ/tn__048215_09refsetMODEL_1_WNGXL",
]

RIGHT_VISUAL_PRIMS = [
    "/World/drone/tn__098082_02_/tn__113108_00refsetMODEL_zKGfJ/tn__098012_01refsetMODEL_zKGfJ",
    "/World/drone/tn__098082_02_/tn__113108_00refsetMODEL_zKGfJ/tn__048215_09refsetMODEL_2_WNGXL",
    "/World/drone/tn__098082_02_/tn__113108_00refsetMODEL_zKGfJ/tn__048215_09refsetMODEL_3_WNGXL",
]


def make_static_box(stage, path, size, translate, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def disable_physics_under(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    for prim in list(Usd.PrimRange(root)):
        type_name = prim.GetTypeName()
        if type_name.startswith("Physics") and "Joint" in type_name:
            prim.SetActive(False)
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)


def make_iris_base(stage):
    visual_root = UsdGeom.Xform.Define(stage, "/World/iris_visual")
    visual_root.AddTranslateOp().Set(IRIS_WORLD_POS)

    visual = stage.DefinePrim("/World/iris_visual/model", "Xform")
    visual.GetReferences().AddReference(str(IRIS_USD), Sdf.Path("/vehicle"))
    disable_physics_under(stage, "/World/iris_visual")

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(visual_root.GetPrim())
    rigid_api.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(visual_root.GetPrim())
    mass_api.CreateMassAttr(1.5)


def make_gripper_mount(stage, mount_world_pos=IRIS_WORLD_POS):
    mount = UsdGeom.Xform.Define(stage, "/World/gripper_mount")
    mount.AddTranslateOp().Set(Gf.Vec3d(*mount_world_pos))

    base = UsdGeom.Xform.Define(stage, "/World/gripper_mount/gripper_anchor")
    base.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    # Invisible anchor collider so the gripper has a clean kinematic joint body.
    anchor = UsdGeom.Cube.Define(stage, "/World/gripper_mount/gripper_anchor/hidden_anchor_collision")
    anchor.CreateSizeAttr(1.0)
    anchor.AddScaleOp().Set(Gf.Vec3f(0.20, 0.20, 0.08))
    anchor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(anchor.GetPrim())
    UsdGeom.Imageable(anchor.GetPrim()).MakeInvisible()

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    rigid_api.CreateKinematicEnabledAttr(False)
    mass_api = UsdPhysics.MassAPI.Apply(base.GetPrim())
    mass_api.CreateMassAttr(1.5)
    return mount.GetPrim()


def make_gripper_mount_joint(stage, drone_body_path="/World/iris_visual"):
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/gripper_mount/fixed_to_iris")
    joint.CreateBody0Rel().SetTargets([drone_body_path])
    joint.CreateBody1Rel().SetTargets(["/World/gripper_mount/gripper_anchor"])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    yaw_rad = math.radians(float(GRIPPER_MOUNT_YAW_DEG))
    joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return joint.GetPrim()


def rotate_gripper_mount(stage):
    mount = UsdGeom.Xformable(stage.GetPrimAtPath("/World/gripper_mount"))
    mount.AddRotateZOp().Set(float(GRIPPER_MOUNT_YAW_DEG))


def world_pivot(pivot_x):
    return Gf.Vec3d(
        float(pivot_x),
        float(GRIPPER_PIVOT_LOCAL_Y),
        float(GRIPPER_PIVOT_LOCAL_Z),
    )


def make_finger_body(stage, path, pivot_x, mass=0.12):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(world_pivot(pivot_x))

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid_api.CreateKinematicEnabledAttr(bool(FREEZE_FOR_ALIGNMENT))
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(float(mass))
    return body.GetPrim()


def enable_mesh_collision_under(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"MESH_COLLISION skipped missing root: {root_path}", flush=True)
        return 0

    stage.Load(root.GetPath())
    count = 0
    prim_types = {}
    for prim in Usd.PrimRange(root):
        prim.Load()
        prim_types[prim.GetTypeName()] = prim_types.get(prim.GetTypeName(), 0) + 1
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(GRIPPER_COLLISION_APPROXIMATION)
        count += 1

    print(
        f"MESH_COLLISION {root_path}: enabled {count} mesh prims "
        f"approximation={GRIPPER_COLLISION_APPROXIMATION} types={prim_types}",
        flush=True,
    )
    return count


def enable_collision_on_mesh(stage, mesh_path):
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        print(f"MESH_COLLISION missing mesh: {mesh_path}", flush=True)
        return False
    if not prim.IsA(UsdGeom.Mesh):
        print(f"MESH_COLLISION not a Mesh: {mesh_path} type={prim.GetTypeName()}", flush=True)
        return False

    try:
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(GRIPPER_COLLISION_APPROXIMATION)
    except Exception as exc:
        print(f"MESH_COLLISION failed {mesh_path}: {exc}", flush=True)
        return False
    print(
        f"MESH_COLLISION explicit {mesh_path}: "
        f"collision_api={prim.HasAPI(UsdPhysics.CollisionAPI)} "
        f"mesh_collision_api={prim.HasAPI(UsdPhysics.MeshCollisionAPI)} "
        f"approximation={GRIPPER_COLLISION_APPROXIMATION}",
        flush=True,
    )
    return True


def collision_prim_paths_under(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []

    paths = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            paths.append(prim.GetPath())
    return paths


def disable_collisions_between(stage, source_paths, target_paths, label):
    target_paths = [path for path in target_paths if stage.GetPrimAtPath(path).IsValid()]
    if not target_paths:
        print(f"COLLISION_FILTER {label}: no targets", flush=True)
        return 0

    count = 0
    for source_path in source_paths:
        source = stage.GetPrimAtPath(source_path)
        if not source.IsValid():
            continue
        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(source)
        filtered_pairs.CreateFilteredPairsRel().SetTargets(target_paths)
        count += 1

    print(f"COLLISION_FILTER {label}: sources={count} targets={len(target_paths)}", flush=True)
    return count


def filter_gripper_internal_collisions(stage, drone_body_path):
    left_colliders = collision_prim_paths_under(stage, "/World/gripper_mount/left_gripper_finger")
    right_colliders = collision_prim_paths_under(stage, "/World/gripper_mount/right_gripper_finger")
    anchor_colliders = collision_prim_paths_under(stage, "/World/gripper_mount/gripper_anchor")
    drone_colliders = collision_prim_paths_under(stage, drone_body_path)

    left_ignore = right_colliders + anchor_colliders + drone_colliders
    right_ignore = left_colliders + anchor_colliders + drone_colliders

    disable_collisions_between(stage, left_colliders, left_ignore, "left_gripper_vs_right_anchor_drone")
    disable_collisions_between(stage, right_colliders, right_ignore, "right_gripper_vs_left_anchor_drone")


def make_joint_marker(stage, path, local_pos):
    marker_root = UsdGeom.Xform.Define(stage, path)
    marker_root.AddTranslateOp().Set(Gf.Vec3d(*local_pos))

    sphere = UsdGeom.Sphere.Define(stage, f"{path}/pivot_sphere")
    sphere.CreateRadiusAttr(0.012)
    UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])

    axis_bar = UsdGeom.Cube.Define(stage, f"{path}/x_axis_bar")
    axis_bar.CreateSizeAttr(1.0)
    axis_bar.AddScaleOp().Set(Gf.Vec3f(0.09, 0.004, 0.004))
    UsdGeom.Gprim(axis_bar.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])


def _finger_mesh_transform(side):
    if side == "left":
        return Gf.Vec3d(-.01, 0, -0.2), Gf.Vec3f(180.0, 0.0, 180.0)
    return Gf.Vec3d(.01, 0, -0.2), Gf.Vec3f(180.0, 0.0, -180.0)


def add_osprey_visuals(stage, link_path, visual_prim_paths, side):
    visual_root = UsdGeom.Xform.Define(stage, f"{link_path}/osprey_visual")
    visual_translate, visual_rotation = _finger_mesh_transform(side)

    # Placement stays on the parent so changing visual rotation does not move
    # the gripper away from the finger pivot.
    visual_root.AddTranslateOp().Set(visual_translate)

    oriented_visual = UsdGeom.Xform.Define(stage, f"{link_path}/osprey_visual/oriented")
    oriented_visual.AddRotateXYZOp().Set(visual_rotation)
    # CAD is millimeter-scale. Apply 0.001 once in this visual branch only.
    oriented_visual.AddScaleOp().Set(Gf.Vec3f(0.001, 0.001, 0.001))

    for i, src_prim_path in enumerate(visual_prim_paths):
        prim = stage.DefinePrim(f"{link_path}/osprey_visual/oriented/part_{i}", "Xform")
        prim.GetReferences().AddReference(str(OSPREY_MODEL_USD), Sdf.Path(src_prim_path))


def add_osprey_collision_meshes(stage, link_path, visual_prim_paths, side):
    collision_root = UsdGeom.Xform.Define(stage, f"{link_path}/osprey_collision")
    collision_translate, collision_rotation = _finger_mesh_transform(side)
    visual_root = stage.GetPrimAtPath(f"{link_path}/osprey_visual")
    if visual_root.IsValid():
        translate_attr = visual_root.GetAttribute("xformOp:translate")
        if translate_attr.IsValid() and translate_attr.Get() is not None:
            collision_translate = translate_attr.Get()
    collision_root.AddTranslateOp().Set(collision_translate)

    oriented_collision = UsdGeom.Xform.Define(stage, f"{link_path}/osprey_collision/oriented")
    oriented_collision.AddRotateXYZOp().Set(collision_rotation)
    oriented_collision.AddScaleOp().Set(Gf.Vec3f(0.001, 0.001, 0.001))

    for i, src_prim_path in enumerate(visual_prim_paths):
        prim = stage.DefinePrim(f"{link_path}/osprey_collision/oriented/part_{i}", "Xform")
        prim.SetInstanceable(False)
        prim.GetReferences().AddReference(str(OSPREY_MODEL_USD), Sdf.Path(src_prim_path))

    UsdGeom.Imageable(collision_root.GetPrim()).MakeInvisible()
    enabled = 0
    for i in range(len(visual_prim_paths)):
        if enable_collision_on_mesh(stage, f"{link_path}/osprey_collision/oriented/part_{i}/Mesh"):
            enabled += 1
    if enabled == 0:
        enabled = enable_mesh_collision_under(stage, f"{link_path}/osprey_collision/oriented")
    return enabled


def make_finger_joint(stage, path, body1_path, pivot_x, lower_deg, upper_deg, start_deg):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateAxisAttr("Y")
    joint.CreateBody0Rel().SetTargets(["/World/gripper_mount/gripper_anchor"])
    joint.CreateBody1Rel().SetTargets([body1_path])

    # Body0 is the Iris. Body1 origin is exactly at the finger pivot.
    joint.CreateLocalPos0Attr(Gf.Vec3f(float(pivot_x), GRIPPER_PIVOT_LOCAL_Y, GRIPPER_PIVOT_LOCAL_Z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr(float(lower_deg))
    joint.CreateUpperLimitAttr(float(upper_deg))

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(40.0)
    drive.CreateDampingAttr(20.0)
    drive.CreateMaxForceAttr(15.0)
    drive.CreateTargetPositionAttr(float(start_deg))
    drive.CreateTargetVelocityAttr(0.0)
    return drive


def _vec3_tuple(v):
    return (round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6))


def _bbox_center(stage, paths):
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    combined_range = Gf.Range3d()
    found = False
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        bbox_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bbox_range.IsEmpty():
            continue
        if not found:
            combined_range = bbox_range
            found = True
        else:
            combined_range.UnionWith(bbox_range)
    if not found:
        return None
    return combined_range.GetMidpoint()


def center_gripper_visuals(stage, target_world=GRIPPER_VISUAL_CENTER_WORLD):
    visual_paths = [
        "/World/gripper_mount/left_gripper_finger/osprey_visual/oriented",
        "/World/gripper_mount/right_gripper_finger/osprey_visual/oriented",
    ]
    current_center = _bbox_center(stage, visual_paths)
    if current_center is None:
        print("CENTER_GRIPPER_VISUALS skipped: no gripper bbox found", flush=True)
        return

    target_world = Gf.Vec3d(*target_world)
    correction = target_world - current_center
    for path in [
        "/World/gripper_mount/left_gripper_finger/osprey_visual",
        "/World/gripper_mount/right_gripper_finger/osprey_visual",
    ]:
        prim = stage.GetPrimAtPath(path)
        translate_attr = prim.GetAttribute("xformOp:translate")
        old_value = translate_attr.Get()
        translate_attr.Set(Gf.Vec3d(old_value[0] + correction[0], old_value[1] + correction[1], old_value[2] + correction[2]))

    print(
        "CENTER_GRIPPER_VISUALS "
        f"current={_vec3_tuple(current_center)} "
        f"target={_vec3_tuple(target_world)} "
        f"correction={_vec3_tuple(correction)}",
        flush=True,
    )


def attach_osprey_gripper(stage, drone_body_path, mount_world_pos=None, animate=False):
    if not OSPREY_MODEL_USD.exists():
        raise FileNotFoundError(f"Missing OSPREY model: {OSPREY_MODEL_USD}")

    if mount_world_pos is None:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        drone_prim = stage.GetPrimAtPath(drone_body_path)
        if not drone_prim.IsValid():
            raise ValueError(f"Missing drone body prim: {drone_body_path}")
        mount_world_pos = cache.GetLocalToWorldTransform(drone_prim).ExtractTranslation()
    else:
        mount_world_pos = Gf.Vec3d(float(mount_world_pos[0]), float(mount_world_pos[1]), float(mount_world_pos[2]))

    if stage.GetPrimAtPath("/World/gripper_mount").IsValid():
        stage.RemovePrim("/World/gripper_mount")

    make_gripper_mount(stage, mount_world_pos=mount_world_pos)

    left_pivot_x = -PIVOT_SPACING_M / 2.0
    right_pivot_x = PIVOT_SPACING_M / 2.0

    make_joint_marker(stage, "/World/gripper_mount/joint_markers/left", (left_pivot_x, GRIPPER_PIVOT_LOCAL_Y, GRIPPER_PIVOT_LOCAL_Z))
    make_joint_marker(stage, "/World/gripper_mount/joint_markers/right", (right_pivot_x, GRIPPER_PIVOT_LOCAL_Y, GRIPPER_PIVOT_LOCAL_Z))

    make_finger_body(stage, "/World/gripper_mount/left_gripper_finger", left_pivot_x)
    make_finger_body(stage, "/World/gripper_mount/right_gripper_finger", right_pivot_x)

    add_osprey_visuals(stage, "/World/gripper_mount/left_gripper_finger", RIGHT_VISUAL_PRIMS, side="right")
    add_osprey_visuals(stage, "/World/gripper_mount/right_gripper_finger", LEFT_VISUAL_PRIMS, side="left")

    gripper_center_world = mount_world_pos + Gf.Vec3d(0.0, 0.0, GRIPPER_PIVOT_LOCAL_Z)
    center_gripper_visuals(stage, target_world=gripper_center_world)

    if ENABLE_GRIPPER_MESH_COLLISION:
        add_osprey_collision_meshes(stage, "/World/gripper_mount/left_gripper_finger", RIGHT_VISUAL_PRIMS, side="right")
        add_osprey_collision_meshes(stage, "/World/gripper_mount/right_gripper_finger", LEFT_VISUAL_PRIMS, side="left")
        filter_gripper_internal_collisions(stage, drone_body_path)

    rotate_gripper_mount(stage)
    make_gripper_mount_joint(stage, drone_body_path=drone_body_path)

    left_drive = make_finger_joint(
        stage,
        "/World/gripper_mount/left_finger_pivot",
        "/World/gripper_mount/left_gripper_finger",
        left_pivot_x,
        lower_deg=-90.0,
        upper_deg=0.0,
        start_deg=0,
    )
    right_drive = make_finger_joint(
        stage,
        "/World/gripper_mount/right_finger_pivot",
        "/World/gripper_mount/right_gripper_finger",
        right_pivot_x,
        lower_deg=0.0,
        upper_deg=90.0,
        start_deg=0,
    )

    print(
        f"[gripper] attached to {drone_body_path} at mount={_vec3_tuple(mount_world_pos)} "
        f"animate={bool(animate)}",
        flush=True,
    )
    return {"left_drive": left_drive, "right_drive": right_drive, "animate": bool(animate)}


def print_side_check(stage, label):
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )

    pairs = [
        (
            "left",
            "/World/gripper_mount/left_gripper_finger",
            "/World/gripper_mount/left_gripper_finger/osprey_visual/oriented",
        ),
        (
            "right",
            "/World/gripper_mount/right_gripper_finger",
            "/World/gripper_mount/right_gripper_finger/osprey_visual/oriented",
        ),
    ]

    print(f"SIDE_CHECK {label}", flush=True)
    for name, joint_body_path, visual_path in pairs:
        joint_body = stage.GetPrimAtPath(joint_body_path)
        visual = stage.GetPrimAtPath(visual_path)
        if not joint_body.IsValid() or not visual.IsValid():
            print(f"  {name}: missing prim(s)", flush=True)
            continue

        joint_pos = xform_cache.GetLocalToWorldTransform(joint_body).ExtractTranslation()
        bbox_range = bbox_cache.ComputeWorldBound(visual).ComputeAlignedRange()
        if bbox_range.IsEmpty():
            print(f"  {name}: joint={_vec3_tuple(joint_pos)} visual_bbox=EMPTY", flush=True)
        else:
            center = bbox_range.GetMidpoint()
            print(
                f"  {name}: joint={_vec3_tuple(joint_pos)} "
                f"visual_center={_vec3_tuple(center)} "
                f"delta_visual_minus_joint={_vec3_tuple(center - joint_pos)}",
                flush=True,
            )


def main():
    simulation_app = SimulationApp({"headless": False})

    if not OSPREY_MODEL_USD.exists():
        raise FileNotFoundError(f"Missing OSPREY model: {OSPREY_MODEL_USD}")
    if not IRIS_USD.exists():
        raise FileNotFoundError(f"Missing Iris model: {IRIS_USD}")

    create_new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
    stage = world.stage

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    light = UsdLux.DistantLight.Define(stage, "/World/sun")
    light.CreateIntensityAttr(5000.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))

    make_static_box(stage, "/World/floor", (3.0, 3.0, 0.04), (0.0, 0.0, -0.02), (0.25, 0.25, 0.25))
    make_iris_base(stage)
    make_gripper_mount(stage)
    make_gripper_mount_joint(stage)

    left_pivot_x = -PIVOT_SPACING_M / 2.0
    right_pivot_x = PIVOT_SPACING_M / 2.0

    make_joint_marker(stage, "/World/gripper_mount/joint_markers/left", (left_pivot_x, GRIPPER_PIVOT_LOCAL_Y, GRIPPER_PIVOT_LOCAL_Z))
    make_joint_marker(stage, "/World/gripper_mount/joint_markers/right", (right_pivot_x, GRIPPER_PIVOT_LOCAL_Y, GRIPPER_PIVOT_LOCAL_Z))

    make_finger_body(stage, "/World/gripper_mount/left_gripper_finger", left_pivot_x)
    make_finger_body(stage, "/World/gripper_mount/right_gripper_finger", right_pivot_x)

    add_osprey_visuals(stage, "/World/gripper_mount/left_gripper_finger", RIGHT_VISUAL_PRIMS, side="right")
    add_osprey_visuals(stage, "/World/gripper_mount/right_gripper_finger", LEFT_VISUAL_PRIMS, side="left")
    center_gripper_visuals(stage)
    if ENABLE_GRIPPER_MESH_COLLISION:
        add_osprey_collision_meshes(stage, "/World/gripper_mount/left_gripper_finger", RIGHT_VISUAL_PRIMS, side="right")
        add_osprey_collision_meshes(stage, "/World/gripper_mount/right_gripper_finger", LEFT_VISUAL_PRIMS, side="left")
        filter_gripper_internal_collisions(stage, "/World/iris_visual")
    rotate_gripper_mount(stage)

    left_drive = make_finger_joint(
        stage,
        "/World/gripper_mount/left_finger_pivot",
        "/World/gripper_mount/left_gripper_finger",
        left_pivot_x,
        lower_deg=-90.0,
        upper_deg=0.0,
        start_deg=0,
    )
    right_drive = make_finger_joint(
        stage,
        "/World/gripper_mount/right_finger_pivot",
        "/World/gripper_mount/right_gripper_finger",
        right_pivot_x,
        lower_deg=0.0,
        upper_deg=90.0,
        start_deg=0,
    )

    world.reset()
    print_side_check(stage, "after_reset")

    step = 0
    while simulation_app.is_running():
        if ANIMATE_FINGERS:
            t = step * PHYSICS_DT
            close = 0.5 + 0.5 * math.sin(0.75 * t)
            left_target = -90.0 * close
            right_target = 90.0 * close
            left_drive.GetTargetPositionAttr().Set(float(left_target))
            right_drive.GetTargetPositionAttr().Set(float(right_target))
        world.step(render=True)
        if step in (0, 5, 30):
            print_side_check(stage, f"after_step_{step}")
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
