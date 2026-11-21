from pxr import Sdf, Usd, UsdPhysics, UsdShade


def apply_high_friction_to_colliders(
    stage,
    root_paths,
    *,
    material_path="/World/PhysicsMaterials/high_friction_grasp",
    static_friction=2.0,
    dynamic_friction=1.5,
    restitution=0.0,
):
    material = _get_or_create_physics_material(
        stage,
        material_path,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )
    bound = 0
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            print(f"FRICTION_MATERIAL missing root {root_path}", flush=True)
            continue

        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )
            bound += 1

    print(
        "FRICTION_MATERIAL "
        f"path={material_path} static={static_friction} dynamic={dynamic_friction} "
        f"restitution={restitution} bound_colliders={bound}",
        flush=True,
    )
    return bound


def _get_or_create_physics_material(
    stage,
    material_path,
    *,
    static_friction,
    dynamic_friction,
    restitution,
):
    parent_path = Sdf.Path(material_path).GetParentPath()
    if parent_path != Sdf.Path.absoluteRootPath:
        stage.DefinePrim(parent_path, "Scope")

    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(float(static_friction))
    physics_material.CreateDynamicFrictionAttr(float(dynamic_friction))
    physics_material.CreateRestitutionAttr(float(restitution))
    return material
