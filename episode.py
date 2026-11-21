import numpy as np


TRASH_X_RANGE = (3.0, 6.0)
TRASH_Y_RANGE = (1.0, 8.0)
TRASH_Z = 0.1


def sample_trash_position(rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return np.array(
        [
            rng.uniform(*TRASH_X_RANGE),
            rng.uniform(*TRASH_Y_RANGE),
            TRASH_Z,
        ],
        dtype=float,
    )


def randomize_trash_position(stage, trash_path, rng=None):
    from pxr import Gf, UsdGeom

    position = sample_trash_position(rng)

    prim = stage.GetPrimAtPath(trash_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Trash prim not found: {trash_path}")

    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break

    if translate_op is None:
        translate_op = xform.AddTranslateOp()

    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    return position
