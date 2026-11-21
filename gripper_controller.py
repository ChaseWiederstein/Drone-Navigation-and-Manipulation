class GripperController:
    """Commands the OSPREY finger joint drives in degrees."""

    def __init__(
        self,
        left_drive,
        right_drive,
        *,
        open_left_deg=0.0,
        open_right_deg=0.0,
        closed_left_deg=-90.0,
        closed_right_deg=90.0,
    ):
        self.left_drive = left_drive
        self.right_drive = right_drive
        self.open_left_deg = float(open_left_deg)
        self.open_right_deg = float(open_right_deg)
        self.closed_left_deg = float(closed_left_deg)
        self.closed_right_deg = float(closed_right_deg)
        self.left_target_deg = self.open_left_deg
        self.right_target_deg = self.open_right_deg

    @classmethod
    def from_attach_result(cls, attach_result, **kwargs):
        return cls(
            attach_result["left_drive"],
            attach_result["right_drive"],
            **kwargs,
        )

    def open(self):
        self.set_angles(self.open_left_deg, self.open_right_deg)

    def close(self):
        self.set_angles(self.closed_left_deg, self.closed_right_deg)

    def close_fraction(self, fraction):
        fraction = max(0.0, min(1.0, float(fraction)))
        left = self.open_left_deg + fraction * (self.closed_left_deg - self.open_left_deg)
        right = self.open_right_deg + fraction * (self.closed_right_deg - self.open_right_deg)
        self.set_angles(left, right)

    def hold(self):
        self.set_angles(self.left_target_deg, self.right_target_deg)

    def set_angles(self, left_deg, right_deg):
        self.left_target_deg = float(left_deg)
        self.right_target_deg = float(right_deg)
        self.left_drive.GetTargetPositionAttr().Set(self.left_target_deg)
        self.right_drive.GetTargetPositionAttr().Set(self.right_target_deg)

    def set_strength(self, *, stiffness=None, damping=None, max_force=None):
        if stiffness is not None:
            self.left_drive.GetStiffnessAttr().Set(float(stiffness))
            self.right_drive.GetStiffnessAttr().Set(float(stiffness))
        if damping is not None:
            self.left_drive.GetDampingAttr().Set(float(damping))
            self.right_drive.GetDampingAttr().Set(float(damping))
        if max_force is not None:
            self.left_drive.GetMaxForceAttr().Set(float(max_force))
            self.right_drive.GetMaxForceAttr().Set(float(max_force))
