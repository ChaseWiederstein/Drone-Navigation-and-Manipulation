import numpy as np


class TrajectoryDebugDraw:
    """Viewport-only trajectory debug lines.

    Uses Isaac Sim debug draw instead of USD geometry, so these lines are not
    physics objects, map obstacles, or normal scene prims.
    """

    PAST_COLOR = (0.0, 1.0, 0.0, 1.0)
    BELIEF_COLOR = (0.0, 0.25, 1.0, 1.0)
    PLAN_COLOR = (1.0, 0.0, 0.0, 1.0)
    KINO_COLOR = (0.1, 0.35, 1.0, 1.0)

    def __init__(self, enabled=True, line_width=4.0, max_past_points=2000):
        self.enabled = bool(enabled)
        self.line_width = float(line_width)
        self.max_past_points = int(max_past_points)
        self.past_points = []
        self.belief_points = []
        self.plan_points = []
        self.kino_points = []
        self._draw = None

        if not self.enabled:
            return

        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            self.enabled = False
            print(f"[trajectory_debug] disabled: {exc}")

    def add_actual_position(self, pos):
        if not self.enabled:
            return

        self.past_points.append(_point3(pos))
        if len(self.past_points) > self.max_past_points:
            self.past_points = self.past_points[-self.max_past_points :]

    def add_belief_position(self, pos):
        if not self.enabled:
            return

        self.belief_points.append(_point3(pos))
        if len(self.belief_points) > self.max_past_points:
            self.belief_points = self.belief_points[-self.max_past_points :]

    def set_plan_from_refs(self, refs):
        if not self.enabled:
            return

        self.plan_points = [_point3(ref[0][0:3]) for ref in refs]

    def set_kino_path(self, kino_path):
        self.kino_points = []

    def clear_plan(self):
        self.plan_points = []
        self.kino_points = []

    def clear_past(self):
        self.past_points = []
        self.belief_points = []

    def clear_all(self):
        self.clear_plan()
        self.clear_past()

    def draw(self):
        if not self.enabled or self._draw is None:
            return

        self._draw.clear_lines()
        starts = []
        ends = []
        colors = []
        widths = []

        _append_polyline(starts, ends, colors, widths, self.past_points, self.PAST_COLOR, self.line_width)
        _append_polyline(starts, ends, colors, widths, self.belief_points, self.BELIEF_COLOR, self.line_width * 0.75)
        _append_polyline(starts, ends, colors, widths, self.plan_points, self.PLAN_COLOR, self.line_width)
        _append_polyline(starts, ends, colors, widths, self.kino_points, self.KINO_COLOR, self.line_width * 1.5)

        if starts:
            self._draw.draw_lines(starts, ends, colors, widths)

        if self.kino_points:
            self._draw.draw_points(
                self.kino_points,
                [self.KINO_COLOR] * len(self.kino_points),
                [self.line_width * 2.5] * len(self.kino_points),
            )


def _point3(pos):
    pos = np.asarray(pos, dtype=float).reshape(3)
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _append_polyline(starts, ends, colors, widths, points, color, width):
    if len(points) < 2:
        return

    for start, end in zip(points[:-1], points[1:]):
        starts.append(start)
        ends.append(end)
        colors.append(color)
        widths.append(width)
