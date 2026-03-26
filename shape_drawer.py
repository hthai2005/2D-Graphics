"""
Interactive 2D Shape Drawer — GLFW + PyOpenGL (+ NumPy)
Orthographic projection, dynamic VBO, mouse placement, keyboard colors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Literal

import glfw
import tkinter as tk
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

W, H = 1024, 768

VERT_SRC = """
#version 330 core
layout (location = 0) in vec2 aPos;
uniform mat4 uMVP;
uniform vec3 uColor;
out vec3 vColor;
void main() {
    vColor = uColor;
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
"""


def ortho_2d_tl_origin(width: float, height: float) -> np.ndarray:
    """Row-major 4x4: screen (0,0)=top-left, y down -> NDC. Use with glUniformMatrix4fv(..., GL_TRUE, ...)."""
    w, h = float(width), float(height)
    return np.array(
        [
            [2.0 / w, 0.0, 0.0, -1.0],
            [0.0, -2.0 / h, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def compile_program() -> int:
    vs = shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER)
    fs = shaders.compileShader(FRAG_SRC, GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vs, fs)


def cursor_to_framebuffer(win, x: float, y: float) -> Tuple[float, float]:
    """Map window-space cursor to framebuffer pixels (handles HiDPI scaling)."""
    fw, fh = glfw.get_framebuffer_size(win)
    ww, wh = glfw.get_window_size(win)
    if ww <= 0 or wh <= 0:
        return x, y
    return x * fw / ww, y * fh / wh


def polygon_signed_area(verts: List[Tuple[float, float]]) -> float:
    if len(verts) < 3:
        return 0.0
    a = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def point_in_triangle(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> bool:
    px, py = p
    ax, ay = a
    bx, by = b
    cx, cy = c
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-9:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= 0.0 and v >= 0.0 and (u + v) <= 1.0


def triangulate_polygon(verts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ear clipping for simple polygons (supports concave, non-self-intersecting)."""
    if len(verts) < 3:
        return []
    pts = verts[:]
    if polygon_signed_area(pts) < 0.0:
        pts.reverse()
    idx = list(range(len(pts)))
    out: List[Tuple[float, float]] = []

    guard = 0
    while len(idx) > 2 and guard < 10000:
        guard += 1
        ear_found = False
        m = len(idx)
        for i in range(m):
            i_prev = idx[(i - 1) % m]
            i_curr = idx[i]
            i_next = idx[(i + 1) % m]
            a, b, c = pts[i_prev], pts[i_curr], pts[i_next]

            # Convex test (CCW orientation expected)
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-8:
                continue

            inside = False
            for j in idx:
                if j in (i_prev, i_curr, i_next):
                    continue
                if point_in_triangle(pts[j], a, b, c):
                    inside = True
                    break
            if inside:
                continue

            out.extend([a, b, c])
            del idx[i]
            ear_found = True
            break

        if not ear_found:
            # Fallback: return empty so caller can skip filled rendering.
            return []
    return out


PALETTE: List[Tuple[float, float, float]] = [
    (0.95, 0.45, 0.35),
    (0.35, 0.75, 0.55),
    (0.45, 0.55, 0.95),
    (0.95, 0.85, 0.35),
    (0.85, 0.45, 0.85),
    (0.45, 0.85, 0.95),
    (0.95, 0.55, 0.65),
    (0.55, 0.55, 0.60),
    (1.0, 1.0, 1.0),
]

DrawMode = Literal["polygon", "polyline", "triangles", "points"]
ToolKind = Literal["freeform", "triangle", "rectangle", "square", "circle", "pentagon", "star"]


@dataclass
class Shape:
    mode: DrawMode
    vertices: List[Tuple[float, float]]
    color: Tuple[float, float, float]
    filled: bool

def main() -> None:
    global W, H

    if not glfw.init():
        raise RuntimeError("glfw.init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(
        W,
        H,
        "2D Shape Drawer — Tool: Click-drag (Triangle/Circle/Rect/Pentagon/Star) | Freeform: LMB add | RMB del | MMB drag",
        None,
        None,
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    program = compile_program()
    u_mvp = glGetUniformLocation(program, "uMVP")
    u_color = glGetUniformLocation(program, "uColor")

    vbo = glGenBuffers(1)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    shapes: List[Shape] = []
    current: List[Tuple[float, float]] = []
    draw_mode: DrawMode = "polygon"
    tool_kind: ToolKind = "freeform"
    tool_dragging = False
    tool_start: Tuple[float, float] = (0.0, 0.0)
    color_idx = 0
    filled = True
    drag_idx: int | None = None
    hit_radius = 14.0

    # Mouse-based move mode (checkbox in UI)
    move_mouse_dragging = False
    move_mouse_target_idx: int | None = None
    moved_shape_idx: int | None = None
    move_mouse_last_pos: Tuple[float, float] = (0.0, 0.0)

    def shape_hit_index(fx: float, fy: float, pad: float = 10.0) -> int | None:
        """Return last-hit shape index by AABB test (fast + good enough for UI move)."""
        for i in range(len(shapes) - 1, -1, -1):
            verts = shapes[i].vertices
            if not verts:
                continue
            minx = min(v[0] for v in verts) - pad
            maxx = max(v[0] for v in verts) + pad
            miny = min(v[1] for v in verts) - pad
            maxy = max(v[1] for v in verts) + pad
            if minx <= fx <= maxx and miny <= fy <= maxy:
                return i
        return None

    def upload(verts: List[Tuple[float, float]]) -> int:
        """Upload verts to the shared VBO. Returns vertex count uploaded."""
        if verts:
            arr = np.array(verts, dtype=np.float32).flatten()
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)
            return len(verts)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        return 0

    def commit_shape() -> None:
        nonlocal current
        if not current:
            return
        r, g, b = PALETTE[color_idx]
        shapes.append(Shape(mode=draw_mode, vertices=current[:], color=(r, g, b), filled=filled))
        current = []

    def apply_scale_current(factor: float) -> None:
        if not current:
            return
        cx = sum(v[0] for v in current) / len(current)
        cy = sum(v[1] for v in current) / len(current)
        current[:] = [
            (float(cx + (vx - cx) * factor), float(cy + (vy - cy) * factor))
            for vx, vy in current
        ]

    def apply_scale_target(factor: float) -> None:
        verts = transform_target_vertices()
        if not verts:
            return
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        verts[:] = [
            (float(cx + (vx - cx) * factor), float(cy + (vy - cy) * factor))
            for vx, vy in verts
        ]

    def tool_vertices(kind: ToolKind, x0: float, y0: float, x1: float, y1: float) -> List[Tuple[float, float]]:
        """Create a preset shape from a drag bbox (x0,y0)->(x1,y1)."""
        dx = x1 - x0
        dy = y1 - y0
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        w = abs(dx)
        h = abs(dy)
        size = max(w, h)
        if kind == "rectangle":
            left, right = (x0, x1) if x0 <= x1 else (x1, x0)
            top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)
            return [
                (float(left), float(top)),
                (float(right), float(top)),
                (float(right), float(bottom)),
                (float(left), float(bottom)),
            ]
        if kind == "square":
            # Use the bigger drag axis to define the side length.
            side = size
            if side < 1.0:
                side = 1.0
            sx = 1.0 if dx >= 0 else -1.0
            sy = 1.0 if dy >= 0 else -1.0
            x2 = x0 + sx * side
            y2 = y0 + sy * side
            left, right = (x0, x2) if x0 <= x2 else (x2, x0)
            top, bottom = (y0, y2) if y0 <= y2 else (y2, y0)
            return [
                (float(left), float(top)),
                (float(right), float(top)),
                (float(right), float(bottom)),
                (float(left), float(bottom)),
            ]
        if kind == "triangle":
            left, right = (x0, x1) if x0 <= x1 else (x1, x0)
            top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)
            apex_x = (left + right) * 0.5
            # If user drags downwards -> apex on top. Drag upwards -> apex on bottom.
            apex_y = top if dy >= 0 else bottom
            base_y = bottom if dy >= 0 else top
            return [
                (float(apex_x), float(apex_y)),
                (float(left), float(base_y)),
                (float(right), float(base_y)),
            ]
        if kind == "circle":
            radius = max(1.0, size * 0.5)
            n = max(24, int(radius / 4))
            start_ang = -math.pi / 2.0
            pts: List[Tuple[float, float]] = []
            for i in range(n):
                a = start_ang + 2.0 * math.pi * i / n
                pts.append((float(cx + radius * math.cos(a)), float(cy + radius * math.sin(a))))
            return pts
        if kind == "pentagon":
            radius = max(1.0, size * 0.5)
            n = 5
            start_ang = -math.pi / 2.0
            pts = []
            for i in range(n):
                a = start_ang + 2.0 * math.pi * i / n
                pts.append((float(cx + radius * math.cos(a)), float(cy + radius * math.sin(a))))
            return pts
        if kind == "star":
            points = 5
            outer_r = max(1.0, size * 0.5)
            inner_r = outer_r * 0.45
            steps = points * 2
            start_ang = -math.pi / 2.0
            pts = []
            for i in range(steps):
                a = start_ang + 2.0 * math.pi * i / steps
                r = outer_r if i % 2 == 0 else inner_r
                pts.append((float(cx + r * math.cos(a)), float(cy + r * math.sin(a))))
            return pts
        return []

    def set_preset_polygon(sides: int, radius: float = 90.0) -> None:
        nonlocal current, draw_mode
        cx, cy = float(W) * 0.5, float(H) * 0.5
        pts: List[Tuple[float, float]] = []
        start_ang = -math.pi / 2.0
        for i in range(max(3, sides)):
            a = start_ang + (2.0 * math.pi * i / sides)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        current = pts
        draw_mode = "polygon"
        mode_var.set(draw_mode)

    def set_preset_rectangle(width: float = 180.0, height: float = 110.0) -> None:
        nonlocal current, draw_mode
        cx, cy = float(W) * 0.5, float(H) * 0.5
        hw, hh = width * 0.5, height * 0.5
        current = [
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
        ]
        draw_mode = "polygon"
        mode_var.set(draw_mode)

    def set_preset_star(points: int = 5, radius: float = 95.0) -> None:
        nonlocal current, draw_mode
        cx, cy = float(W) * 0.5, float(H) * 0.5
        pts: List[Tuple[float, float]] = []
        inner_r = radius * 0.45
        start_ang = -math.pi / 2.0
        steps = max(5, points) * 2
        for i in range(steps):
            a = start_ang + (2.0 * math.pi * i / steps)
            r = radius if i % 2 == 0 else inner_r
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        current = pts
        draw_mode = "polygon"
        mode_var.set(draw_mode)

    def rgb_to_hex(r: float, g: float, b: float) -> str:
        # Tkinter button background expects hex like "#RRGGBB"
        return "#{:02x}{:02x}{:02x}".format(int(max(0, min(1, r)) * 255), int(max(0, min(1, g)) * 255), int(max(0, min(1, b)) * 255))

    # --- Menu UI (Tkinter) ---
    ui_root = tk.Tk()
    ui_root.title("2D Shape Drawer Menu")
    ui_root.resizable(False, False)

    mode_var = tk.StringVar(value=draw_mode)
    filled_var = tk.BooleanVar(value=filled)
    selected_color_var = tk.StringVar(value=f"Selected color: {color_idx + 1}/{len(PALETTE)}")

    def set_mode(m: DrawMode) -> None:
        nonlocal draw_mode, tool_kind, tool_dragging
        draw_mode = m
        mode_var.set(m)
        tool_kind = "freeform"
        tool_dragging = False
        current.clear()

    def set_color(i: int) -> None:
        nonlocal color_idx
        color_idx = i
        selected_color_var.set(f"Selected color: {color_idx + 1}/{len(PALETTE)}")

    def set_filled(v: bool) -> None:
        nonlocal filled
        filled = v
        filled_var.set(v)

    def ui_commit() -> None:
        if tool_kind == "freeform":
            commit_shape()

    def ui_undo() -> None:
        # In tool mode, `current` is a preview; undo should remove last committed shape.
        if tool_kind == "freeform" and current:
            current.pop()
        elif shapes:
            shapes.pop()

    def ui_clear() -> None:
        shapes.clear()
        current.clear()

    def ui_scale_up() -> None:
        # Kept for compatibility if you re-enable scale buttons.
        apply_scale_target(1.1)

    def ui_scale_down() -> None:
        # Kept for compatibility if you re-enable scale buttons.
        apply_scale_target(0.9)

    def start_tool(kind: ToolKind) -> None:
        """Select a draw tool. User will click-drag on the OpenGL canvas to place a new shape."""
        nonlocal tool_kind, tool_dragging, draw_mode
        tool_kind = kind
        tool_dragging = False
        current.clear()
        draw_mode = "polygon"
        mode_var.set(draw_mode)

    transform_target_var = tk.StringVar(value="last")

    def transform_target_vertices() -> List[Tuple[float, float]] | None:
        nonlocal moved_shape_idx
        if moved_shape_idx is not None and 0 <= moved_shape_idx < len(shapes):
            return shapes[moved_shape_idx].vertices
        if transform_target_var.get() == "current":
            return current if current else None
        if shapes:
            return shapes[-1].vertices
        return None

    def move_target(dx: float, dy: float) -> None:
        verts = transform_target_vertices()
        if not verts:
            return
        verts[:] = [(float(x + dx), float(y + dy)) for x, y in verts]

    def rotate_target(deg: float) -> None:
        verts = transform_target_vertices()
        if not verts:
            return
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        out: List[Tuple[float, float]] = []
        for x, y in verts:
            rx, ry = x - cx, y - cy
            out.append((float(cx + rx * ca - ry * sa), float(cy + rx * sa + ry * ca)))
        verts[:] = out

    def ui_close() -> None:
        glfw.set_window_should_close(window, True)
        try:
            ui_root.destroy()
        except tk.TclError:
            pass

    ui_root.protocol("WM_DELETE_WINDOW", ui_close)

    color_frame = tk.LabelFrame(ui_root, text="Color")
    color_frame.pack(fill="x", padx=8, pady=6)
    tk.Label(color_frame, textvariable=selected_color_var).pack(anchor="w", padx=8, pady=(4, 0))

    palette_grid = tk.Frame(color_frame)
    palette_grid.pack(padx=8, pady=6)
    cols = 4
    for i, (r, g, b) in enumerate(PALETTE):
        hexv = rgb_to_hex(r, g, b)
        btn = tk.Button(
            palette_grid,
            bg=hexv,
            activebackground=hexv,
            width=4,
            height=2,
            command=lambda i=i: set_color(i),
        )
        row, col = divmod(i, cols)
        btn.grid(row=row, column=col, padx=2, pady=2)

    fill_frame = tk.LabelFrame(ui_root, text="Options")
    fill_frame.pack(fill="x", padx=8, pady=6)
    tk.Checkbutton(fill_frame, text="Filled (polygon/triangles)", variable=filled_var, command=lambda: set_filled(bool(filled_var.get()))).pack(anchor="w", padx=8, pady=6)

    preset_frame = tk.LabelFrame(ui_root, text="Tools (Click-drag trên màn hình)")
    preset_frame.pack(fill="x", padx=8, pady=6)
    tk.Button(preset_frame, text="Freeform", command=lambda: start_tool("freeform")).pack(side="left", expand=True, fill="x", padx=2, pady=6)
    tk.Button(preset_frame, text="Triangle", command=lambda: start_tool("triangle")).pack(side="left", expand=True, fill="x", padx=2, pady=6)
    tk.Button(preset_frame, text="Square", command=lambda: start_tool("square")).pack(side="left", expand=True, fill="x", padx=2, pady=6)
    tk.Button(preset_frame, text="Circle", command=lambda: start_tool("circle")).pack(side="left", expand=True, fill="x", padx=2, pady=6)
    tk.Button(preset_frame, text="Rectangle", command=lambda: start_tool("rectangle")).pack(side="left", expand=True, fill="x", padx=2, pady=6)
    tk.Button(preset_frame, text="Pentagon", command=lambda: start_tool("pentagon")).pack(side="left", expand=True, fill="x", padx=2, pady=6)
    tk.Button(preset_frame, text="Star", command=lambda: start_tool("star")).pack(side="left", expand=True, fill="x", padx=2, pady=6)

    # Scale buttons removed (scale via Ctrl+Wheel)

    transform_frame = tk.LabelFrame(ui_root, text="Move (mouse drag) & Rotate")
    transform_frame.pack(fill="x", padx=8, pady=6)

    move_mouse_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        transform_frame,
        text="Move bằng chuột (tick để kéo shape)",
        variable=move_mouse_var,
    ).pack(anchor="w", padx=8, pady=(6, 2))

    rot_row = tk.Frame(transform_frame)
    rot_row.pack(fill="x", padx=8, pady=(2, 6))
    tk.Button(rot_row, text="Rotate -15°", command=lambda: rotate_target(-15.0)).pack(side="left", expand=True, fill="x", padx=(0, 4))
    tk.Button(rot_row, text="Rotate +15°", command=lambda: rotate_target(15.0)).pack(side="left", expand=True, fill="x", padx=(4, 0))

    buttons_frame = tk.Frame(ui_root)
    buttons_frame.pack(fill="x", padx=8, pady=6)
    tk.Button(buttons_frame, text="Commit (Enter)", command=ui_commit).pack(side="left", expand=True, fill="x", padx=(0, 4))
    tk.Button(buttons_frame, text="Undo (Ctrl+Z)", command=ui_undo).pack(side="left", expand=True, fill="x", padx=(4, 0))

    tk.Button(ui_root, text="Clear (C)", command=ui_clear).pack(fill="x", padx=8, pady=(0, 8))

    def print_help() -> None:
        print(
            "\n".join(
                [
                    "",
                    "=== 2D Shape Drawer Controls ===",
                    "Mouse:",
                    "  Tools       => LMB click-drag to create new shape",
                    "  Freeform    => LMB add vertex (then Enter commit)",
                    "  RMB          delete nearest vertex (current)",
                    "  MMB drag     move a vertex (current)",
                    "  Wheel       cycle color",
                    "  Ctrl+Wheel  scale moved/last shape",
                    "  Move tick    (checkbox) => LMB drag to move shapes",
                    "",
                    "Keyboard:",
                    "  Q / E       Rotate moved/last shape",
                    "  Enter       Commit (only Freeform tool)",
                    "  F           Toggle filled (where applicable)",
                    "  M / N       Next / previous color",
                    "  1..9        Pick palette color",
                    "  Ctrl+Z      Undo (vertex in current, else last shape)",
                    "  C           Clear all shapes + current",
                    "  H           Show this help",
                    "  Esc         Quit",
                    "",
                ]
            )
        )

    def on_resize(_w, width: int, height: int) -> None:
        global W, H
        W, H = max(320, width), max(240, height)
        glViewport(0, 0, W, H)

    def on_mouse_button(win, button: int, action: int, mods: int) -> None:
        nonlocal drag_idx, tool_kind, tool_dragging, tool_start, draw_mode
        nonlocal move_mouse_dragging, move_mouse_target_idx, move_mouse_last_pos
        nonlocal moved_shape_idx
        if action != glfw.PRESS and action != glfw.RELEASE:
            return
        mx, my = glfw.get_cursor_pos(win)
        mx, my = cursor_to_framebuffer(win, mx, my)

        # If Move-by-mouse is enabled, we only move committed shapes.
        if move_mouse_var.get():
            # Cancel tool preview when starting to move.
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
                idx = shape_hit_index(mx, my)
                if idx is not None:
                    tool_dragging = False
                    drag_idx = None
                    current.clear()
                    move_mouse_dragging = True
                    move_mouse_target_idx = idx
                    move_mouse_last_pos = (float(mx), float(my))
                    moved_shape_idx = idx
                # Always block drawing while move mode is on.
                return
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
                move_mouse_dragging = False
                move_mouse_target_idx = None
                # Keep moved_shape_idx so Rotate will target the last moved shape.
                return
            return

        # Tool-based drawing (triangle/circle/rectangle/star/...)
        if tool_kind != "freeform":
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
                tool_dragging = True
                tool_start = (float(mx), float(my))
                draw_mode = "polygon"
                mode_var.set(draw_mode)
                current[:] = tool_vertices(tool_kind, mx, my, mx, my)
                return
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE and tool_dragging:
                tool_dragging = False
                verts = tool_vertices(
                    tool_kind,
                    tool_start[0],
                    tool_start[1],
                    mx,
                    my,
                )
                if verts:
                    r, g, b = PALETTE[color_idx]
                    shapes.append(
                        Shape(
                            mode="polygon",
                            vertices=verts,
                            color=(r, g, b),
                            filled=filled,
                        )
                    )
                current.clear()
                return
            # Ignore other mouse actions while a tool is active.
            return

        # Freeform vertex editing
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if drag_idx is None:
                current.append((float(mx), float(my)))
        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            if not current:
                return
            best_i = min(
                range(len(current)),
                key=lambda i: (current[i][0] - mx) ** 2 + (current[i][1] - my) ** 2,
            )
            if math.hypot(current[best_i][0] - mx, current[best_i][1] - my) <= hit_radius:
                current.pop(best_i)
        elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
            if not current:
                return
            best_i = min(
                range(len(current)),
                key=lambda i: (current[i][0] - mx) ** 2 + (current[i][1] - my) ** 2,
            )
            if math.hypot(current[best_i][0] - mx, current[best_i][1] - my) <= hit_radius:
                drag_idx = best_i
        elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
            drag_idx = None

    def on_cursor_pos(win, x: float, y: float) -> None:
        nonlocal tool_dragging, tool_kind, tool_start, draw_mode
        nonlocal move_mouse_dragging, move_mouse_target_idx, move_mouse_last_pos
        if move_mouse_dragging and move_mouse_target_idx is not None:
            fx, fy = cursor_to_framebuffer(win, x, y)
            dx = float(fx) - move_mouse_last_pos[0]
            dy = float(fy) - move_mouse_last_pos[1]
            if dx != 0.0 or dy != 0.0:
                verts = shapes[move_mouse_target_idx].vertices
                verts[:] = [(float(vx + dx), float(vy + dy)) for vx, vy in verts]
                move_mouse_last_pos = (float(fx), float(fy))
            return
        if tool_kind != "freeform" and tool_dragging:
            fx, fy = cursor_to_framebuffer(win, x, y)
            draw_mode = "polygon"
            mode_var.set(draw_mode)
            current[:] = tool_vertices(tool_kind, tool_start[0], tool_start[1], fx, fy)
            return
        if drag_idx is None or not current:
            return
        fx, fy = cursor_to_framebuffer(win, x, y)
        current[drag_idx] = (float(fx), float(fy))

    def on_scroll(_w, _dx: float, dy: float) -> None:
        nonlocal color_idx
        if move_mouse_var.get():
            return
        ctrl_down = (
            glfw.get_key(_w, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(_w, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )
        if ctrl_down:
            if dy > 0:
                apply_scale_target(1.08)
            elif dy < 0:
                apply_scale_target(0.92)
            return
        if dy > 0:
            color_idx = (color_idx + 1) % len(PALETTE)
        elif dy < 0:
            color_idx = (color_idx - 1) % len(PALETTE)
        selected_color_var.set(f"Selected color: {color_idx + 1}/{len(PALETTE)}")

    def on_key(_w, key: int, _scancode: int, action: int, mods: int) -> None:
        nonlocal filled, color_idx, draw_mode
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_w, True)
        elif key == glfw.KEY_C:
            shapes.clear()
            current.clear()
        elif key == glfw.KEY_M:
            # Cycle palette forward (in case mouse wheel doesn't work well)
            color_idx = (color_idx + 1) % len(PALETTE)
            selected_color_var.set(f"Selected color: {color_idx + 1}/{len(PALETTE)}")
        elif key == glfw.KEY_N:
            # Cycle palette backward
            color_idx = (color_idx - 1) % len(PALETTE)
            selected_color_var.set(f"Selected color: {color_idx + 1}/{len(PALETTE)}")
        elif key == glfw.KEY_P:
            draw_mode = "polygon"
            mode_var.set(draw_mode)
        elif key == glfw.KEY_L:
            draw_mode = "polyline"
            mode_var.set(draw_mode)
        elif key == glfw.KEY_T:
            draw_mode = "triangles"
            mode_var.set(draw_mode)
        elif key == glfw.KEY_O:
            draw_mode = "points"
            mode_var.set(draw_mode)
        elif key == glfw.KEY_ENTER or key == glfw.KEY_KP_ENTER:
            if tool_kind == "freeform":
                commit_shape()
        elif key == glfw.KEY_H:
            print_help()
        elif key == glfw.KEY_UP:
            move_target(0, -12)
        elif key == glfw.KEY_DOWN:
            move_target(0, 12)
        elif key == glfw.KEY_LEFT:
            move_target(-12, 0)
        elif key == glfw.KEY_RIGHT:
            move_target(12, 0)
        elif key == glfw.KEY_Q:
            rotate_target(-15.0)
        elif key == glfw.KEY_E:
            rotate_target(15.0)
        elif key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL):
            if current:
                current.pop()
            elif shapes:
                shapes.pop()
        elif key == glfw.KEY_F:
            filled = not filled
            filled_var.set(filled)
        elif glfw.KEY_1 <= key <= glfw.KEY_9:
            n = key - glfw.KEY_1
            if n < len(PALETTE):
                color_idx = n
                selected_color_var.set(f"Selected color: {color_idx + 1}/{len(PALETTE)}")

    glfw.set_framebuffer_size_callback(window, on_resize)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    glViewport(0, 0, fb_w, fb_h)
    W, H = fb_w, fb_h

    glClearColor(0.08, 0.09, 0.12, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw_shape(mode: DrawMode, verts: List[Tuple[float, float]], rgb: Tuple[float, float, float], fill: bool) -> None:
        r, g, b = rgb
        n = upload(verts)
        if n <= 0:
            return
        glBindVertexArray(vao)

        # Fill (where applicable)
        if mode == "polygon":
            if fill and n >= 3:
                tris = triangulate_polygon(verts)
                if tris:
                    tri_n = upload(tris)
                    glUniform3f(u_color, r, g, b)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    glDrawArrays(GL_TRIANGLES, 0, tri_n)
                    n = upload(verts)
                else:
                    # Fallback: triangle fan fill (works best for convex polygons).
                    glUniform3f(u_color, r, g, b)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    glDrawArrays(GL_TRIANGLE_FAN, 0, n)
            if n >= 2:
                er, eg, eb = (min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2)) if fill else (r, g, b)
                glUniform3f(u_color, er, eg, eb)
                glLineWidth(2.0)
                glDrawArrays(GL_LINE_LOOP, 0, n)
        elif mode == "polyline":
            if n >= 2:
                glUniform3f(u_color, r, g, b)
                glLineWidth(2.0)
                glDrawArrays(GL_LINE_STRIP, 0, n)
        elif mode == "triangles":
            tri_n = (n // 3) * 3
            if tri_n >= 3:
                if fill:
                    glUniform3f(u_color, r, g, b)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    glDrawArrays(GL_TRIANGLES, 0, tri_n)
                # outline each triangle
                er, eg, eb = (min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2)) if fill else (r, g, b)
                glUniform3f(u_color, er, eg, eb)
                glLineWidth(2.0)
                for i in range(0, tri_n, 3):
                    glDrawArrays(GL_LINE_LOOP, i, 3)
        elif mode == "points":
            pass

        # Points on top (all modes)
        glUniform3f(u_color, r, g, b)
        glPointSize(8.0)
        glDrawArrays(GL_POINTS, 0, n)
        glBindVertexArray(0)

    print_help()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        try:
            ui_root.update()
        except tk.TclError:
            glfw.set_window_should_close(window, True)
            break

        mvp = ortho_2d_tl_origin(float(W), float(H))

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(program)
        glUniformMatrix4fv(u_mvp, 1, GL_TRUE, mvp)

        # Draw committed shapes first
        for s in shapes:
            draw_shape(s.mode, s.vertices, s.color, s.filled)

        # Draw current (in-progress) shape last, using current palette
        cur_rgb = PALETTE[color_idx]
        draw_shape(draw_mode, current, cur_rgb, filled)

        glUseProgram(0)

        title = (
            f"2D Shape Drawer | mode={draw_mode} | current_verts={len(current)} | shapes={len(shapes)} | "
            f"color={color_idx + 1}/{len(PALETTE)} | {'filled' if filled else 'edge/points'}"
        )
        glfw.set_window_title(window, title)

        glfw.swap_buffers(window)

    glDeleteBuffers(1, [vbo])
    glDeleteVertexArrays(1, [vao])
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
