from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Literal

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from pyopengltk import OpenGLFrame

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

# Fallback for Tk/WGL contexts that reject 330 core shaders
VERT_SRC_120 = """
#version 120
attribute vec2 aPos;
uniform mat4 uMVP;
uniform vec3 uColor;
varying vec3 vColor;
void main() {
    vColor = uColor;
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
}
"""
FRAG_SRC_120 = """
#version 120
varying vec3 vColor;
void main() {
    gl_FragColor = vec4(vColor, 1.0);
}
"""


def ortho_2d_tl_origin(width: float, height: float) -> np.ndarray:
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
    try:
        vs = shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER)
        fs = shaders.compileShader(FRAG_SRC, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vs, fs)
    except Exception:
        vs = shaders.compileShader(VERT_SRC_120, GL_VERTEX_SHADER)
        fs = shaders.compileShader(FRAG_SRC_120, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vs, fs)


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

    # Layout / colors aligned with "2Interactive 2D Shape Drawer" (editor_panels toolbar + properties + status)
    TOOLBAR_WIDTH = 118
    PROPERTIES_WIDTH = 240
    STATUS_BAR_HEIGHT = 24
    PANEL_BG = "#ffffff"
    PANEL_BORDER = "#d8dce6"
    BUTTON_BG = "#eef1f6"
    BUTTON_ACTIVE = "#c8d4f5"
    TEXT_COLOR = "#1c1c28"
    SUBTEXT_COLOR = "#5a6170"
    PALETTE_WELL_BG = "#e8eaef"

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

    move_mouse_dragging = False
    move_mouse_target_idx: int | None = None
    move_mouse_drag_free_current = False
    moved_shape_idx: int | None = None
    move_mouse_last_pos: Tuple[float, float] = (0.0, 0.0)

    root = tk.Tk()
    root.title("2D Shape Drawer")
    root.minsize(TOOLBAR_WIDTH + PROPERTIES_WIDTH + 480, 400)
    root.configure(bg=PANEL_BG)

    mode_var = tk.StringVar(value=draw_mode)
    filled_var = tk.BooleanVar(value=filled)
    selected_color_var = tk.StringVar(value=f"Color {color_idx + 1}/{len(PALETTE)}")
    status_var = tk.StringVar(value="")
    move_mouse_var = tk.BooleanVar(value=False)
    mouse_xy: List[float] = [0.0, 0.0]
    hover_shape_idx: int | None = None

    tool_bar_buttons: dict[ToolKind, tk.Button] = {}
    _canvas_singleton: list = []

    def shape_hit_index(fx: float, fy: float, pad: float = 10.0) -> int | None:
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

    def current_bbox_hit(fx: float, fy: float, pad: float = 10.0) -> bool:
        if not current:
            return False
        minx = min(v[0] for v in current) - pad
        maxx = max(v[0] for v in current) + pad
        miny = min(v[1] for v in current) - pad
        maxy = max(v[1] for v in current) + pad
        return minx <= fx <= maxx and miny <= fy <= maxy

    def upload(verts: List[Tuple[float, float]]) -> int:
        vb = _canvas_singleton[0]._vbo
        if verts:
            arr = np.array(verts, dtype=np.float32).flatten()
            glBindBuffer(GL_ARRAY_BUFFER, vb)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)
            return len(verts)
        glBindBuffer(GL_ARRAY_BUFFER, vb)
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
        return "#{:02x}{:02x}{:02x}".format(int(max(0, min(1, r)) * 255), int(max(0, min(1, g)) * 255), int(max(0, min(1, b)) * 255))

    def set_mode(m: DrawMode) -> None:
        nonlocal draw_mode, tool_kind, tool_dragging
        draw_mode = m
        mode_var.set(m)
        tool_kind = "freeform"
        tool_dragging = False
        current.clear()
        refresh_toolbar()
        sync_canvas()

    def set_color(i: int) -> None:
        nonlocal color_idx
        color_idx = i
        if palette_affects_shape_under_cursor():
            hi = hover_shape_idx
            if hi is not None:
                shapes[hi].color = PALETTE[i]
                selected_color_var.set(f"Shape #{hi + 1} — màu {i + 1}/{len(PALETTE)}")
            else:
                selected_color_var.set(f"Color {color_idx + 1}/{len(PALETTE)}")
        else:
            selected_color_var.set(f"Color {color_idx + 1}/{len(PALETTE)}")
        sync_canvas()

    def cycle_color_hover_or_brush(delta: int) -> None:
        """M/N: giống lăn chuột — trỏ shape (và không đang vẽ/kéo xung đột) thì đổi màu shape + màu nét."""
        nonlocal color_idx
        if palette_affects_shape_under_cursor():
            hi = hover_shape_idx
            if hi is not None:
                pi = palette_index_for_rgb(shapes[hi].color)
                pi = (pi + delta) % len(PALETTE)
                shapes[hi].color = PALETTE[pi]
                color_idx = pi
                selected_color_var.set(f"Shape #{hi + 1} — màu {pi + 1}/{len(PALETTE)}")
        else:
            color_idx = (color_idx + delta) % len(PALETTE)
            selected_color_var.set(f"Color {color_idx + 1}/{len(PALETTE)}")

    def set_filled(v: bool) -> None:
        nonlocal filled
        filled = v
        filled_var.set(v)
        sync_canvas()

    def ui_commit() -> None:
        if tool_kind == "freeform":
            commit_shape()
        sync_canvas()

    def ui_undo() -> None:
        nonlocal hover_shape_idx
        if tool_kind == "freeform" and current:
            current.pop()
        elif shapes:
            shapes.pop()
        hover_shape_idx = shape_hit_index(mouse_xy[0], mouse_xy[1])
        sync_canvas()

    def ui_clear() -> None:
        nonlocal hover_shape_idx
        shapes.clear()
        current.clear()
        hover_shape_idx = None
        sync_canvas()

    def ui_scale_up() -> None:
        apply_scale_target(1.1)
        sync_canvas()

    def ui_scale_down() -> None:
        apply_scale_target(0.9)
        sync_canvas()

    def refresh_toolbar() -> None:
        for k, btn in tool_bar_buttons.items():
            btn.configure(bg=BUTTON_ACTIVE if tool_kind == k else BUTTON_BG)

    def start_tool(kind: ToolKind) -> None:
        nonlocal tool_kind, tool_dragging, draw_mode
        tool_kind = kind
        tool_dragging = False
        current.clear()
        draw_mode = "polygon"
        mode_var.set(draw_mode)
        refresh_toolbar()
        sync_canvas()

    def transform_target_vertices() -> List[Tuple[float, float]] | None:
        nonlocal moved_shape_idx
        if moved_shape_idx is not None and 0 <= moved_shape_idx < len(shapes):
            return shapes[moved_shape_idx].vertices
        if shapes:
            return shapes[-1].vertices
        return None

    def palette_index_for_rgb(rgb: Tuple[float, float, float]) -> int:
        r, g, b = rgb
        best_i = 0
        best_d = 1e9
        for i, p in enumerate(PALETTE):
            d = (p[0] - r) ** 2 + (p[1] - g) ** 2 + (p[2] - b) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def palette_affects_shape_under_cursor() -> bool:
        """Trỏ vào shape + không đang kéo preset / không đang vẽ freeform có đỉnh → bảng màu tô shape đó (và vẫn cập nhật màu nét)."""
        if tool_dragging:
            return False
        if tool_kind == "freeform" and current:
            return False
        hi = hover_shape_idx
        return hi is not None and 0 <= hi < len(shapes)

    def move_target(dx: float, dy: float) -> None:
        verts = transform_target_vertices()
        if not verts:
            return
        verts[:] = [(float(x + dx), float(y + dy)) for x, y in verts]
        sync_canvas()

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
        sync_canvas()

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
                    "  Wheel       trỏ shape → màu shape; không → màu nét (Ctrl+Wheel scale)",
                    "  Trỏ shape    => bảng / lăn / M N đổi màu shape (không cần Move; freeform đang vẽ chỉ màu nét)",
                    "  Move mode    => LMB kéo shape hoặc poly freeform (trong bbox)",
                    "",
                    "Keyboard:",
                    "  Q / E       Rotate moved/last shape",
                    "  Enter       Commit (only Freeform tool)",
                    "  F           Toggle filled (where applicable)",
                    "  M / N       Next/prev color (trỏ shape: shape đó; không: màu vẽ)",
                    "  1..9        Pick palette color",
                    "  Ctrl+Z      Undo (vertex in current, else last shape)",
                    "  C           Clear all shapes + current",
                    "  H           Show this help",
                    "  Esc         Quit",
                    "",
                ]
            )
        )

    def draw_shape(mode: DrawMode, verts: List[Tuple[float, float]], rgb: Tuple[float, float, float], fill: bool) -> None:
        cv = _canvas_singleton[0]
        u_c = cv._u_color
        vao_id = cv._vao
        r, g, b = rgb
        n = upload(verts)
        if n <= 0:
            return
        glBindVertexArray(vao_id)

        if mode == "polygon":
            if fill and n >= 3:
                tris = triangulate_polygon(verts)
                if tris:
                    tri_n = upload(tris)
                    glUniform3f(u_c, r, g, b)
                    glDrawArrays(GL_TRIANGLES, 0, tri_n)
                    n = upload(verts)
                else:
                    glUniform3f(u_c, r, g, b)
                    glDrawArrays(GL_TRIANGLE_FAN, 0, n)
            if n >= 2:
                er, eg, eb = (min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2)) if fill else (r, g, b)
                glUniform3f(u_c, er, eg, eb)
                glLineWidth(2.0)
                glDrawArrays(GL_LINE_LOOP, 0, n)
        elif mode == "polyline":
            if n >= 2:
                glUniform3f(u_c, r, g, b)
                glLineWidth(2.0)
                glDrawArrays(GL_LINE_STRIP, 0, n)
        elif mode == "triangles":
            tri_n = (n // 3) * 3
            if tri_n >= 3:
                if fill:
                    glUniform3f(u_c, r, g, b)
                    glDrawArrays(GL_TRIANGLES, 0, tri_n)
                er, eg, eb = (min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2)) if fill else (r, g, b)
                glUniform3f(u_c, er, eg, eb)
                glLineWidth(2.0)
                for i in range(0, tri_n, 3):
                    glDrawArrays(GL_LINE_LOOP, i, 3)
        elif mode == "points":
            pass

        glUniform3f(u_c, r, g, b)
        glPointSize(8.0)
        glDrawArrays(GL_POINTS, 0, n)
        glBindVertexArray(0)

    class ShapeCanvas(OpenGLFrame):
        def tkResize(self, evt: tk.Event) -> None:
            global W, H
            self.width, self.height = evt.width, evt.height
            if self.winfo_ismapped():
                self.tkMakeCurrent()
                glViewport(0, 0, max(1, self.width), max(1, self.height))
                W, H = max(320, self.width), max(240, self.height)

        def initgl(self) -> None:
            self.tkMakeCurrent()
            w = max(1, self.winfo_width())
            h = max(1, self.winfo_height())
            glViewport(0, 0, w, h)
            global W, H
            W, H = max(320, w), max(240, h)
            self._program = compile_program()
            self._u_mvp = glGetUniformLocation(self._program, "uMVP")
            self._u_color = glGetUniformLocation(self._program, "uColor")
            self._vbo = glGenBuffers(1)
            self._vao = glGenVertexArrays(1)
            glBindVertexArray(self._vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)
            glBindVertexArray(0)
            glClearColor(0.08, 0.09, 0.12, 1.0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.animate = 16

        def redraw(self) -> None:
            self.tkMakeCurrent()
            mvp = ortho_2d_tl_origin(float(W), float(H))
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self._program)
            glUniformMatrix4fv(self._u_mvp, 1, GL_TRUE, mvp)
            for s in shapes:
                draw_shape(s.mode, s.vertices, s.color, s.filled)
            cur_rgb = PALETTE[color_idx]
            draw_shape(draw_mode, current, cur_rgb, filled)
            glUseProgram(0)
            tool_lbl = tool_kind if tool_kind == "freeform" else f"{tool_kind}"
            status_var.set(
                f"Tool: {tool_lbl} | mode={draw_mode} | verts: {len(current)} | shapes: {len(shapes)} | "
                f"{selected_color_var.get()} | mouse ({mouse_xy[0]:.0f}, {mouse_xy[1]:.0f})"
            )
            root.title(f"2D Shape Drawer — {tool_lbl} | {draw_mode} | {len(shapes)} shapes")

    body = tk.Frame(root, bg=PANEL_BG)
    body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    left = tk.Frame(body, width=TOOLBAR_WIDTH, bg=PANEL_BG, highlightthickness=1, highlightbackground=PANEL_BORDER)
    left.pack(side=tk.LEFT, fill=tk.Y)
    left.pack_propagate(False)

    tool_font = tkfont.Font(family="Segoe UI", size=9)
    tool_hint = tkfont.Font(family="Segoe UI", size=7)
    tool_specs: List[Tuple[ToolKind, str, str]] = [
        ("freeform", "Freeform", "click từng đỉnh"),
        ("triangle", "Triangle", "kéo LMB"),
        ("square", "Square", "kéo LMB"),
        ("circle", "Circle", "kéo LMB"),
        ("rectangle", "Rectangle", "kéo LMB"),
        ("pentagon", "Pentagon", "kéo LMB"),
        ("star", "Star", "kéo LMB"),
    ]

    tk.Label(left, text="Công cụ", fg=TEXT_COLOR, bg=PANEL_BG, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 4))
    tools_box = tk.Frame(left, bg=PANEL_BG)
    tools_box.pack(fill=tk.X, padx=6, pady=(0, 6))
    for kind, lab, hint in tool_specs:
        bf = tk.Frame(tools_box, bg=PANEL_BG)
        bf.pack(fill=tk.X, pady=3)
        b = tk.Button(
            bf,
            text=lab,
            font=tool_font,
            fg=TEXT_COLOR,
            bg=BUTTON_BG,
            activebackground=BUTTON_ACTIVE,
            activeforeground=TEXT_COLOR,
            relief=tk.FLAT,
            anchor="w",
            padx=6,
            pady=4,
            command=lambda k=kind: start_tool(k),
        )
        b.pack(fill=tk.X)
        tk.Label(bf, text=hint, fg=SUBTEXT_COLOR, bg=PANEL_BG, font=tool_hint).pack(anchor="w", padx=4)
        tool_bar_buttons[kind] = b

    tk.Label(left, text="Màu", fg=TEXT_COLOR, bg=PANEL_BG, font=("Segoe UI", 10, "bold")).pack(
        anchor="w", padx=8, pady=(8, 4)
    )
    pal_well = tk.Frame(left, bg=PALETTE_WELL_BG, highlightthickness=1, highlightbackground=PANEL_BORDER)
    pal_well.pack(fill=tk.X, padx=8, pady=(0, 8))
    pal_grid = tk.Frame(pal_well, bg=PALETTE_WELL_BG)
    pal_grid.pack(fill=tk.X, padx=6, pady=6)
    cols = 3
    for idx, (pr, pg, pb) in enumerate(PALETTE):
        row, col = divmod(idx, cols)
        hx = rgb_to_hex(pr, pg, pb)
        pbbtn = tk.Button(
            pal_grid,
            bg=hx,
            activebackground=hx,
            width=3,
            height=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=PANEL_BORDER,
            command=lambda i=idx: set_color(i),
        )
        pbbtn.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
    for c in range(cols):
        pal_grid.columnconfigure(c, weight=1)

    right = tk.Frame(body, width=PROPERTIES_WIDTH, bg=PANEL_BG, highlightthickness=1, highlightbackground=PANEL_BORDER)
    right.pack(side=tk.RIGHT, fill=tk.Y)
    right.pack_propagate(False)

    mid = tk.Frame(body, bg=PANEL_BG)
    mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    gl_view = ShapeCanvas(mid, width=1024, height=640)
    _canvas_singleton.append(gl_view)
    try:
        gl_view.configure(takefocus=1, highlightthickness=1, highlightbackground=PANEL_BORDER)
    except tk.TclError:
        pass
    gl_view.pack(fill=tk.BOTH, expand=True)

    rp = tk.Frame(right, bg=PANEL_BG)
    rp.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    title_f = tkfont.Font(family="Segoe UI", size=11, weight="bold")
    tk.Label(rp, text="Properties", font=title_f, fg=TEXT_COLOR, bg=PANEL_BG).pack(anchor="w")
    tk.Label(rp, textvariable=selected_color_var, fg=SUBTEXT_COLOR, bg=PANEL_BG).pack(anchor="w", pady=(4, 8))

    tk.Label(rp, text="Draw mode", fg=TEXT_COLOR, bg=PANEL_BG).pack(anchor="w")
    mode_row = tk.Frame(rp, bg=PANEL_BG)
    mode_row.pack(fill=tk.X, pady=(2, 8))
    for label, dm in [("Poly", "polygon"), ("Line", "polyline"), ("Tri", "triangles"), ("Pts", "points")]:
        tk.Button(
            mode_row,
            text=label,
            font=tool_font,
            fg=TEXT_COLOR,
            bg=BUTTON_BG,
            command=lambda m=dm: set_mode(m),  # type: ignore[misc]
        ).pack(side=tk.LEFT, padx=2)

    tk.Checkbutton(
        rp,
        text="Filled (polygon / triangles)",
        variable=filled_var,
        fg=TEXT_COLOR,
        bg=PANEL_BG,
        selectcolor=PANEL_BORDER,
        activebackground=PANEL_BG,
        activeforeground=TEXT_COLOR,
        command=lambda: set_filled(bool(filled_var.get())),
    ).pack(anchor="w", pady=(0, 8))

    tk.Checkbutton(
        rp,
        text="Move: kéo shape / poly đang vẽ (LMB)",
        variable=move_mouse_var,
        fg=TEXT_COLOR,
        bg=PANEL_BG,
        selectcolor=PANEL_BORDER,
        activebackground=PANEL_BG,
        activeforeground=TEXT_COLOR,
    ).pack(anchor="w", pady=(0, 2))
    tk.Label(
        rp,
        text="Đưa con trỏ lên shape → lăn chuột hoặc chọn ô màu menu trái (Ctrl+wheel = scale)",
        fg=SUBTEXT_COLOR,
        bg=PANEL_BG,
        font=("Segoe UI", 8),
        wraplength=PROPERTIES_WIDTH - 20,
        justify=tk.LEFT,
    ).pack(anchor="w", pady=(0, 6))

    tk.Label(rp, text="Scale (Ctrl + lăn chuột)", fg=TEXT_COLOR, bg=PANEL_BG).pack(anchor="w", pady=(4, 0))
    sc_row = tk.Frame(rp, bg=PANEL_BG)
    sc_row.pack(fill=tk.X, pady=(2, 8))
    tk.Button(sc_row, text="Lớn +", font=tool_font, fg=TEXT_COLOR, bg=BUTTON_BG, command=ui_scale_up).pack(
        side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
    )
    tk.Button(sc_row, text="Nhỏ −", font=tool_font, fg=TEXT_COLOR, bg=BUTTON_BG, command=ui_scale_down).pack(
        side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0)
    )

    rot_row = tk.Frame(rp, bg=PANEL_BG)
    rot_row.pack(fill=tk.X, pady=(0, 8))
    tk.Button(rot_row, text="Rot −15°", fg=TEXT_COLOR, bg=BUTTON_BG, command=lambda: rotate_target(-15.0)).pack(
        side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
    )
    tk.Button(rot_row, text="Rot +15°", fg=TEXT_COLOR, bg=BUTTON_BG, command=lambda: rotate_target(15.0)).pack(
        side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0)
    )

    act = tk.Frame(rp, bg=PANEL_BG)
    act.pack(fill=tk.X, pady=(0, 4))
    tk.Button(act, text="Commit", fg=TEXT_COLOR, bg=BUTTON_BG, command=ui_commit).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
    tk.Button(act, text="Undo", fg=TEXT_COLOR, bg=BUTTON_BG, command=ui_undo).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
    tk.Button(rp, text="Clear", fg=TEXT_COLOR, bg=BUTTON_BG, command=ui_clear).pack(fill=tk.X, pady=(4, 0))

    status = tk.Frame(root, height=STATUS_BAR_HEIGHT, bg=PANEL_BG, highlightthickness=1, highlightbackground=PANEL_BORDER)
    status.pack(side=tk.BOTTOM, fill=tk.X)
    status.pack_propagate(False)
    tk.Label(status, textvariable=status_var, fg=TEXT_COLOR, bg=PANEL_BG, font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=2)

    def handle_mouse(mx: float, my: float, button: int, is_press: bool) -> None:
        nonlocal drag_idx, tool_kind, tool_dragging, tool_start, draw_mode
        nonlocal move_mouse_dragging, move_mouse_target_idx, move_mouse_drag_free_current, move_mouse_last_pos
        nonlocal moved_shape_idx
        if move_mouse_var.get():
            if button == 1 and is_press:
                idx = shape_hit_index(mx, my)
                if idx is not None:
                    tool_dragging = False
                    drag_idx = None
                    current.clear()
                    move_mouse_dragging = True
                    move_mouse_target_idx = idx
                    move_mouse_drag_free_current = False
                    move_mouse_last_pos = (float(mx), float(my))
                    moved_shape_idx = idx
                    return
                if tool_kind != "freeform":
                    return
                if current and current_bbox_hit(mx, my):
                    tool_dragging = False
                    drag_idx = None
                    move_mouse_dragging = True
                    move_mouse_target_idx = None
                    move_mouse_drag_free_current = True
                    move_mouse_last_pos = (float(mx), float(my))
                    return
            elif button == 1 and not is_press:
                move_mouse_dragging = False
                move_mouse_target_idx = None
                move_mouse_drag_free_current = False
                return
            elif tool_kind != "freeform":
                return

        if tool_kind != "freeform":
            if button == 1 and is_press:
                tool_dragging = True
                tool_start = (float(mx), float(my))
                draw_mode = "polygon"
                mode_var.set(draw_mode)
                current[:] = tool_vertices(tool_kind, mx, my, mx, my)
                return
            if button == 1 and not is_press and tool_dragging:
                tool_dragging = False
                verts = tool_vertices(tool_kind, tool_start[0], tool_start[1], mx, my)
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
            return

        if button == 1 and is_press:
            if drag_idx is None:
                current.append((float(mx), float(my)))
        elif button == 3 and is_press:
            if not current:
                return
            best_i = min(
                range(len(current)),
                key=lambda i: (current[i][0] - mx) ** 2 + (current[i][1] - my) ** 2,
            )
            if math.hypot(current[best_i][0] - mx, current[best_i][1] - my) <= hit_radius:
                current.pop(best_i)
        elif button == 2 and is_press:
            if not current:
                return
            best_i = min(
                range(len(current)),
                key=lambda i: (current[i][0] - mx) ** 2 + (current[i][1] - my) ** 2,
            )
            if math.hypot(current[best_i][0] - mx, current[best_i][1] - my) <= hit_radius:
                drag_idx = best_i
        elif button == 2 and not is_press:
            drag_idx = None

    def handle_cursor_motion(fx: float, fy: float) -> None:
        nonlocal tool_dragging, tool_kind, tool_start, draw_mode
        nonlocal move_mouse_dragging, move_mouse_target_idx, move_mouse_drag_free_current, move_mouse_last_pos
        if move_mouse_dragging:
            dx = float(fx) - move_mouse_last_pos[0]
            dy = float(fy) - move_mouse_last_pos[1]
            if dx != 0.0 or dy != 0.0:
                if move_mouse_drag_free_current and current:
                    current[:] = [(float(vx + dx), float(vy + dy)) for vx, vy in current]
                elif move_mouse_target_idx is not None:
                    verts = shapes[move_mouse_target_idx].vertices
                    verts[:] = [(float(vx + dx), float(vy + dy)) for vx, vy in verts]
                move_mouse_last_pos = (float(fx), float(fy))
            return
        if tool_kind != "freeform" and tool_dragging:
            draw_mode = "polygon"
            mode_var.set(draw_mode)
            current[:] = tool_vertices(tool_kind, tool_start[0], tool_start[1], fx, fy)
            return
        if drag_idx is None or not current:
            return
        current[drag_idx] = (float(fx), float(fy))

    def sync_canvas() -> None:
        """Vẽ lại ngay sau khi đổi dữ liệu (tránh chỉ dựa timer 16ms). Không gọi _display() để không chồng after()."""
        try:
            gl_view.update_idletasks()
            gl_view.tkMakeCurrent()
            gl_view.redraw()
            gl_view.tkSwapBuffers()
        except Exception:
            pass

    def canvas_mouse(event: tk.Event) -> None:
        gl_view.focus_set()
        t = event.type
        is_press = t == 4 or getattr(t, "name", "") == "ButtonPress"
        bn = int(getattr(event, "num", 1) or 1)
        handle_mouse(float(event.x), float(event.y), bn, is_press)
        sync_canvas()

    def canvas_motion(event: tk.Event) -> None:
        nonlocal hover_shape_idx
        mouse_xy[0] = float(event.x)
        mouse_xy[1] = float(event.y)
        hover_shape_idx = shape_hit_index(float(event.x), float(event.y))
        handle_cursor_motion(float(event.x), float(event.y))
        sync_canvas()

    def finish_tool_drag_outside(event: tk.Event) -> None:
        nonlocal tool_dragging
        if tool_kind == "freeform" or not tool_dragging:
            return
        if event.widget == gl_view:
            return
        handle_mouse(mouse_xy[0], mouse_xy[1], 1, False)
        sync_canvas()

    def canvas_wheel(event: tk.Event) -> None:
        nonlocal color_idx
        mx, my = float(event.x), float(event.y)
        ctrl_down = bool(event.state & 0x0004)
        dy = getattr(event, "delta", 0) or 0

        if ctrl_down:
            if dy > 0:
                apply_scale_target(1.08)
            elif dy < 0:
                apply_scale_target(0.92)
            sync_canvas()
            return

        hi = shape_hit_index(mx, my)
        if hi is not None and dy != 0 and not tool_dragging and not (tool_kind == "freeform" and current):
            pi = palette_index_for_rgb(shapes[hi].color)
            if dy > 0:
                pi = (pi + 1) % len(PALETTE)
            else:
                pi = (pi - 1) % len(PALETTE)
            shapes[hi].color = PALETTE[pi]
            color_idx = pi
            selected_color_var.set(f"Shape #{hi + 1} — màu {pi + 1}/{len(PALETTE)}")
            sync_canvas()
            return

        if dy > 0:
            color_idx = (color_idx + 1) % len(PALETTE)
        elif dy < 0:
            color_idx = (color_idx - 1) % len(PALETTE)
        else:
            return
        selected_color_var.set(f"Color {color_idx + 1}/{len(PALETTE)}")
        sync_canvas()

    def canvas_key(event: tk.Event) -> None:
        nonlocal filled, color_idx, draw_mode, tool_dragging
        keysym = event.keysym
        try:
            if keysym == "Escape":
                tool_dragging = False
                root.quit()
                return
            if keysym in ("Return", "KP_Enter") and tool_kind == "freeform":
                commit_shape()
                return
            if keysym == "c" or keysym == "C":
                shapes.clear()
                current.clear()
                return
            if keysym == "m" or keysym == "M":
                cycle_color_hover_or_brush(1)
                return
            if keysym == "n" or keysym == "N":
                cycle_color_hover_or_brush(-1)
                return
            if keysym == "p" or keysym == "P":
                draw_mode = "polygon"
                mode_var.set(draw_mode)
                return
            if keysym == "l" or keysym == "L":
                draw_mode = "polyline"
                mode_var.set(draw_mode)
                return
            if keysym == "t" or keysym == "T":
                draw_mode = "triangles"
                mode_var.set(draw_mode)
                return
            if keysym == "o" or keysym == "O":
                draw_mode = "points"
                mode_var.set(draw_mode)
                return
            if keysym == "h" or keysym == "H":
                print_help()
                return
            if keysym == "Up":
                move_target(0, -12)
                return
            if keysym == "Down":
                move_target(0, 12)
                return
            if keysym == "Left":
                move_target(-12, 0)
                return
            if keysym == "Right":
                move_target(12, 0)
                return
            if keysym == "q" or keysym == "Q":
                rotate_target(-15.0)
                return
            if keysym == "e" or keysym == "E":
                rotate_target(15.0)
                return
            if keysym == "f" or keysym == "F":
                filled = not filled
                filled_var.set(filled)
                return
            ch = event.char
            if ch and ch in "123456789":
                n = int(ch) - 1
                if n < len(PALETTE):
                    set_color(n)
        finally:
            if keysym != "Escape":
                sync_canvas()

    def canvas_undo(_event: tk.Event | None = None) -> None:
        nonlocal hover_shape_idx
        if tool_kind == "freeform" and current:
            current.pop()
        elif shapes:
            shapes.pop()
        hover_shape_idx = shape_hit_index(mouse_xy[0], mouse_xy[1])
        sync_canvas()

    def ui_destroy() -> None:
        if _canvas_singleton and getattr(_canvas_singleton[0], "_vbo", None):
            try:
                gl_view.tkMakeCurrent()
                glDeleteBuffers(1, [_canvas_singleton[0]._vbo])
                glDeleteVertexArrays(1, [_canvas_singleton[0]._vao])
            except tk.TclError:
                pass
        root.destroy()

    for ev in (
        "<ButtonPress-1>",
        "<ButtonRelease-1>",
        "<ButtonPress-2>",
        "<ButtonRelease-2>",
        "<ButtonPress-3>",
        "<ButtonRelease-3>",
    ):
        gl_view.bind(ev, canvas_mouse)
    gl_view.bind("<Motion>", canvas_motion)
    gl_view.bind("<B1-Motion>", canvas_motion)
    gl_view.bind("<B2-Motion>", canvas_motion)
    gl_view.bind("<B3-Motion>", canvas_motion)
    gl_view.bind("<MouseWheel>", canvas_wheel)
    gl_view.bind("<Key>", canvas_key)
    gl_view.bind("<Control-z>", canvas_undo)
    gl_view.bind("<Enter>", lambda _e: gl_view.focus_set())
    root.bind_all("<ButtonRelease-1>", finish_tool_drag_outside)

    root.protocol("WM_DELETE_WINDOW", ui_destroy)

    print_help()
    refresh_toolbar()
    root.mainloop()


if __name__ == "__main__":
    main()
