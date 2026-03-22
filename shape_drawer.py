"""
Interactive 2D Shape Drawer — GLFW + PyOpenGL (+ NumPy)
Orthographic projection, dynamic VBO, mouse placement, keyboard colors.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import glfw
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
        "2D Shape Drawer — LMB add | RMB del | MMB drag | Wheel color | F fill | C clear | Ctrl+Z | 1–9",
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

    vertices: List[Tuple[float, float]] = []
    vbo = glGenBuffers(1)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    color_idx = 0
    filled = True
    drag_idx: int | None = None
    hit_radius = 14.0

    def upload() -> None:
        if vertices:
            arr = np.array(vertices, dtype=np.float32).flatten()
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_DYNAMIC_DRAW)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)

    def on_resize(_w, width: int, height: int) -> None:
        global W, H
        W, H = max(320, width), max(240, height)
        glViewport(0, 0, W, H)

    def on_mouse_button(win, button: int, action: int, mods: int) -> None:
        nonlocal drag_idx
        if action != glfw.PRESS and action != glfw.RELEASE:
            return
        mx, my = glfw.get_cursor_pos(win)
        mx, my = cursor_to_framebuffer(win, mx, my)

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if drag_idx is None:
                vertices.append((float(mx), float(my)))
                upload()
        elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            if not vertices:
                return
            best_i = min(
                range(len(vertices)),
                key=lambda i: (vertices[i][0] - mx) ** 2 + (vertices[i][1] - my) ** 2,
            )
            if math.hypot(vertices[best_i][0] - mx, vertices[best_i][1] - my) <= hit_radius:
                vertices.pop(best_i)
                upload()
        elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
            if not vertices:
                return
            best_i = min(
                range(len(vertices)),
                key=lambda i: (vertices[i][0] - mx) ** 2 + (vertices[i][1] - my) ** 2,
            )
            if math.hypot(vertices[best_i][0] - mx, vertices[best_i][1] - my) <= hit_radius:
                drag_idx = best_i
        elif button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.RELEASE:
            drag_idx = None

    def on_cursor_pos(win, x: float, y: float) -> None:
        if drag_idx is None or not vertices:
            return
        fx, fy = cursor_to_framebuffer(win, x, y)
        vertices[drag_idx] = (float(fx), float(fy))
        upload()

    def on_scroll(_w, _dx: float, dy: float) -> None:
        nonlocal color_idx
        if dy > 0:
            color_idx = (color_idx + 1) % len(PALETTE)
        elif dy < 0:
            color_idx = (color_idx - 1) % len(PALETTE)

    def on_key(_w, key: int, _scancode: int, action: int, mods: int) -> None:
        nonlocal filled, color_idx
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_w, True)
        elif key == glfw.KEY_C:
            vertices.clear()
            upload()
        elif key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL):
            if vertices:
                vertices.pop()
                upload()
        elif key == glfw.KEY_F:
            filled = not filled
        elif glfw.KEY_1 <= key <= glfw.KEY_9:
            n = key - glfw.KEY_1
            if n < len(PALETTE):
                color_idx = n

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

    while not glfw.window_should_close(window):
        glfw.poll_events()

        mvp = ortho_2d_tl_origin(float(W), float(H))

        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(program)
        glUniformMatrix4fv(u_mvp, 1, GL_TRUE, mvp)
        r, g, b = PALETTE[color_idx]
        glUniform3f(u_color, r, g, b)

        glBindVertexArray(vao)
        n = len(vertices)
        if n >= 2:
            if filled and n >= 3:
                glUniform3f(u_color, r, g, b)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDrawArrays(GL_TRIANGLE_FAN, 0, n)
            er, eg, eb = (
                (min(1.0, r + 0.2), min(1.0, g + 0.2), min(1.0, b + 0.2))
                if filled
                else (r, g, b)
            )
            glUniform3f(u_color, er, eg, eb)
            glLineWidth(2.0)
            glDrawArrays(GL_LINE_LOOP, 0, n)
            glUniform3f(u_color, r, g, b)
        if n >= 1:
            glPointSize(8.0)
            glDrawArrays(GL_POINTS, 0, n)

        glBindVertexArray(0)
        glUseProgram(0)

        title = (
            f"2D Shape Drawer | verts={n} | color={color_idx + 1}/{len(PALETTE)} | "
            f"{'fill+edge' if filled else 'edge only'}"
        )
        glfw.set_window_title(window, title)

        glfw.swap_buffers(window)

    glDeleteBuffers(1, [vbo])
    glDeleteVertexArrays(1, [vao])
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
