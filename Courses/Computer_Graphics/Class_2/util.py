from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

WINDOWS_WIDTH = 800
WINDOWS_HEIGHT = 400


def draw_dot(x, y):
    glBegin(GL_POINTS)
    x = x * 2.0 / WINDOWS_WIDTH - 1
    y = y * 2.0 / WINDOWS_HEIGHT - 1
    glVertex2f(x, y)
    glEnd()


def draw_point(x, y):
    glPointSize(4)
    draw_dot(x, y)
    glPointSize(1)


def draw_line(x0, y0, x1, y1):
    flag = False
    if abs(x1 - x0) < abs(y1 - y0):
        flag = True
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dy2 = dy * 2
    dx2 = dx * 2

    tx = 1 if x1 - x0 >= 0 else -1
    ty = 1 if y1 - y0 >= 0 else -1

    now_x = x0
    now_y = y0
    d = -dx
    while now_x != x1:
        now_x += tx
        d += dy2
        if d > 0:
            now_y += ty
            d -= dx2
        if flag is False:
            draw_dot(now_x, now_y)
        else:
            draw_dot(now_y, now_x)


def transform(x, y):
    return x, WINDOWS_HEIGHT - y
