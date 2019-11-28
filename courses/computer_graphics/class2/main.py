from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import *
import sys
import time

from util import *


def liang_barsky(line, rectangle):
    xl = min(rectangle[0], rectangle[2])
    xr = max(rectangle[0], rectangle[2])
    yb = min(rectangle[1], rectangle[3])
    yt = max(rectangle[1], rectangle[3])
    x1, y1 = line[0], line[1]
    dx = line[2] - x1
    dy = line[3] - y1
    p1 = -dx
    p2 = dx
    p3 = -dy
    p4 = dy
    q1 = x1 - xl
    q2 = xr - x1
    q3 = y1 - yb
    q4 = yt - y1
    if dx == 0:
        if q1 < 0 or q2 < 0:
            return None
        umax = 0
        umin = 1
        for p, q in zip((p3, p4), (q3, q4)):
            if p < 0:
                umax = max(umax, q / p)
            else:
                umin = min(umin, q / p)
    elif dy == 0:
        if q3 < 0 or q4 < 0:
            return None
        umax = 0
        umin = 1
        for p, q in zip((p1, p2), (q1, q2)):
            if p < 0:
                umax = max(umax, q / p)
            else:
                umin = min(umin, q / p)
    else:
        umax = 0
        umin = 1
        for p, q in zip((p1, p2, p3, p4), (q1, q2, q3, q4)):
            if p < 0:
                umax = max(umax, q / p)
            else:
                umin = min(umin, q / p)
    if umax > umin:
        return None
    return [int(i) for i in [x1 + dx * umax, y1 + dy * umax, x1 + dx * umin, y1 + dy * umin]]


def display_func():
    global line_edges
    if len(rectangle_edges) == 1 and dot_number == 0:
        line_edges = [liang_barsky(line_edge, rectangle_edges[0]) for line_edge in line_edges]
        line_edges = [line_edge for line_edge in line_edges if line_edge is not None]
    glClearColor(52.0 / 255, 73.0 / 255, 94.0 / 255, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    for edge in line_edges:
        glColor3f(231.0 / 255, 76.0 / 255, 60.0 / 255)
        draw_point(edge[0], edge[1])
        draw_point(edge[2], edge[3])
        draw_line(edge[0], edge[1], edge[2], edge[3])
    for edge in rectangle_edges:
        glColor3f(142 / 255, 68 / 255, 173 / 255)
        draw_point(edge[0], edge[1])
        draw_point(edge[2], edge[3])
        draw_line(edge[0], edge[1], edge[0], edge[3])
        draw_line(edge[0], edge[1], edge[2], edge[1])
        draw_line(edge[0], edge[3], edge[2], edge[3])
        draw_line(edge[2], edge[1], edge[2], edge[3])
    glutSwapBuffers()


dot_number = 0
line_edges = []
rectangle_edges = []
now_state = 0  # 0 line 1 rectangle


def mouse_func(button, state, pox, poy):
    global dot_number, now_state, rectangle_edges
    if state == 0:
        return
    if button == 2:
        if dot_number == 0:
            now_state ^= 1
        return
    pox, poy = transform(pox, poy)
    now_edges = line_edges if now_state == 0 else rectangle_edges
    if dot_number == 1:
        now_edges[len(now_edges) - 1][2] = pox
        now_edges[len(now_edges) - 1][3] = poy
        dot_number = 0
    else:
        now_edges.append([pox, poy, pox, poy])
        dot_number = 1
    if len(rectangle_edges) == 2:
        rectangle_edges = rectangle_edges[1:]


def passive_motion_func(pox, poy):
    pox, poy = transform(pox, poy)
    now_edges = line_edges if now_state == 0 else rectangle_edges
    if dot_number == 1:
        now_edges[len(now_edges) - 1][2] = pox
        now_edges[len(now_edges) - 1][3] = poy
    glutPostRedisplay()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(WINDOWS_WIDTH, WINDOWS_HEIGHT)
    glutCreateWindow("Computer Graphics: Class_1")

    glutDisplayFunc(display_func)
    glutMouseFunc(mouse_func)
    glutPassiveMotionFunc(passive_motion_func)
    glutMainLoop()


main()
