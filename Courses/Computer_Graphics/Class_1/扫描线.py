from typing import List, Any

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import *
import sys
import time

WINDOWS_WIDTH = 800
WINDOWS_HEIGHT = 400


def transform(x, y):
    return x, WINDOWS_HEIGHT - y


def display_func():
    pass


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
    glFlush()


def draw_line(x0, y0, x1, y1):
    flag = False
    if abs(x1 - x0) < abs(y1 - y0):
        flag = True
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    # print(x0, y0, x1, y1)
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
    glFlush()


dots = []
edges = []
iet = [[] for i in range(WINDOWS_HEIGHT)]


def polyfill():
    global dots, edges, iet
    aet = []
    for y in range(WINDOWS_HEIGHT):
        for edge in iet[y]:
            i = 0
            while i < len(aet):
                k = (edges[edge][0] - edges[edge][2]) / (edges[edge][1] - edges[edge][3])
                if aet[i][0] + aet[i][1] < edges[edge][0] + k:
                    break
                i += 1
            aet.insert(i, [edges[edge][0],
                           (edges[edge][0] - edges[edge][2]) / (edges[edge][1] - edges[edge][3]),
                           edges[edge][3]
                           ])
        aet = list(filter(lambda x: x[2] != y, aet))
        for i in range(0, len(aet), 2):
            draw_line(round(aet[i][0]), y, round(aet[i + 1][0]), y)
        for edge in aet:
            edge[0] += edge[1]
    dots = []
    edges = []
    iet = [[] for i in range(WINDOWS_HEIGHT)]


def add_edge(x0, y0, x1, y1):
    if y0 == y1:
        return
    if y0 < y1:
        edges.append((x0, y0, x1, y1))
        iet[y0].append(len(edges) - 1)
    else:
        edges.append((x1, y1, x0, y0))
        iet[y1].append(len(edges) - 1)


def mouse_func(button, state, pox, poy):
    if state == 0:
        return
    if button == 2:
        if len(dots) < 3:
            return
        loc = len(dots) - 1
        add_edge(dots[loc][0], dots[loc][1], dots[0][0], dots[0][1])
        draw_line(dots[loc][0], dots[loc][1], dots[0][0], dots[0][1])
        polyfill()
        return
    pox, poy = transform(pox, poy)
    draw_point(pox, poy)
    dots.append((pox, poy))
    if len(dots) >= 2:
        loc = len(dots) - 1
        add_edge(dots[loc - 1][0], dots[loc - 1][1], dots[loc][0], dots[loc][1])
        draw_line(dots[loc - 1][0], dots[loc - 1][1], dots[loc][0], dots[loc][1])


def main():
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(WINDOWS_WIDTH, WINDOWS_HEIGHT)
    glutCreateWindow("Computer Graphics: Class_1")

    glClearColor(52.0 / 255, 73.0 / 255, 94.0 / 255, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(231.0 / 255, 76.0 / 255, 60.0 / 255)
    glBegin(GL_POINTS)
    glEnd()
    glFlush()

    glutDisplayFunc(display_func)
    glutMouseFunc(mouse_func)
    glutMainLoop()


main()
