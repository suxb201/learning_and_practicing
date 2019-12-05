from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import *
import sys

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


array_x = [0, 0]
array_y = [0, 0]
array_length = 0


def mouse_func(_, state, pox, poy):
    if state != 0:
        return
    pox, poy = transform(pox, poy)
    draw_point(pox, poy)
    global array_length
    array_length += 1
    array_x[array_length - 1] = pox
    array_y[array_length - 1] = poy
    if array_length == 2:
        draw_line(array_x[0], array_y[0], array_x[1], array_y[1])
        array_length = 0


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
