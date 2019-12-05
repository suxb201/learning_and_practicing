from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import *

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


def draw_circle(x0, y0, x1, y1):
    r = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    r = round(r)
    x0 = round(x0)
    y0 = round(y0)
    x = 0
    y = r
    p = 3 - 2 * r
    while x <= y:
        draw_dot(x0 + x, y0 + y)
        draw_dot(x0 - x, y0 + y)
        draw_dot(x0 + x, y0 - y)
        draw_dot(x0 - x, y0 - y)
        draw_dot(x0 + y, y0 + x)
        draw_dot(x0 - y, y0 + x)
        draw_dot(x0 + y, y0 - x)
        draw_dot(x0 - y, y0 - x)
        if p >= 0:
            p += 4 * (x - y) + 10
            y -= 1
        else:
            p += 4 * x + 6
        x += 1
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
        draw_circle(array_x[0], array_y[0], array_x[1], array_y[1])
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
