from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from numpy import *
import sys

WINDOWS_WIDTH = 800
WINDOWS_HEIGHT = 400
a = 0
b = 0


def draw_dot(x, y):
    glBegin(GL_POINTS)
    x = x / WINDOWS_WIDTH
    y = y / WINDOWS_HEIGHT
    glVertex2f(x, y)
    glEnd()


def display_func():
    glClearColor(52.0 / 255, 73.0 / 255, 94.0 / 255, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(231.0 / 255, 76.0 / 255, 60.0 / 255)
    x = 0
    y = b
    aa = a * a
    bb = b * b
    dx = bb / sqrt(aa + bb)
    p = bb - aa * b
    while dx <= y:
        draw_dot(x, y)
        draw_dot(x, -y)
        draw_dot(-x, y)
        draw_dot(-x, -y)
        x += 1
        if p > 0:
            y -= 1
        p = bb * ((x + 1) ** 2) + aa * (y * y - y) - aa * bb
    p = bb * (x * x + x) + aa * (y * y - y) - aa * bb
    while y > 0:
        draw_dot(x, y)
        draw_dot(x, -y)
        draw_dot(-x, y)
        draw_dot(-x, -y)
        y -= 1
        p -= 2 * aa * y - aa
        if p < 0:
            x += 1
            p += 2 * bb * (x + 1)
    glFlush()


array_x = [0, 0]
array_y = [0, 0]
array_length = 0


def main():
    global a, b
    a = int(input('a:'))
    b = int(input('b:'))
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(WINDOWS_WIDTH, WINDOWS_HEIGHT)
    glutCreateWindow("Computer Graphics: Class_1")
    glutDisplayFunc(display_func)
    glutMainLoop()


main()
