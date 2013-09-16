#!/usr/env/bin python
import numpy

def is_quadrangle(points):
    def test(p):
        def side(a, b, t):
            v = b - a
            return v[1] * (t[0] - a[0]) - v[0] * (t[1] - a[1])
        return side(p[0], p[1], p[2]) * side(p[0], p[1], p[3]) > 0
    return test(points) and test(numpy.roll(points, -1, axis=0))

def angles(points):
    def test(p):
        norm = numpy.linalg.norm
        a, b = p[1] - p[0], p[2] - p[0]
        return numpy.arccos(numpy.dot(a / norm(a), b / norm(b)))
    angles = []
    for i in range(0, -len(points), -1):
        angles.append(test(numpy.roll(points, i, axis=0)))
    return angles
