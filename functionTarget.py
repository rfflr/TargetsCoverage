import math


class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y
'''
    def linearMovement(self, m, q):
        self.y = m * self.x + q
        return self.y

    def ellipticMovement(self, a, b, x0, y0):
        self.y = y0 + b * math.sqrt(1 - math.pow(self.x - x0, 2) / math.pow(a, 2))
        return self.y
'''