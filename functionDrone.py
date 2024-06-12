import math

import numpy as np

class Pursuerer:
    def __init__(self, p, u, theta=math.radians(60), A=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                 B=np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])):
        dt = np.dtype(np.double)
        self.p = np.array(p, dtype=dt)
        self.p = self.p.transpose()

        self.u = np.array(u)
        self.u = self.u.transpose()

        self.theta = theta

        self.A = A

        self.B = B
        '''A = np.array([[1, 2, 3],
                                            [1, 2, 3],
                                            [1, 2, 3]])'''

    def stateUpdate(self):
        Ax = np.matmul(self.A, self.p)
        Bu = np.matmul(self.B, self.u)
        return Ax + Bu

    def visibilityCone(self):
        coneRadius = self.p[2] * math.tan(self.theta / 2)
        return coneRadius

    # nota pDesired deve essere sotto forma di array -> np.array([[1,2,3]])
    # inoltre effettuo la valutazione della velocità per ogni direzione
    # inoltre sia delta che alpha sono parametri per lo spostamento del drone
    def controlUpdate(self, pDesired):
        delta = 10      # parametro per regolare la velocità di movimento lungo x e y
        deltaZ = 1      # parametro per regolare la velocità di movimento lungo z
        alpha = 1
        if np.fabs(self.p[0] - pDesired[0]) >= 5:
            self.u[0] = np.sign(pDesired[0] - self.p[0]) * delta
        else:
            self.u[0] = np.sign(pDesired[0] - self.p[0]) * alpha * np.fabs(self.p[0] - pDesired[0])
        if np.fabs(self.p[1] - pDesired[1]) >= 5:
            self.u[1] = np.sign(pDesired[1] - self.p[1]) * delta
        else:
            self.u[1] = np.sign(pDesired[1] - self.p[1]) * alpha * np.fabs(self.p[1] - pDesired[1])
        if np.fabs(self.p[2] - pDesired[2]) >= 5:
            self.u[2] = np.sign(pDesired[2] - self.p[2]) * deltaZ
        else:
            self.u[2] = np.sign(pDesired[2] - self.p[2]) * alpha * np.fabs(self.p[2] - pDesired[2])

    def controlUpdate2(self, pDesired):
        delta = 1  # parametro per regolare la velocità di movimento lungo x e y
        deltaZ = 1  # parametro per regolare la velocità di movimento lungo z
        alpha = 1
        if np.fabs(self.p[0] - pDesired[0]) >= 1:
            self.u[0] = np.sign(pDesired[0] - self.p[0]) * delta * np.fabs(self.p[0] - pDesired[0])
            prova1 = np.sign(pDesired[0] - self.p[0]) * delta * np.fabs(self.p[0] - pDesired[0])
        else:
            self.u[0] = np.sign(pDesired[0] - self.p[0]) * alpha * np.fabs(self.p[0] - pDesired[0])
            prova11 = np.sign(pDesired[0] - self.p[0]) * alpha * np.fabs(self.p[0] - pDesired[0])
        if np.fabs(self.p[1] - pDesired[1]) >= 1:
            self.u[1] = np.sign(pDesired[1] - self.p[1]) * delta * np.fabs(self.p[1] - pDesired[1])
            prova2 = np.sign(pDesired[1] - self.p[1]) * delta * np.fabs(self.p[1] - pDesired[1])
        else:
            self.u[1] = np.sign(pDesired[1] - self.p[1]) * alpha * np.fabs(self.p[1] - pDesired[1])
        if np.fabs(self.p[2] - pDesired[2]) >= 1:
            self.u[2] = np.sign(pDesired[2] - self.p[2]) * deltaZ * np.fabs(self.p[2] - pDesired[2])
        else:
            self.u[2] = np.sign(pDesired[2] - self.p[2]) * alpha * np.fabs(self.p[2] - pDesired[2])

# funzione che restituisce un True o un False basandosi sulla distanza del drone dal punto desiderato
# è considerato cioè come errore di posizione
    def positionError(self, pDesired):
        epsilon = 1
        errorx = pDesired[0] - self.p[0]
        errory = pDesired[1] - self.p[1]
        errorz = pDesired[2] - self.p[2]
        error_distance = math.sqrt(math.pow(errorx, 2) + math.pow(errory, 2) + math.pow(errorz, 2))
        if error_distance < epsilon:
            return True
        else:
            return False

# TODO scriverla in forma matriciale X+1 = A*X + B*U usando scipy


# todo: scrivere un metodo che cambia il valore di comando U in base al valore fornito in ingresso (che altri non è che la distanza dall'obiettivo)
