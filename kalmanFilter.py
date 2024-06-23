import numpy as np

class KalmanFilter2D:
    def __init__(self, A, C, Q, R, x0, P0):
        self.A = A    # State transition matrix
        self.C = C    # Observation matrix
        self.Q = Q    # Process noise covariance, null in this problem
        self.R = R    # Observation noise covariance
        self.x = x0   # Initial state estimate
        self.P = P0   # Initial estimate covariance

    def predict(self):
        # Predict the state and the state covariance
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        # Compute the Kalman Gain
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        # Update the state estimate
        self.x = self.x + K @ (z - self.C @ self.x)
        # Update the estimate covariance
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P
        return self.x

    def get_state(self):
        return self.x



"""
# Definire le matrici
dt = 1.0  # Intervallo di tempo

# Matrice di transizione dello stato per il movimento in 2D
F = np.array([[1, 0, dt,  0],
              [0, 1,  0, dt],
              [0, 0,  1,  0],
              [0, 0,  0,  1]])

# Matrice di osservazione (posizione soltanto)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Covarianza del rumore di processo
Q = np.eye(4) * 0.001

# Covarianza del rumore di osservazione
R = np.eye(2)

# Stato iniziale (posizione e velocità iniziali)
x0 = np.array([0, 0, 1, 1])

# Covarianza iniziale
P0 = np.eye(4)

# Inizializzare il filtro di Kalman
kf = KalmanFilter2D(F, H, Q, R, x0, P0)


"""