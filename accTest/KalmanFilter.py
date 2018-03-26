import numpy as np
import math


class KalmanFilter:

    def __init__(self, X, f, h, F_x, F_u, H, P, Q, R, StateClass, OutputClass):
        # State of the system
        self.X = X
        # State transition func.
        self.f = f
        # Observation trans. func.
        self.h = h
        # Der. state transition func.
        self.F_x = F_x
        # Der. state transition func.
        self.F_u = F_u
        # Der. observation func.
        self.H = H

        # Covariance martix
        self.P = P
        # The covariance of the process noise
        self.Q = Q
        # The covariance of the observation noise
        self.R = R

        #
        self.StateClass = StateClass
        #
        self.OutputClass = OutputClass

    def predict(self, f_state, f_input, P):
        newState = self.f(f_state, f_input)
        F_x = self.F_x(f_state, f_input)
        F_u = self.F_u(f_state, f_input)

        Q = F_u*self.Q*np.transpose(F_u)

        P = F_x*P*np.transpose(F_x) + Q

        return (newState, P)

    def update(self, f_state, f_measurment, P):
        output_sys = self.h(f_state)
        mes_res = f_measurment._Y - output_sys._Y
        print(mes_res)

        H = self.H(f_state)

        res_cov = H * self.P * np.transpose(H) + self.R

        # print(res_cov)

        res_cov_inv = np.linalg.inv(res_cov)
        state_size = f_state.X.shape[0]
        K = P * np.transpose(H) * res_cov_inv

        cor = K * mes_res
        # cor[0, 0] = 0.0
        # cor[1, 0] = 0.0
        # cor[2, 0] = 0.0
        # cor[3, 0] = 0.0
        # cor[4, 0] = 0.0
        # print(cor)
        X = f_state.X+cor
        P = (np.eye(state_size) - K*H)*P

        newState = self.StateClass()
        newState.X = X
        return (newState, P)


def getStateCovariance():
    P = np.matrix([[0.0, 0, 0, 0, 0], [0, 5.0, 0, 0, 0], [
                  0, 0, 5.0, 0, 0], [0, 0, 0, 0.0, 0], [0, 0, 0, 0, math.pi/180]])
    # math.pi/180
    return P


def getProcessNoise(varAcc, timestep, alphaErr, wheelbase, maxVel):
    Q = np.zeros((2, 2))
    Q[0, 0] = varAcc
    Q[1, 1] = alphaErr
    # Q[2, 2] = varAcc*timestep**2/2
    # Q[3, 3] = np.tan(np.radians(alphaErr))*maxVel/wheelbase
    # Q[4, 4] = 0.0
    # # np.tan(np.radians(alphaErr))*maxVel/wheelbase * timestep
    # print('SS', np.tan(np.radians(alphaErr))*maxVel/wheelbase,
    #       np.tan(np.radians(alphaErr))*maxVel/wheelbase * timestep)
    return Q


def getMeasurmentNoise(velf, groErr):
    R = np.zeros((2, 2))
    R[0, 0] = velf
    R[1, 1] = groErr
    print('PP', velf, groErr)
    return R
