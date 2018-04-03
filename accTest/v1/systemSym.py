import numpy as np
import math

from matplotlib import pyplot as plt
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------


class Vehicle:
    def __init__(self, wheelbase, timestep):
        self.timestep = timestep
        self.wheelbase = wheelbase
        self.X = np.matrix([[0], [0], [0], [0]])

    def f(self, f_state, f_input):
        r = self.wheelbase/np.tan(f_input[1, 0])
        beta = self.X[0, 0] / self.wheelbase*np.tan(f_input[1, 0])*self.timestep
        newX = self.X + np.matrix([[self.timestep*f_input[0, 0]],
                                   [-r*np.sin(self.X[3, 0])+r*np.sin(self.X[3, 0]+beta)],
                                   [r*np.cos(self.X[3, 0])-r*np.cos(self.X[3, 0]+beta)], [beta]])
        return newX

    def h(self, f_state):
        return np.Matrix([[f_state[0, 0]]])

    # F: Matrix([[1, 0, 0, 0], [t*cos(beta + theta), 1, 0, -R*cos(theta) + R*cos(beta + theta)], [t*sin(beta + theta), 0, 1, -R*sin(theta) + R*sin(beta + theta)], [t*tan(alpha)/w, 0, 0, 1]])

    # Matrix([[1, 0, 0, 0], [t*cos(t*v*tan(alpha)/w + theta), 1, 0, -w*cos(theta)/tan(alpha) + w*cos(t*v*tan(alpha)/w + theta)/tan(alpha)], [t*sin(t*v*tan(alpha)/w + theta), 0, 1, -w*sin(theta)/tan(alpha) + w*sin(t*v*tan(alpha)/w + theta)/tan(alpha)], [t*tan(alpha)/w, 0, 0, 1]])

    def Fx(self, f_state, f_input):
        d = self.timestep * f_state[0, 0]
        alpha = f_input[1, 0]
        beta = d/self.wheelbase*np.tan(alpha)
        r = self.wheelbase/np.tan(alpha)
        theta = f_state[3, 0]
        return np.matrix([[1.0, 0.0, 0.0, 0.0],
                          [self.timestep*np.cos(beta+theta), 1.0, 0.0,
                           -r*np.cos(theta)+r*np.cos(beta+theta)],
                          [self.timestep*np.sin(beta+theta), 0.0, 1.0, -
                           r*np.sin(theta)+r*np.sin(beta+theta)],
                          [self.timestep*np.tan(alpha)/self.wheelbase, 0.0, 0.0, 1.0]])
    # d = v*time
    # Robot rotati1on
    # beta = (d/w)*sympy.tan(alpha)
    # R
    # r = w/sympy.tan(alpha)

    # V: Matrix([[t, 0],
    #           , [0, d*(tan(alpha)**2 + 1)*cos(beta + theta)/tan(alpha) - w*(-tan(alpha)**2 - 1)*sin(theta)/tan(alpha)**2 + w*(-tan(alpha)**2 - 1)*sin(beta + theta)/tan(alpha)**2]
    #           , [0, d*(tan(alpha)**2 + 1)*sin(beta + theta)/tan(alpha) + w*(-tan(alpha)**2 - 1)*cos(theta)/tan(alpha)**2 - w*(-tan(alpha)**2 - 1)*cos(beta + theta)/tan(alpha)**2]
    #           , [0, d*(tan(alpha)**2 + 1)/w]])
    def Fu(self, f_state, f_input):
        d = self.timestep * f_state[0, 0]
        beta = d/self.wheelbase*np.tan(f_input[1, 0])
        r = self.wheelbase/np.tan(f_input[1, 0])
        alpha = f_input[1, 0]
        theta = f_state[3, 0]
        return np.matrix([[self.timestep, 0],
                          [0, d*(np.tan(alpha)**2+1)*np.cos(beta+theta)/np.tan(alpha) - self.wheelbase*(-1*np.tan(alpha)**2-1)*np.sin(
                              theta) / np.tan(alpha)**2 + self.wheelbase*(-1*np.tan(alpha)**2 - 1)*np.sin(beta+theta)/np.tan(alpha)**2],
                          [0, d*(np.tan(alpha)**2+1)*np.sin(beta+theta)/np.tan(alpha) + self.wheelbase*(-1*np.tan(alpha)**2-1)*np.cos(
                              theta) / np.tan(alpha)**2 - self.wheelbase*(-1*np.tan(alpha)**2 - 1)*np.cos(beta+theta)/np.tan(alpha)**2],
                          [0, d*(np.tan(alpha)**2+1)/self.wheelbase]])

    def H(self, f_state):
        return np.Matrix([[1]])
# ----------------------------------------------------------------------------1--


def main():
    car1 = Vehicle(26, 0.1)

    alpha = math.pi/18*2

    V = []
    X = []
    Y = []

    l_input = np.matrix([[100], [alpha]])
    newX = car1.f(car1.X, l_input)
    car1.X = newX

    acc = 0
    l_input = np.matrix([[acc], [alpha]])
    for i in range(100):
        newX = car1.f(car1.X, l_input)
        car1.X = newX
        V.append(newX[0, 0])
        X.append(newX[1, 0])
        Y.append(newX[2, 0])

    l_input = np.matrix([[-200], [alpha]])
    newX = car1.f(car1.X, l_input)
    car1.X = newX

    alpha = -math.pi/18*2
    acc = 0
    l_input = np.matrix([[acc], [alpha]])
    for i in range(100):
        newX = car1.f(car1.X, l_input)
        car1.X = newX
        V.append(newX[0, 0])
        X.append(newX[1, 0])
        Y.append(newX[2, 0])

    plt.figure()
    plt.plot(V)
    plt.figure()
    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    main()
