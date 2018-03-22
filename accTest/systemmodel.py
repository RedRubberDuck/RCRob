import numpy as np
from matplotlib import pyplot as plt
import testhelp


# class Vechile:
#     def __init__(self, wheelbase, timeStep, x=0, y=0, gamma=0):
#         self.wheelbase = wheelbase
#         self.timeStep = timeStep
#
#         self.x = 0
#         self.y = 0
#         self.gamma = 0
#
#     def StateTransition(self, forwardVelocity, steeringAngle):
#         x = self.x + np.cos(self.gamma)*self.timeStep * forwardVelocity
#         y = self.y + np.sin(self.gamma)*self.timeStep * forwardVelocity
#         gamma = self.gamma + forwardVelocity * \
#             np.tan(np.radians(steeringAngle))/self.wheelbase*self.timeStep
#
#         self.x = x
#         self.y = y
#         self.gamma = gamma
#
#     def GetObservation(self):
#         return self.x, self.y, self.gamma


class Vechile:
    def __init__(self, wheelbase, timestep, x=0, y=0, gamma=0):
        self.wheelbase = wheelbase
        self.timestep = timestep

        self.x = x
        self.y = y
        self.gamma = gamma

        self.v_f = 0
        self.w = 0

    def f(self, accel_f, alpha):
        v_f = self.v_f + self.timestep*accel_f
        x = self.x + self.timestep * self.v_f * \
            np.cos(self.gamma) + self.timestep**2 * np.cos(self.gamma) * accel_f/2

        y = self.y + self.timestep * self.v_f * \
            np.sin(self.gamma) + self.timestep**2 * np.sin(self.gamma) * accel_f/2

        w = np.tan(np.radians(alpha))/self.wheelbase*self.v_f
        gamma = self.gamma + self.w * self.timestep

        return v_f, x, y, w, gamma

    def setState(self, v_f, x, y, w, gamma):
        self.v_f = v_f
        self.x = x
        self.y = y
        self.w = w
        self.gamma = gamma

    def h(self):
        return (self.v_f, self.w)

    def F(self, accel_f, alpha):
        F = np.matrix([[1, 0, 0, 0, 0],
                       [np.cos(self.gamma)*self.timestep, 1, 0, 0, -1*np.sin(self.gamma)
                        * (self.v_f*self.timestep+self.timestep**2*accel_f/2)],
                       [np.sin(self.gamma)*self.timestep, 0, 1, 0, np.cos(self.gamma)
                        * (self.v_f*self.timestep+self.timestep**2*accel_f/2)],
                       [np.tan(np.radians(alpha))/self.wheelbase, 0, 0, 0, 0],
                       [0, 0, 0, self.timestep, 1]])
        return F

    def H(self):
        H = np.matrix([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
        return H


def testSecvenceGenerate(timestep):
    alpha_a, accel_a, vel_a = testhelp.testSemnalGenerate(timestep)
    alpha_a_err, accel_a_err, vel_a_err = testhelp.generateError(
        alpha_a, 3, 3, accel_a, 0.0, 0.5, vel_a, 0.01, 0.01)

    plt.figure()
    plt.subplot(311)
    plt.plot(alpha_a)
    plt.plot(alpha_a_err)
    plt.subplot(312)
    plt.plot(accel_a)
    plt.plot(accel_a_err)
    plt.subplot(313)
    plt.plot(vel_a)
    plt.plot(vel_a_err)
    plt.show()

    return alpha_a, alpha_a_err, accel_a, accel_a_err, vel_a, vel_a_err


def main():
    print("Start systemmodel main")
    timestep = 0.1
    car1 = Vechile(26, timestep)
    car1Err = Vechile(26, timestep)

    print(car1.F(0, 0))

    alpha_a, alpha_a_err, accel_a, accel_a_err, vel_a, vel_a_err = testSecvenceGenerate(timestep)

    X_car1 = []
    Y_car1 = []
    Gamma_car1 = []

    X_car1Err = []
    Y_car1Err = []
    Gamma_car1Err = []

    for accel, alpha, accel_err, alpha_err in zip(accel_a, alpha_a, accel_a_err, alpha_a_err):

        v_f, x, y, w, gamma = car1.f(accel, alpha)
        car1.setState(v_f, x, y, w, gamma)
        X_car1.append(x)
        Y_car1.append(y)
        Gamma_car1.append(gamma)

        v_f, x, y, w, gamma = car1Err.f(accel_err, alpha_err)
        car1Err.setState(v_f, x, y, w, gamma)
        X_car1Err.append(x)
        Y_car1Err.append(y)
        Gamma_car1Err.append(gamma)

    plt.figure()
    plt.subplot(211)
    plt.plot(X_car1, Y_car1)
    plt.plot(X_car1Err, Y_car1Err)
    plt.subplot(212)
    plt.plot(Gamma_car1)
    plt.plot(Gamma_car1Err)

    plt.show()

    print("End systemmodel main")


if __name__ == '__main__':
    main()
