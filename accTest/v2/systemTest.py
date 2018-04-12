import sympy
import numpy as np
from matplotlib import pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF


'''
    In this case the states fo the model are the position and  the oriantetion of the robot.
    The inputs of the system model are the forward velocity and the steering angle.
    The outputs of the system are the robot position and the robot oriantion.
    The robot oriantion can be obtain base on the IMU sensor, the positon can get from simulated localization system.


'''


def systemModelTest():
    x, y, theta, v, alpha, dt, w, dtheta = sympy.symbols(
        'x, y, theta, v, alpha, dt, w, dtheta')

    # dtheta = v/w*sympy.tan(alpha)
    f_xu = sympy.Matrix([[x+dt*v*sympy.cos(theta)],  # position
                         [y+dt*v*sympy.sin(theta)],  # position
                         [theta+dtheta*dt],          # orianntation
                         [v/w*sympy.tan(alpha)]])                  # oriantation

    print("State transition function", f_xu)

    F_x = f_xu.jacobian(sympy.Matrix([x, y, theta, dtheta]))
    F_u = f_xu.jacobian(sympy.Matrix([v, alpha]))

    print("F_x", F_x)
    print("F_u", F_u)
    subs = {x: 0, y: 0, theta: np.radians(
        0), dtheta: 0.0, v: 0.2, alpha: np.radians(20), dt: 0.02, w: 0.26}

    States = None
    States_Lin = None
    prevX = np.matrix([[subs[x]], [subs[y]], [subs[theta]],
                       [subs[dtheta]]], dtype=np.float)
    curX = prevX
    prevU = np.matrix([[0], [subs[alpha]]])
    curU = np.matrix([[subs[v]], [subs[alpha]]])
    subs_ = subs.copy()
    subs_[v] = 0.0
    # subs_[alpha] = 0.0

    rob1 = RobotEKF(wheelbase=0.26, dt=0.02,
                    std_v=0.1, std_alpha=np.radians(1))
    #
    States_Sim = rob1.x

    for i in range(200):
        X = f_xu.evalf(subs=subs)
        rob1.predict(curU)
        newSimX = rob1.x
        States_Sim = np.concatenate((States_Sim, newSimX), axis=1)

        # F_xV = np.matrix(F_x.evalf(subs=subs_))
        F_xV = rob1.F_x(curX, prevU)

        # F_uV = np.matrix(F_u.evalf(subs=subs_))
        F_uV = rob1.F_u(curX, prevU)

        subs[x] = X[0, 0]
        subs[y] = X[1, 0]
        subs[theta] = X[2, 0]
        subs[dtheta] = X[3, 0]

        dX = curX - prevX
        dU = curU - prevU

        newX = np.matrix(curX + F_xV.dot(dX) + F_uV.dot(dU), dtype=np.float)
        prevX = curX
        curX = newX
        prevU = curU

        if i == 0:
            States = np.matrix(X)
            print(dU, dX)
            States_Lin = newX.copy()
        else:
            States = np.concatenate((States, np.matrix(X)), axis=1)
            States_Lin = np.concatenate((States_Lin, newX), axis=1)

        subs_ = subs

    plt.figure()
    plt.plot(States[2, :].tolist()[0])
    plt.plot(States_Lin[2, :].tolist()[0])
    plt.plot(States_Sim[2, :].tolist()[0])
    plt.title('Oriantation')
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(States[0, :].tolist()[0], States[1, :].tolist()[0])
    plt.plot(States_Lin[0, :].tolist()[0], States_Lin[1, :].tolist()[0])
    plt.plot(States_Sim[0, :].tolist()[0], States_Sim[1, :].tolist()[0])
    plt.title('Position')
    # --------------------------------------------------------------------------
    plt.figure()
    plt.plot(States_Lin[3, :].tolist()[0])
    plt.plot(States[3, :].tolist()[0])
    plt.plot(States_Sim[3, :].tolist()[0])
    plt.title('Angle velocity')
    plt.show()


class RobotEKF(EKF):
    # Constructor
    #  @param wheelbase     =   the robot wheelbase
    #  @param dt            =   the timestep of the simulation
    def __init__(self, wheelbase, dt, std_v, std_alpha):
        super(RobotEKF, self).__init__(4, 1, 2)
        self.__w = wheelbase
        self.__dt = dt

        self.M = np.matrix([[std_v**2, 0], [0, std_alpha**2]])

    # Simulate the robot move
    # @param  x             =   the robot initial state
    # @param  u             =   the system model input parameters
    def move(self, x, u):
        dtheta = x[3, 0]
        x[3, 0] = 0.0
        return x + np.matrix([
            [u[0, 0]*self.__dt*np.cos(x[2, 0])],
            [u[0, 0]*self.__dt*np.sin(x[2, 0])],
            [dtheta*self.__dt],
            [u[0, 0]/self.__w*np.tan(u[1, 0])]
        ])

    # Predict the state of the robot
    # @param  u             =   the input of the system model
    def predict(self, u):
        self.x = newX = self.move(self.x, u)

        F_x = self.F_x(self.x, u)
        F_u = self.F_u(self.x, u)

        # self.x = newX
        self.P = np.dot(F_x, self.P).dot(F_x.T) + \
            np.dot(F_u, self.M).dot(F_u.T)

    def update(self, mes, R):
        # print('X1', self.x)
        super(RobotEKF, self).update(z=mes, HJacobian=self.H_x, Hx=self.h, R=R)
        # print('X2', self.x)

    # Jacobian matrix of the transition function
    # @param  x             =   the state of the robot
    # @param  u             =   the input of the robot
    def F_x(self, x, u):
        return np.matrix([
            [1.0, 0.0, -self.__dt*u[0, 0]*np.sin(x[2, 0]), 0.0],
            [0.0, 1.0, self.__dt*u[0, 0]*np.cos(x[2, 0]), 0.0],
            [0.0, 0.0, 1.0, self.__dt],
            [0.0, 0.0, 0.0, 0.0]
        ])

    # Jcobian matrix of the transition function
    # @param  x             =   the state of the robot
    # @param  u             =   the input of the robot
    def F_u(self, x, u):
        return np.matrix([
            [self.__dt*np.cos(x[2, 0]), 0.0],
            [self.__dt*np.sin(x[2, 0]), 0.0],
            [0.0, 0.0],
            [np.tan(u[1, 0])/self.__w,
             u[0, 0]*(np.tan(u[1, 0])**2+1)/self.__w]
        ])

    def h(self, x):
        return x
        # return x.copy()

    def H_x(self, x):
        return np.matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])


if __name__ == "__main__":
    systemModelTest()
