from filterpy.kalman import ExtendedKalmanFilter as EKF
import numpy as np


class RobotEKF(EKF):
    # Robot states
    # X = [px,py,theta,dtheta] - px and py is the robot position[m], theta is the robot oriantation[rad], dtheta is the robot angular velocity [rad/s]
    # U = [v,alpha] - v is the robot forward speed [m/s] and alpha is the steering angle [rad] (trigonometry oriantation)
    # Y = [dtheta] - angular velocity

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
        return x[3, 0]
        # return x.copy()

    def H_x(self, x):
        return np.matrix([[0.0, 0.0, 0.0, 1.0]])
        # return np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def robPosition(self):
        return np.matrix([[self.x[0, 0]], [self.x[1, 0]]])

    def robOriantation(self):
        return np.matrix([[self.x[2, 0]]])
