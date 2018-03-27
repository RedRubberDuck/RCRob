import numpy as np
import math
from matplotlib import pyplot as plt
import testhelp
import plotHelp


class Vechile:
    class State:
        def __init__(self, x=0, y=0, gamma=0):
            self.__X = np.zeros((5, 1))
            self.__X[1, 0] = x
            self.__X[2, 0] = y
            self.__X[4, 0] = gamma

        # Forward speed getter
        @property
        def vf(self):
            return self.__X[0, 0]

        # Forward speed setter
        @vf.setter
        def vf(self, vf):
            self.__X[0, 0] = vf

        # Robot x coordinate
        @property
        def x(self):
            return self.__X[1, 0]

        # Robot x coordinate
        @x.setter
        def x(self, x):
            self.__X[1, 0] = x

        # Robot y coordinate
        @property
        def y(self):
            return self.__X[2, 0]

        # Robot y coordinate
        @y.setter
        def y(self, y):
            self.__X[2, 0] = y

        # Robot angular speed
        @property
        def w(self):
            return self.__X[3, 0]

        # Robot angular speed
        @w.setter
        def w(self, w):
            self.__X[3, 0] = w

        # Robot oriantation
        @property
        def gamma(self):
            return self.__X[4, 0]

        # Robot oriantation
        @gamma.setter
        def gamma(self, gamma):
            self.__X[4, 0] = gamma

        # State
        @property
        def X(self):
            return self.__X

        # State
        @X.setter
        def X(self, X):
            self.__X = X

    class Input:
        def __init__(self, accel_f=0, alpha=0):
            self._U = np.zeros((2, 1))
            self._U[0, 0] = accel_f
            self._U[1, 0] = alpha

        # Forward acceleration
        @property
        def accel(self):
            return self._U[0, 0]

        # Forward acceleration
        @accel.setter
        def accel(self, accel):
            self._U[0, 0] = accel

        # Steering angle
        @property
        def alpha(self):
            return self._U[1, 0]

        # Steering angle
        @alpha.setter
        def alpha(self, alpha):
            self._U[1, 0] = alpha

    class Output:
        def __init__(self, vf=0, w=0):
            self._Y = np.zeros((2, 1))
            self._Y[0, 0] = vf
            self._Y[1, 0] = w

        # Forward speed
        @property
        def vf(self):
            return self._Y[0, 0]

        # Forward speed
        @vf.setter
        def vf(self, vf):
            self._Y[0, 0] = vf

        # Rotation speed
        @property
        def w(self):
            return self._Y[1, 0]

        # Rotation speed
        @w.setter
        def w(self, w):
            self._Y[1, 0] = w

    def __init__(self, wheelbase, timestep, x=0, y=0, gamma=0):
        self.wheelbase = wheelbase
        self.timestep = timestep
        self.timestepPow2 = timestep**2
        self.state = Vechile.State(x, y, gamma)

    # def f(self, f_state, f_input):
    #     newState = Vechile.State()

    #     newState.vf = f_state.vf + self.timestep*f_input.accel
    #     newState.x = f_state.x + \
    #         np.cos(f_state.gamma+f_state.w*self.timestep)*(self.timestep * f_state.vf +
    #                                                        self.timestepPow2 * f_input.accel/2)

    #     newState.y = f_state.y + \
    #         np.sin(f_state.gamma+f_state.w*self.timestep)*(self.timestep * f_state.vf +
    #                                                        self.timestepPow2 * f_input.accel/2)

    #     newState.w = np.tan(np.radians(f_input.alpha)) / \
    #         self.wheelbase*f_state.vf
    #     newState.gamma = f_state.gamma + f_state.w * self.timestep

    #     return newState

    def f(self, f_state, f_input):
        newState = Vechile.State()

        newState.vf = f_state.vf + self.timestep*f_input.accel
        newState.x = f_state.x + \
            np.cos(f_state.gamma)*(self.timestep * f_state.vf +
                                   self.timestepPow2 * f_input.accel/2)

        newState.y = f_state.y + \
            np.sin(f_state.gamma)*(self.timestep * f_state.vf +
                                   self.timestepPow2 * f_input.accel/2)

        newState.w = f_state.vf*np.tan(f_input.alpha) / self.wheelbase
        newState.gamma = f_state.gamma + f_state.w * self.timestep

        return newState

    def h(self, f_state):
        output = Vechile.Output(f_state.vf, f_state.w)
        return output

    # def F_x(self, f_state, f_input):
    #     F_x = np.matrix([[1, 0, 0, 0, 0],
    #                      [math.cos(f_state.gamma+f_state.w*self.timestep)*self.timestep, 1, 0, self.timestep*-1*math.sin(f_state.gamma+f_state.w*self.timestep)
    #                       * (f_state.vf*self.timestep+self.timestepPow2*f_input.accel /
    #                          2), -1*math.sin(f_state.gamma+f_state.w*self.timestep)
    #                       * (f_state.vf*self.timestep+self.timestepPow2*f_input.accel/2)],
    #                      [math.sin(f_state.gamma+f_state.w*self.timestep)*self.timestep, 0, 1, self.timestep*math.cos(f_state.gamma+f_state.w*self.timestep)
    #                       * (f_state.vf*self.timestep+self.timestepPow2*f_input.accel /
    #                          2), math.cos(f_state.gamma+f_state.w*self.timestep)
    #                       * (f_state.vf*self.timestep+self.timestepPow2*f_input.accel/2)],
    #                      [math.tan(np.radians(f_input.alpha)) /
    #                       self.wheelbase, 0, 0, 0, 0],
    #                      [0, 0, 0, self.timestep, 1]])
    #     return F_x

    def F_x(self, f_state, f_input):
        F_x = np.matrix([[1, 0, 0, 0, 0],
                         # -1*math.sin(f_state.gamma)* (f_state.vf*self.timestep+self.timestepPow2*f_input.accel/2)
                         [math.cos(f_state.gamma)*self.timestep, 1, 0, 0, -1*math.sin(f_state.gamma)
                          * (f_state.vf*self.timestep+self.timestepPow2*f_input.accel/2)],
                         #  math.cos(f_state.gamma)* (f_state.vf*self.timestep+self.timestepPow2*f_input.accel/2)
                         [math.sin(f_state.gamma)*self.timestep, 0, 1, 0, math.cos(f_state.gamma)
                          * (f_state.vf*self.timestep+self.timestepPow2*f_input.accel/2)],
                         [math.tan(f_input.alpha)/self.wheelbase, 0, 0, 0, 0],
                         [0, 0, 0, self.timestep, 1]])
        return F_x

    def F_u(self, f_state, f_input):
        s = 0

        F_u = np.matrix([[self.timestep, 0.0],
                         [self.timestepPow2*np.cos(f_state.gamma)/2, 0.0],
                         [self.timestepPow2*np.sin(f_state.gamma)/2, 0.0],
                         [0.0, f_state.vf/self.wheelbase /
                             (np.cos(f_input.alpha)**2)],
                         [0.0, 0.0]])
        return F_u

    # def F_u(self, f_state, f_input):
    #     s = 0

    #     F_u = np.matrix([[self.timestep, 0.0],
    #                      [math.cos(f_state.gamma+f_state.w *
    #                                self.timestep)/2*self.timestepPow2, 0.0],
    #                      [math.sin(f_state.gamma+f_state.w *
    #                                self.timestep)/2*self.timestepPow2, 0.0],
    #                      [0.0, f_state.vf/(math.cos(np.radians(f_input.alpha))
    #                                        ** 2*self.wheelbase)],
    #                      [0.0, 0.0]])
    #     return F_u

    def H(self, state):
        H = np.matrix([[1, 0, 0, 0, 0], [0, 0, 0, 1.0, 0]])
        return H

    # Check the function
