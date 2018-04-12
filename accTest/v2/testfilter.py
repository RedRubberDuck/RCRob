import systemTest
import numpy as np


from matplotlib import pyplot as plt


def generateRefence(std_v, std_alpha, std_px, std_py, std_theta, std_dtheta):
    #  std_x, std_y,
    s = 0
    rob1 = systemTest.RobotEKF(
        wheelbase=0.265, dt=0.05, std_v=0.1, std_alpha=0.1)
    u = np.matrix([[0.2], [np.radians(23)]])
    Outputs = rob1.h(rob1.x)
    Inputs = u

    for i in range(440):
        rob1.predict(u)
        out = rob1.h(rob1.x)

        Outputs = np.concatenate((Outputs, out), axis=1)
        Inputs = np.concatenate((Inputs, u), axis=1)

    # u = np.matrix([[0.30], [np.radians(-23)]])
    # for i in range(200):
    #     rob1.predict(u)
    #     out = np.matrix([[rob1.h(rob1.x)]])
    #
    #     Outputs = np.concatenate((Outputs, out), axis=1)
    #     Inputs = np.concatenate((Inputs, u), axis=1)

    Outputs_err = Outputs.copy()
    Inputs_err = Inputs.copy()

    # Outputs_err[0, :] = Outputs[0, :] + \
    #     np.random.normal(loc=0.0, scale=std_x, size=Outputs_err.shape[1])
    # Outputs_err[1, :] = Outputs[1, :] + \
    #     np.random.normal(loc=0.0, scale=std_y, size=Outputs_err.shape[1])

    Outputs_err[0, :] = Outputs[0, :] + \
        np.random.normal(loc=0.0, scale=std_px, size=Outputs_err.shape[1])
    Outputs_err[1, :] = Outputs[1, :] + \
        np.random.normal(loc=0.0, scale=std_py, size=Outputs_err.shape[1])
    Outputs_err[2, :] = Outputs[2, :] + \
        np.random.normal(loc=0.0, scale=std_theta, size=Outputs_err.shape[1])
    Outputs_err[3, :] = Outputs[3, :] + \
        np.random.normal(loc=0.0, scale=std_dtheta, size=Outputs_err.shape[1])

    Inputs_err[0, :] = Inputs[0, :] + \
        np.random.normal(loc=0.0, scale=std_v, size=Inputs.shape[1])

    Inputs_err[1, :] = Inputs[1, :] + \
        np.random.normal(loc=np.radians(2), scale=std_alpha, size=Inputs.shape[1])

    # plt.figure()
    # plt.subplot(411)
    # plt.plot(Outputs[0, :].tolist()[0])
    # plt.plot(Outputs_err[0, :].tolist()[0])
    # plt.title('x')
    # plt.subplot(412)
    # plt.plot(Outputs[1, :].tolist()[0])
    # plt.plot(Outputs_err[1, :].tolist()[0])
    # plt.title('y')
    # plt.subplot(413)
    # plt.plot(Outputs[2, :].tolist()[0])
    # plt.plot(Outputs_err[2, :].tolist()[0])
    # plt.title('theta')
    # plt.subplot(414)
    # plt.plot(Outputs[3, :].tolist()[0])
    # plt.plot(Outputs_err[3, :].tolist()[0])
    # plt.title('dtheta')
    # plt.legend(['init', 'error'])
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(Inputs[0, :].tolist()[0])
    # plt.plot(Inputs_err[0, :].tolist()[0])
    # plt.legend(['init', 'error'])
    # plt.subplot(212)
    # plt.plot(Inputs[1, :].tolist()[0])
    # plt.plot(Inputs_err[1, :].tolist()[0])
    # plt.legend(['init', 'error'])
    # plt.show()

    return Outputs, Inputs, Outputs_err, Inputs_err


def testFilter():

    # std_v, std_alpha, std_px, std_py, std_theta, std_dtheta
    Outputs, Inputs, Outputs_err, Inputs_err = generateRefence(
        std_v=0.05, std_alpha=np.radians(0.0), std_px=0.1, std_py=0.1, std_theta=np.radians(10.0), std_dtheta=np.radians(1.0))

    rob_flt = systemTest.RobotEKF(
        wheelbase=0.265, dt=0.05, std_v=0.01, std_alpha=np.radians(2.0))
    rob_flt.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob_flt.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])
    X = rob_flt.x.copy()
    X_SimErr = rob_flt.x.copy()
    X_non = rob_flt.x.copy()

    rob2 = systemTest.RobotEKF(wheelbase=0.265, dt=0.05, std_v=1, std_alpha=np.radians(1))
    rob2.P = rob_flt.P.copy()
    rob2.x = rob_flt.x.copy()

    rob = systemTest.RobotEKF(wheelbase=0.265, dt=0.05, std_v=1, std_alpha=np.radians(1))
    rob.P = rob_flt.P.copy()
    rob.x = rob_flt.x.copy()

    SimErr = np.abs(X_non - X_SimErr)
    FltErr = np.abs(X_non - X)
    #
    index = 0
    for U_err, Mess_err, U, Mess in zip(Inputs_err.T, Outputs_err.T, Inputs.T, Outputs.T):

        rob_flt.predict(U_err.T)

        if(index % 20 == 0):
            rob_flt.update(Mess_err.T,
                           R=np.matrix([
                               [0.1**2, 0.0, 0.0, 0.0],
                               [0.0, 0.1**2, 0.0, 0.0],
                               [0.0, 0.0, np.radians(10)**2, 0.0],
                               [0.0, 0.0, 0.0, np.radians(1.0)**2]
                           ]))
        else:
            Mess_temp = rob_flt.x.copy()
            Mess_temp[0, 0] = 0.0
            Mess_temp[1, 0] = 0.0
            Mess_temp[2, 0] = 0.0
            Mess_temp[3, 0] = Mess_err.T[3, 0]
            rob_flt.update(Mess_temp,
                           R=np.matrix([
                               [10.0**2, 0.0, 0.0, 0.0],
                               [0.0, 10.0**2, 0.0, 0.0],
                               [0.0, 0.0, 10.0**2, 0.0],
                               [0.0, 0.0, 0.0, np.radians(1.0)**2]
                           ]))
        X = np.concatenate((X, rob_flt.x.copy()), axis=1)
        #-----------------------------------------------------------------------
        rob2.predict(U_err.T)
        X_SimErr = np.concatenate((X_SimErr, rob2.x.copy()), axis=1)

        fltErr = Mess.T - rob_flt.x
        simErr = Mess.T - rob2.x
        # print(Mess)

        SimErr = np.concatenate((SimErr, simErr), axis=1)
        FltErr = np.concatenate((FltErr, fltErr), axis=1)

        index += 1
    legend = ['Initial', 'Filtered', 'Simulated_Error', 'Mess']

    plt.figure()
    plt.subplot(411)
    plt.plot(Outputs[0, :].tolist()[0])
    plt.plot(X[0, :].tolist()[0])
    plt.plot(X_SimErr[0, :].tolist()[0])
    plt.plot(Outputs_err[0, :].tolist()[0])
    plt.legend(legend)
    plt.title('x')
    plt.subplot(412)
    plt.plot(Outputs[1, :].tolist()[0])
    plt.plot(X[1, :].tolist()[0])
    plt.plot(X_SimErr[1, :].tolist()[0])
    plt.plot(Outputs_err[1, :].tolist()[0])
    plt.legend(legend)
    plt.title('y')
    plt.subplot(413)
    plt.plot(Outputs[2, :].tolist()[0])
    plt.plot(X[2, :].tolist()[0])
    plt.plot(X_SimErr[2, :].tolist()[0])
    plt.plot(Outputs_err[2, :].tolist()[0])
    plt.legend(legend)
    plt.title('theta')
    plt.subplot(414)
    plt.plot(Outputs[3, :].tolist()[0])
    plt.plot(X[3, :].tolist()[0])
    plt.plot(X_SimErr[3, :].tolist()[0])
    plt.plot(Outputs_err[3, :].tolist()[0])
    plt.legend(legend)
    plt.title('dtheta')

    legend = ['SimErr', 'FltErr']
    plt.figure()
    plt.subplot(411)
    plt.plot(SimErr[0, :].tolist()[0])
    plt.plot(FltErr[0, :].tolist()[0])

    plt.legend(legend)
    plt.title('x')
    plt.subplot(412)
    plt.plot(SimErr[1, :].tolist()[0])
    plt.plot(FltErr[1, :].tolist()[0])

    plt.legend(legend)
    plt.title('y')
    plt.subplot(413)
    plt.plot(SimErr[2, :].tolist()[0])
    plt.plot(FltErr[2, :].tolist()[0])

    plt.legend(legend)
    plt.title('theta')
    plt.subplot(414)
    plt.plot(SimErr[3, :].tolist()[0])
    plt.plot(FltErr[3, :].tolist()[0])

    plt.legend(legend)
    plt.title('dtheta')
    plt.show()


if __name__ == '__main__':
    testFilter()
