import systemTest
import numpy as np


from matplotlib import pyplot as plt


def generateRefence(std_v, std_alpha, std_dtheta):
    #  std_x, std_y,
    s = 0
    rob1 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=0.1, std_alpha=0.1)
    u = np.matrix([[0.2], [np.radians(23)]])
    Outputs = np.matrix([[rob1.h(rob1.x)]])
    print('Outputs', Outputs)
    Inputs = u

    for i in range(235):
        rob1.predict(u)
        out = np.matrix([[rob1.h(rob1.x)]])

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
        np.random.normal(loc=0.0, scale=std_dtheta, size=Outputs_err.shape[1])

    Inputs_err[0, :] = Inputs[0, :] + \
        np.random.normal(loc=0.0, scale=std_v, size=Inputs.shape[1])

    Inputs_err[1, :] = Inputs[1, :] + \
        np.random.normal(loc=np.radians(5), scale=std_alpha, size=Inputs.shape[1])

    plt.figure()
    plt.plot(Outputs[0, :].tolist()[0])
    plt.plot(Outputs_err[0, :].tolist()[0])
    plt.figure()
    plt.subplot(211)
    plt.plot(Inputs[0, :].tolist()[0])
    plt.plot(Inputs_err[0, :].tolist()[0])
    plt.subplot(212)
    plt.plot(Inputs[1, :].tolist()[0])
    plt.plot(Inputs_err[1, :].tolist()[0])
    plt.show()

    return Outputs, Inputs, Outputs_err, Inputs_err


def testFilter():
    # std_x=0.03*np.sqrt(2), std_y=0.03*np.sqrt(2),
    Outputs, Inputs, Outputs_err, Inputs_err = generateRefence(
        std_v=0.001, std_alpha=np.radians(0.001), std_dtheta=np.radians(0.2))
    rob1 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=0.001, std_alpha=np.radians(5.001))
    rob1.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob1.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])
    X = rob1.x.copy()
    X_err = rob1.x.copy()
    X_non = rob1.x.copy()

    rob2 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=1, std_alpha=np.radians(1))
    rob2.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob2.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])

    rob3 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=1, std_alpha=np.radians(1))
    rob3.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob3.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])

    for U_err, Mess_err, U in zip(Inputs_err.T, Outputs_err.T, Inputs.T):
        rob1.predict(U_err.T)
        rob1.update(Mess_err.T, R=np.matrix(
            [[np.radians(0.2)**2]]))
        X = np.concatenate((X, rob1.x.copy()), axis=1)

        rob2.predict(U_err.T)
        X_err = np.concatenate((X_err, rob2.x.copy()), axis=1)

        rob3.predict(U.T)
        X_non = np.concatenate((X_non, rob3.x.copy()), axis=1)
    #
    plt.figure()
    plt.plot(X[0, :].tolist()[0], X[1, :].tolist()[0])
    plt.plot(X_err[0, :].tolist()[0], X_err[1, :].tolist()[0])
    plt.plot(X_non[0, :].tolist()[0], X_non[1, :].tolist()[0])

    plt.legend(['Filtered', 'Error', 'NonError'])
    plt.figure()
    plt.plot(X[3, :].tolist()[0])
    plt.plot(X_err[3, :].tolist()[0])
    plt.plot(X_non[3, :].tolist()[0])
    plt.plot(Outputs_err[0, :].tolist()[0], '--')
    plt.legend(['Filtered', 'Error__', 'NonError', 'Error__1'])
    plt.show()


if __name__ == '__main__':
    testFilter()
