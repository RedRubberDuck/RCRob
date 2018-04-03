import systemTest
import numpy as np


from matplotlib import pyplot as plt


def generateRefence(std_v, std_alpha, std_x, std_y, std_theta):
    s = 0
    rob1 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=0.1, std_alpha=0.1)
    u = np.matrix([[-0.2], [np.radians(20)]])
    Outputs = rob1.h(rob1.x)
    Inputs = u

    for i in range(200):
        rob1.predict(u)
        out = rob1.h(rob1.x)

        Outputs = np.concatenate((Outputs, out), axis=1)
        Inputs = np.concatenate((Inputs, u), axis=1)

    u = np.matrix([[-0.2], [np.radians(-20)]])
    for i in range(200):
        rob1.predict(u)
        out = rob1.h(rob1.x)

        Outputs = np.concatenate((Outputs, out), axis=1)
        Inputs = np.concatenate((Inputs, u), axis=1)

    Outputs_err = Outputs.copy()
    Inputs_err = Inputs.copy()

    Outputs_err[0, :] = Outputs[0, :] + \
        np.random.normal(loc=0.0, scale=std_x, size=Outputs_err.shape[1])
    Outputs_err[1, :] = Outputs[1, :] + \
        np.random.normal(loc=0.0, scale=std_y, size=Outputs_err.shape[1])

    Outputs_err[2, :] = Outputs[2, :] + \
        np.random.normal(loc=0.0, scale=std_theta, size=Outputs_err.shape[1])

    Inputs_err[0, :] = Inputs[0, :] + \
        np.random.normal(loc=0.0, scale=std_v, size=Inputs.shape[1])

    Inputs_err[1, :] = Inputs[1, :] + \
        np.random.normal(loc=0.0, scale=std_alpha, size=Inputs.shape[1])

    print('Outputs', Outputs[0, :].tolist())
    plt.figure()
    plt.plot(Outputs[0, :].tolist()[0], Outputs[1, :].tolist()[0])
    plt.plot(Outputs_err[0, :].tolist()[0], Outputs_err[1, :].tolist()[0])
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
    Outputs, Inputs, Outputs_err, Inputs_err = generateRefence(std_v=1, std_alpha=np.radians(1), std_x=0.03*np.sqrt(2),
                                                               std_y=0.03*np.sqrt(2), std_theta=np.radians(10))
    rob1 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=1, std_alpha=np.radians(1))
    rob1.P = np.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rob1.x = np.matrix([[0.0], [0.0], [0.0]])
    X = rob1.x.copy()
    for U, Mess in zip(Inputs_err.T, Outputs_err.T):
        rob1.predict(U.T)
        rob1.update(Mess.T, R=np.matrix(
            [[0.03**2*2, 0, 0], [0, 0.03**2*2, 0], [0, 0, np.radians(10)**2]]))
        X = np.concatenate((X, rob1.x.copy()), axis=1)

    plt.figure()
    plt.plot(X[0, :].tolist()[0], X[1, :].tolist()[0])
    plt.plot(Outputs[0, :].tolist()[0], Outputs[1, :].tolist()[0])
    plt.figure()
    plt.plot(X[2, :].tolist()[0])
    plt.plot(Outputs[2, :].tolist()[0])
    plt.show()


if __name__ == '__main__':
    testFilter()
