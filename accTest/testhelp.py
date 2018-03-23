import numpy as np
import math
import random


def testSemnalGenerate(timestep):
    nrPoint = 40
    accelStart = 20

    vel = 0

    alpha_a = []
    accel_a = []
    vel_a = []
    for i in range(nrPoint):
        alpha = 0.0
        # *np.sin(i/nrPoint*0.5*math.pi+math.pi/2)
        alpha_a.append(alpha)

        if i % (nrPoint) == 0:
            accel_a.append(accelStart)
            accelStart = -0.5 * accelStart
        else:
            accel_a.append(0)
        vel_a.append(vel)
        vel += accel_a[-1]*timestep
    return (alpha_a, accel_a, vel_a)


def generateError(alpha_a, meanAlphaErr, varAlphaEr, accel_a, meanAccEr, varAccEr, vel_a, meanVelEr, varVelEr):
    alpha_a_err = []
    accel_a_err = []
    vel_a_err = []
    for alpha, acc, vel in zip(alpha_a, accel_a, vel_a):
        errAlpha, errAcc, errVel = np.random.normal([meanAlphaErr, meanAccEr, meanVelEr], [
            varAlphaEr, varAccEr, varVelEr])
        alpha_a_err.append(alpha+errAlpha)
        accel_a_err.append(acc+errAcc)
        vel_a_err.append(vel+errVel)

    return (alpha_a_err, accel_a_err, vel_a_err)
