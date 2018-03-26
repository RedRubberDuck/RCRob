import numpy as np
import math
import random


def generateInputSemnal(timestep):
    nrPoint = 1000
    accelStart = 600

    vel = 0

    alpha_a = []
    accel_a = []
    vel_a = []
    alpha = 20.0
    accel = 0
    alpha_a.append(alpha)
    accel_a.append(accelStart)

    for i in range(nrPoint):
        alpha_a.append(alpha)
        accel_a.append(accel)
        if (i+1) % (nrPoint//2) == 0:
            # accel_a.append(0)
            accel_a[-1] = -4.0*accelStart

            alpha = -5
        # else:
        #     accel_a.append(0)
        vel_a.append(vel)
        vel += accel_a[-1]*timestep
    return (alpha_a, accel_a)


def generateInputError(alpha_a, meanAlphaErr, varAlphaEr, accel_a, meanAccEr, varAccEr):
    alpha_a_err = []
    accel_a_err = []
    # vel_a_err = []
    for alpha, acc in zip(alpha_a, accel_a):
        errAlpha, errAcc = np.random.normal([meanAlphaErr, meanAccEr], [
            varAlphaEr, varAccEr])
        alpha_a_err.append(alpha+errAlpha)
        accel_a_err.append(acc+errAcc)

    return (alpha_a_err, accel_a_err)


def generateMesError(Vf, meanVfErr, varVfErr, W, meanWErr, varWErr):
    Vf_err = []
    W_err = []

    for vf, w in zip(Vf, W):
        errVf, errW = np.random.normal([meanVfErr, meanWErr], [
            varVfErr, varWErr])
        Vf_err.append(vf+errVf)
        W_err.append(w+errW)

    return (Vf_err, W_err)
