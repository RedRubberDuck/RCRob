from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math


def plotEllipse(meanX, meanY, cov):
    U, s, v = np.linalg.svd(cov)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width = math.sqrt(s[0])
    height = math.sqrt(s[1])

    ell = Ellipse(xy=(meanX, meanY), width=width, height=height,
                  angle=np.degrees(orientation), alpha=0.4)

    return ell
