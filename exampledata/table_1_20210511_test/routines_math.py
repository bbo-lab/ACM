import configuration as cfg
import numpy as np

def rodrigues2rotMat_single(r):
    if np.all(abs(r) <= cfg.num_tol):
        rotMat = np.identity(3, dtype=np.float64)
    else:
        sqrt_arg = np.sum(r**2)
        theta = np.sqrt(sqrt_arg)
        u = r / theta
        # row 1
        rotMat_00 = np.cos(theta) + u[0]**2 * (1.0 - np.cos(theta))
        rotMat_01 = u[0] * u[1] * (1.0 - np.cos(theta)) - u[2] * np.sin(theta)
        rotMat_02 = u[0] * u[2] * (1.0 - np.cos(theta)) + u[1] * np.sin(theta)
        # row 2
        rotMat_10 = u[0] * u[1] * (1.0 - np.cos(theta)) + u[2] * np.sin(theta)
        rotMat_11 = np.cos(theta) + u[1]**2 * (1 - np.cos(theta))
        rotMat_12 = u[1] * u[2] * (1.0 - np.cos(theta)) - u[0] * np.sin(theta)
        # row 3
        rotMat_20 = u[0] * u[2] * (1.0 - np.cos(theta)) - u[1] * np.sin(theta)
        rotMat_21 = u[1] * u[2] * (1.0 - np.cos(theta)) + u[0] * np.sin(theta)
        rotMat_22 = np.cos(theta) + u[2]**2 * (1.0 - np.cos(theta))
        # output
        rotMat = np.array([[rotMat_00, rotMat_01, rotMat_02],
                           [rotMat_10, rotMat_11, rotMat_12],
                           [rotMat_20, rotMat_21, rotMat_22]],
                          dtype=np.float64)
    return rotMat

def rotMat2rodrigues_single(R):
    if (abs(np.trace(R) - 3.0) <= cfg.num_tol):
        r = np.zeros(3, dtype=np.float64)
    else:
        theta_norm = np.arccos((np.trace(R) - 1.0) / 2.0)
        r = theta_norm / (2.0 * np.sin(theta_norm)) *  \
            np.array([R[2,1] - R[1,2],
                      R[0,2] - R[2,0],
                      R[1,0] - R[0,1]],
                     dtype=np.float64)
    return r
