import numpy as np
from scipy.optimize import minimize

from . import routines_math as rout_m

def find_closest_3d_point(m, n):
    def obj_func(x):
        estimated_points = x[:-3, None] * m + n
        res = 0.5 * np.sum((estimated_points - x[None, -3:])**2)

        jac = np.zeros(len(x), dtype=np.float64)
        jac[:-3] = np.sum(m**2, 1) * x[:-3] + \
                   np.sum(m * n, 1) - \
                   np.sum(m * x[None, -3:], 1)
        jac[-3:] = (np.float64(len(x)) - 3.0) * x[-3:] - np.sum(estimated_points, 0)
        return res, jac
    
    nPoints = np.size(m, 0)
    x0 = np.zeros(nPoints + 3)
    tolerance = np.finfo(np.float32).eps
    min_result = minimize(obj_func,
                          x0,
                          method='l-bfgs-b',
                          jac=True,
                          tol=tolerance,
                          options={'disp':False,
                                   'maxcor':20,
                                   'maxfun':15000,
                                   'maxiter':15000,
                                   'maxls':40})
    if not(min_result.success):
        print('WARNING: 3D point interpolation did not converge')
        print('\tnPoints\t', nPoints)
        print('\tsuccess:\t', min_result.success)
        print('\tstatus:\t', min_result.status)
        print('\tmessage:\t',min_result.message)
        print('\tnit:\t', min_result.nit) 
    return min_result.x

# look at: Rational Radial Distortion Models with Analytical Undistortion Formulae, Lili Ma et al.
# source: https://arxiv.org/pdf/cs/0307047.pdf
# only works for k = [k1, k2, 0, 0, 0]
def calc_udst(m_dst, k):
    assert np.all(k[2:] == 0.0), 'ERROR: Undistortion only valid for up to two radial distortion coefficients.'
    x_2 = m_dst[:, 0]
    y_2 = m_dst[:, 1]
    # use r directly instead of c
    nPoints = np.size(m_dst, 0)
    p = np.zeros(6, dtype=np.float64)
    p[4] = 1.0
    sol = np.zeros(3, dtype=np.float64)
    x_1 = np.zeros(nPoints, dtype=np.float64)
    y_1 = np.zeros(nPoints, dtype=np.float64)
    for i_point in range(nPoints):
        cond = (np.abs(x_2[i_point]) > np.abs(y_2[i_point]))
        if (cond):
            c = y_2[i_point] / x_2[i_point]
            p[5] = -x_2[i_point]
        else:
            c = x_2[i_point] / y_2[i_point]
            p[5] = -y_2[i_point]
        p[2] = k[0] * (1 + c**2)
        p[0] = k[1] * (1 + c**2)**2
        sol = np.real(np.roots(p))
        sol_abs = np.abs(sol)
        if (cond):
            x_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            y_1[i_point] = c * x_1[i_point]
        else:
            y_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            x_1[i_point] = c * y_1[i_point]
    m_udst = np.concatenate([[x_1], [y_1], [m_dst[:, 2]]], 0).T
    return m_udst

def calc_3d_point(points_2d, A, k, rX1, tX1):
    mask_nan = ~np.any(np.isnan(points_2d), 1)
    mask_zero = ~np.any((points_2d == 0.0), 1)
    mask = np.logical_and(mask_nan, mask_zero)
    nValidPoints = np.sum(mask)
    if (nValidPoints < 2):
        print("WARNING: Less than 2 valid 2D locations for 3D point interpolation.")
    n = np.zeros((nValidPoints, 3))
    m = np.zeros((nValidPoints, 3))
    nCameras = np.size(points_2d, 0)
    
    index = 0
    for i_cam in range(nCameras):
        if (mask[i_cam]):
            point = np.array([[(points_2d[i_cam, 0] - A[i_cam][0, 2]) / A[i_cam][0, 0],
                               (points_2d[i_cam, 1] - A[i_cam][1, 2]) / A[i_cam][1, 1],
                               1.0]], dtype=np.float64)
            point = calc_udst(point, k[i_cam]).T
            line = point * np.linspace(0, 1, 2)
            RX1 = rout_m.rodrigues2rotMat_single(rX1[i_cam])
            line = np.dot(RX1.T, line - tX1[i_cam].reshape(3, 1))
            n[index] = line[:, 0]
            m[index] = line[:, 1] - line[:, 0]
            index = index + 1
    x = find_closest_3d_point(m, n)
    return x[-3:]
