import sympy as sm
import numpy as np
from sympy.physics.vector import ReferenceFrame


def getPRVfromDCM(C):
    """
    Return the Principal Rotation Vector (PRV) in terms of rotation angle (phi) and unit vector (e) from a DCM input.
    It should provide rotation angles between -pi and +pi and exclusively about the + rotation axis.
    It should also be capable of handling singular rotations (i.e. rotations of 0, +/-90, or +/-180 about one of the principal axes).
    
    :param C: 3x3 Direction Cosine Matrix representing the rotation.
    :return: e (ndarray), Unit vector representing the axis of rotation.
             phi (float), Rotation angle in radians (between -pi and +pi).
    """
    foo = (1/2)*(np.trace(C) - 1)
    u   = np.array([C[1,2]-C[2,1],
                    C[2,0]-C[0,2],
                    C[0,1]-C[1,0]])
    with np.errstate(all='raise'):
        try:
            n = abs(u/np.linalg.norm(u))
        except:
            eigs = np.linalg.eig(C)
            unique, counts = np.unique(eigs.eigenvalues, return_counts=True)
            d = dict(zip(unique, counts))
            if 1.0 in d and d[1.0] == 1:
                idx = np.where(eigs.eigenvalues == 1.0)[0][0]
                n = eigs.eigenvectors[idx]
            else:
                raise ValueError("Error: The QTY of eigenvectors with eigenvalues equal to unity is not exactly 1. Is your rotation matrix Identity or not right-handed?")
            
    Kn = np.array([[0, -n[2], n[1]],
                   [n[2], 0, -n[0]],
                   [-n[1], n[0], 0]])
    
    phis = np.arcsin(-np.trace(np.matmul(Kn, C))/2)
    phic = np.arccos(foo)
    
    if phis >= 0:
        phi = phic
    else:
        phi = -phic

    e = n
    return e, phi


def getPRVfromEuler(angles, rotOrder):
    """
    Return the Principal Rotation Vector (PRV) in terms of rotation angle (phi) and unit vector (e) from Euler angle inputs (including rotation order).
    This function leverages the sympy.physics.mechanics package to shortcut some of the steps.
    
    :param angles: A tuple (a, b, c) representing the Euler angles (in degrees).
    :param rotOrder: A string indicating the rotation order (e.g., 'XYZ', 'ZYX').
    :return: e (ndarray), Unit vector representing the axis of rotation.
             phi (float), Rotation angle in radians (between -pi and +pi).
    """
    a, b, c = angles
    psi, theta, phi = sm.symbols('psi, theta, phi')
    N = ReferenceFrame('N', indices=('1', '2', '3'))
    B = ReferenceFrame('B', indices=('1', '2', '3'))
    B.orient_body_fixed(N, (psi, theta, phi), rotOrder)
    C = B.dcm(N).subs({psi: np.radians(a),
                       theta: np.radians(b),
                       phi: np.radians(c)})
    return getPRVfromDCM(np.array(C).astype(np.float64))


def addPRVs(phi1, e1, phi2, e2):
    """
    Add two Principal Rotation Vectors (PRVs) together.
    
    :param phi1: Rotation angle for the first PRV (in radians).
    :param e1: Unit vector representing the axis of rotation for the first PRV.
    :param phi2: Rotation angle for the second PRV (in radians).
    :param e2: Unit vector representing the axis of rotation for the second PRV.
    :return: phi (float), The resulting rotation angle (in radians).
             e (ndarray), The resulting unit vector representing the axis of rotation.
    """
    sp1 = np.sin(phi1 / 2)
    sp2 = np.sin(phi2 / 2)
    cp1 = np.cos(phi1 / 2)
    cp2 = np.cos(phi2 / 2)
    phi = 2 * np.arccos(cp1 * cp2 - sp1 * sp2 * np.dot(e1, e2))
    e = (cp2 * sp1 * e1 + cp1 * sp2 * e2 + sp1 * sp2 * np.cross(e1, e2)) / np.sin(phi / 2)
    return phi, e


def subPRVs(phi, e, phi1, e1):
    """
    Subtract one Principal Rotation Vector (PRV) from another.
    
    :param phi: Rotation angle for the resulting PRV (in radians).
    :param e: Unit vector representing the axis of rotation for the resulting PRV.
    :param phi1: Rotation angle for the PRV to be subtracted (in radians).
    :param e1: Unit vector representing the axis of rotation for the PRV to be subtracted.
    :return: phi2 (float), The resulting rotation angle (in radians).
             e2 (ndarray), The resulting unit vector representing the axis of rotation.
    """
    sp1 = np.sin(phi1 / 2)
    sp = np.sin(phi / 2)
    cp1 = np.cos(phi1 / 2)
    cp = np.cos(phi / 2)
    phi2 = 2 * np.arccos(cp * cp1 + sp * sp1 * np.dot(e, e1))
    e2 = (cp1 * sp * e - cp * sp1 * e1 + sp * sp1 * np.cross(e, e1)) / np.sin(phi2 / 2)
    return phi2, e2


def tilde(x):
    """
    Returns the skew-symmetric matrix (also known as the "tilde" matrix) of a vector x.
    The skew-symmetric matrix is used for cross products in vector algebra.

    :param x: A 3-element vector (array-like) representing the vector to be converted into a skew-symmetric matrix.
    :return: x_tilde (ndarray), A 3x3 skew-symmetric matrix corresponding to the input vector.
    """
    x = np.array(x).reshape(3)
    x_tilde = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return x_tilde


def untilde(x_tilde):
    """
    Recovers a vector from its skew-symmetric matrix (tilde matrix).
    This function extracts the original vector by reversing the skew-symmetric transformation.

    :param x_tilde: A 3x3 skew-symmetric matrix.
    :return: x (ndarray), A 3-element vector recovered from the skew-symmetric matrix.
    """
    x = np.zeros(3)
    x[0] = x_tilde[2, 1]
    x[1] = x_tilde[0, 2]
    x[2] = x_tilde[1, 0]
    return x


def rotated(dim, deg):
    """
    Returns the Direction Cosine Matrix (DCM) for a rotation of `deg` degrees about the `dim` axis.
    This is a convenience function that calls the rotate function with the given parameters.
    
    :param dim: The axis about which to rotate, represented as an integer: 
                1 for x-axis, 2 for y-axis, and 3 for z-axis.
    :param deg: The rotation angle in degrees.
    :return: R (ndarray), The 3x3 Direction Cosine Matrix representing the rotation.
    """
    rad = np.deg2rad(deg)
    return rotate(dim, rad)


def rotate(dim, rad):
    """
    Returns the Direction Cosine Matrix (DCM) for a rotation of `rad` radians about the `dim` axis.
    
    :param dim: The axis about which to rotate, represented as an integer: 
                1 for x-axis, 2 for y-axis, and 3 for z-axis.
    :param rad: The rotation angle in radians.
    :return: R (ndarray), The 3x3 Direction Cosine Matrix representing the rotation.
    """
    orders = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
    r = np.array([[1, 0, 0], [0, np.cos(rad), np.sin(rad)], [0, -np.sin(rad), np.cos(rad)]])
    R = r[orders[dim-1] - 1][:, orders[dim-1] - 1]
    return R


def dcm_2_quat(r):
    """
    Convert a Direction Cosine Matrix (DCM) to its equivalent quaternion rotation.
    
    :param r: 3x3 Direction Cosine Matrix representing the rotation.
    :return: ep (list), Quaternion representing the same rotation as the DCM.
    """
    b0 = np.sqrt(1/4*(1+np.trace(r)))
    b1 = np.sqrt(1/4*max([(1-np.trace(r)+2*r[0][0]),0]))
    b2 = np.sqrt(1/4*max([(1-np.trace(r)+2*r[1][1]),0]))
    b3 = np.sqrt(1/4*max([(1-np.trace(r)+2*r[2][2]),0]))
    s1 = r[1][2]-r[2][1]
    b1 = np.sign(s1)*b1
    s2 = r[2][0]-r[0][2]
    b2 = np.sign(s2)*b2
    s3 = r[0][1]-r[1][0]
    b3 = np.sign(s3)*b3
    ep = [b0, b1, b2, b3]
    return ep


def dcmdot_2_omega(dcm1, dcm0, dt):
    """
    Convert the time derivative of a Direction Cosine Matrix (DCM) to angular velocity (omega).
    
    :param dcm1: Current Direction Cosine Matrix.
    :param dcm0: Previous Direction Cosine Matrix.
    :param dt: Time step between dcm1 and dcm0.
    :return: omega (ndarray), Angular velocity vector.
    """
    dcm_dot = (dcm1 - dcm0) / dt
    return untilde(-dcm_dot @ dcm1.transpose())


def quat_2_dcm(q):
    """
    Convert a quaternion to its equivalent Direction Cosine Matrix (DCM).
    
    :param q: A list or array representing the quaternion (b0, b1, b2, b3).
    :return: dcm (ndarray), 3x3 Direction Cosine Matrix representing the same rotation as the quaternion.
    """
    dcm = [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] + q[0]*q[3]), 2*(q[1]*q[3] - q[0]*q[2]),
           2*(q[1]*q[2] - q[0]*q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*(q[2]*q[3] + q[0]*q[1]),
           2*(q[1]*q[3] + q[0]*q[2]), 2*(q[2]*q[3] - q[0]*q[1]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]
    return np.array(dcm).reshape(3, 3)


def quat_add(qs, q2):
    """
    Add two quaternions together to perform quaternion rotation.
    Rotates attitude `qs` by `q2` to achieve the final attitude `qf`.
    
    :param qs: The initial quaternion attitude.
    :param q2: The quaternion that is applied to `qs` to get the final attitude.
    :return: qf (ndarray), The final quaternion after applying the rotation.
    """
    qs = np.array(qs).reshape(4, 1)
    m = np.array([[q2[0], -q2[1], -q2[2], -q2[3]],
                  [q2[1], q2[0], q2[3], -q2[2]],
                  [q2[2], -q2[3], q2[0], q2[1]],
                  [q2[3], q2[2], -q2[1], q2[0]],
                  ])
    qf = m @ qs
    return qf


def quat_diff(qs, qf):
    """
    Compute the quaternion difference (the relative rotation) that transforms `qs` into `qf`.
    
    :param qs: The initial quaternion attitude.
    :param qf: The final quaternion attitude.
    :return: q2 (ndarray), The quaternion representing the relative rotation between `qs` and `qf`.
    """
    m = np.array([[qs[0], -qs[1], -qs[2], -qs[3]],
                  [qs[1], qs[0], -qs[3], qs[2]],
                  [qs[2], qs[3], qs[0], -qs[1]],
                  [qs[3], -qs[2], qs[1], qs[0]],
                  ])
    q2 = m.transpose() @ qf
    return q2


def quat_revert(qf, q2):
    """
    Undo the rotation of quaternion `q2` from `qf` to return to the original attitude `qs`.
    
    :param qf: The final quaternion attitude after the rotation.
    :param q2: The quaternion representing the rotation to be undone.
    :return: qs (ndarray), The original quaternion attitude before the rotation.
    """
    return quat_add(qf, q2 * np.array([1, -1, -1, -1]))


def quat_omega_DKE(q, w):
    """
    Get quaternion rate of change from body angular velocity (omega) and current quaternion attitude.
    
    :param q: The current quaternion representing the attitude.
    :param w: The angular velocity vector in the body frame.
    :return: q_dot (ndarray), The rate of change of the quaternion.
    """
    m = np.array([[-q[1], -q[2], -q[3]],
                  [q[0], -q[3], q[2]],
                  [q[3], q[0], -q[1]],
                  [-q[2], q[1], q[0]],
                  ])
    q_dot = 0.5 * m @ w
    return q_dot


def quatdot_2_omega(q, qdot):
    """
    Convert quaternion rate of change (qdot) to angular velocity (omega).
    
    :param q: The current quaternion representing the attitude.
    :param qdot: The quaternion rate of change.
    :return: omega (ndarray), The angular velocity vector.
    """
    m = np.array([[q[0], -q[1], -q[2], -q[3]],
                  [q[1], q[0], -q[3], q[2]],
                  [q[2], q[3], q[0], -q[1]],
                  [q[3], -q[2], q[1], q[0]],
                  ])
    omega = 2 * m.transpose() @ qdot
    return omega


def normalized_integrator(qdot, q0, t0, tf, dt):
    """
    Integrate quaternion rates to obtain the new attitude quaternion using numerical integration.
    
    :param qdot: The function that returns the quaternion rate of change.
    :param q0: The initial quaternion attitude.
    :param t0: The initial time.
    :param tf: The final time.
    :param dt: The time step for the integration.
    :return: q (ndarray), The array of quaternion attitudes at each time step.
    """
    length = int((tf - t0) / dt) + 1
    q = np.zeros([length, 4])
    q[0, :] = np.array([q0])
    for i, t in enumerate(np.arange(t0, tf, dt)):
        q_new = q[i] + qdot(q[i], t) * dt
        q[i + 1] = q_new / np.linalg.norm(q_new)
    return q


def rotate(dim, rad):
    """
    Returns the Direction Cosine Matrix (DCM) for a rotation of `rad` radians about the `dim` axis.
    
    :param dim: The axis about which to rotate, represented as an integer: 
                1 for x-axis, 2 for y-axis, and 3 for z-axis.
    :param rad: The rotation angle in radians.
    :return: R (ndarray), The 3x3 Direction Cosine Matrix representing the rotation.
    """
    orders = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
    r = np.array([[1, 0, 0], [0, np.cos(rad), np.sin(rad)], [0, -np.sin(rad), np.cos(rad)]])
    R = r[orders[dim-1] - 1][:, orders[dim-1] - 1]
    return R


def eulerRad_2_dcm(dims, rads):
    """
    Returns the Direction Cosine Matrix (DCM) equivalent of a set of Euler angles in radians.
    
    :param dims: A list of integers representing the axes of rotation in the Euler angle order: 
                 1 for x-axis, 2 for y-axis, and 3 for z-axis.
    :param rads: A list or array of rotation angles in radians, corresponding to each axis in `dims`.
    :return: R (ndarray), The 3x3 Direction Cosine Matrix representing the total rotation.
    """
    R = np.eye(3)
    for i in range(len(dims)):
        R = rotate(dims[i], rads[i]).dot(R)
    return R


def eulerDeg_2_dcm(dims, deg):
    """
    Returns the Direction Cosine Matrix (DCM) equivalent of a set of Euler angles in degrees.
    
    :param dims: A list of integers representing the axes of rotation in the Euler angle order: 
                 1 for x-axis, 2 for y-axis, and 3 for z-axis.
    :param deg: A list or array of rotation angles in degrees, corresponding to each axis in `dims`.
    :return: R (ndarray), The 3x3 Direction Cosine Matrix representing the total rotation.
    """
    R = eulerRad_2_dcm(dims, np.deg2rad(deg))
    return R