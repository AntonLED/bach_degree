import numpy as np
from collections import namedtuple


# popt = namedtuple("popt", ["a_w", "b_w", "c_w", "kappa_d", "z_w", "q_w"])

# POPT = popt(
#     2.70515393e06,  # 1 / m^2
#     # -6.64102261e05,  # 1 / m^2
#     0.0,
#     1.17398024e06,  # 1 / m^2
#     -4.01575619e10,  # m
#     3.36475362e-01,  # безразм
#     9.50330424e-15,  # кулон
# )

eps_0 = 8.85418781762039e-12  # vacuum dielectric permitivity
k = 1.0 / (eps_0 * 4.0 * np.pi)
r_D_e = 0.0016622715189364113

TOP_IDX = 0
BOT_IDX = 1

popt = namedtuple("popt", ["a_w", "c_w", "kappa_d", "z_w", "q_w"])
POPT = popt(
    3.03731419e06, 1.14352594e06, -3.84037247e10, 3.80913378e-01, 9.64691720e-15
)


def Ec(x, y, z, q, popt: popt):
    r_ij = np.sqrt(
        np.square(x[1] - x[0]) + np.square(y[1] - y[0]) + np.square(z[1] - z[0])
    )

    Ec_x = np.zeros(2)
    Ec_y = np.zeros(2)
    Ec_z = np.zeros(2)

    Ec_x[1] = (
        np.exp(-r_ij / popt.kappa_d) * (x[1] - x[0]) * k * q[0] / r_ij**3
        + np.exp(-r_ij / popt.kappa_d)
        * (x[1] - x[0])
        * k
        * q[0]
        / r_ij**2
        / popt.kappa_d
    )
    Ec_x[0] = (
        np.exp(-r_ij / popt.kappa_d) * (x[0] - x[1]) * k * q[1] / r_ij**3
        + np.exp(-r_ij / popt.kappa_d)
        * (x[0] - x[1])
        * k
        * q[1]
        / r_ij**2
        / popt.kappa_d
    )

    Ec_y[1] = (
        np.exp(-r_ij / popt.kappa_d) * (y[1] - y[0]) * k * q[0] / r_ij**3
        + np.exp(-r_ij / popt.kappa_d)
        * (y[1] - y[0])
        * k
        * q[0]
        / r_ij**2
        / popt.kappa_d
    )
    Ec_y[0] = (
        np.exp(-r_ij / popt.kappa_d) * (y[0] - y[1]) * k * q[1] / r_ij**3
        + np.exp(-r_ij / popt.kappa_d)
        * (y[0] - y[1])
        * k
        * q[1]
        / r_ij**2
        / popt.kappa_d
    )

    Ec_z[1] = (
        np.exp(-r_ij / popt.kappa_d) * (z[1] - z[0]) * k * q[0] / r_ij**3
        + np.exp(-r_ij / popt.kappa_d)
        * (z[1] - z[0])
        * k
        * q[0]
        / r_ij**2
        / popt.kappa_d
    )
    Ec_z[0] = (
        np.exp(-r_ij / popt.kappa_d) * (z[0] - z[1]) * k * q[1] / r_ij**3
        + np.exp(-r_ij / popt.kappa_d)
        * (z[0] - z[1])
        * k
        * q[1]
        / r_ij**2
        / popt.kappa_d
    )

    return np.array([Ec_x, Ec_y, Ec_z])


def Ew(x, y, z, q, popt: popt):
    Ew_x = np.zeros(2)
    Ew_y = np.zeros(2)
    Ew_z = np.zeros(2)

    Ew_x[TOP_IDX] = 0.0
    Ew_y[TOP_IDX] = 0.0
    Ew_z[TOP_IDX] = 0.0

    r_ij = np.sqrt(
        np.square(x[BOT_IDX] - x[TOP_IDX]) + np.square(y[TOP_IDX] - y[BOT_IDX])
    )

    z_w = -popt.z_w * r_D_e

    coeff = np.exp(
        -popt.a_w * np.square(r_ij)
        - popt.c_w * np.square(z[BOT_IDX] - z[TOP_IDX] + z_w)
    )

    Ew_x[BOT_IDX] = -1 * (
        coeff * k * popt.q_w / r_D_e * (-2 * popt.a_w * (x[BOT_IDX] - x[TOP_IDX]))
    )
    Ew_y[BOT_IDX] = -1 * (
        coeff * k * popt.q_w / r_D_e * (-2 * popt.a_w * (y[BOT_IDX] - y[TOP_IDX]))
    )
    Ew_z[BOT_IDX] = -1 * (
        coeff * k * popt.q_w / r_D_e * (-2 * popt.c_w * (z[BOT_IDX] - z[TOP_IDX] + z_w))
    )

    return np.array([Ew_x, Ew_y, Ew_z])


def calculateAnalitE(x, y, z, q):
    N = x.shape[0]
    E_x = np.zeros(N)
    E_y = np.zeros(N)
    E_z = np.zeros(N)

    E_c = Ec(x, y, z, q, POPT)
    E_w = Ew(x, y, -z, q, POPT)

    E_x, E_y, E_z = E_c + E_w

    return E_x, E_y, E_z


def getWake(x, y, z, q):
    return Ew(x, y, -z, q, POPT)
