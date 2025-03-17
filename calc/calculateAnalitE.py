import numpy as np
from collections import namedtuple


popt = namedtuple("popt", ["a_w", "b_w", "c_w", "kappa_d", "z_w", "q_w"])

POPT = popt(
    2.70515393e06,  # 1 / m^2
    -6.64102261e05,  # 1 / m^2
    1.17398024e06,  # 1 / m^2
    -4.01575619e10,  # m
    3.36475362e-01,  # безразм
    9.50330424e-15,  # кулон
)

eps_0 = 8.85418781762039e-12  # vacuum dielectric permitivity
k = 1.0 / (eps_0 * 4.0 * np.pi)
r_D_e = 0.0016622715189364113


def newEc(x, y, z, q, popt: popt):
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


def Ec(x, y, z, q, popt: popt):
    N = x.shape[0]
    Ec_x = np.zeros(N)
    Ec_y = np.zeros(N)
    Ec_z = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = np.sqrt(
                    (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
                )

                Ec_x[i] += (
                    np.exp(-r_ij / popt.kappa_d) * (x[i] - x[j]) * k * q[j] / r_ij**3
                    + np.exp(-r_ij / popt.kappa_d)
                    * (x[i] - x[j])
                    * k
                    * q[j]
                    / r_ij**2
                    / popt.kappa_d
                )
                Ec_y[i] += (
                    np.exp(-r_ij / popt.kappa_d) * (y[i] - y[j]) * k * q[j] / r_ij**3
                    + np.exp(-r_ij / popt.kappa_d)
                    * (y[i] - y[j])
                    * k
                    * q[j]
                    / r_ij**2
                    / popt.kappa_d
                )
                Ec_z[i] += (
                    np.exp(-r_ij / popt.kappa_d) * (z[i] - z[j]) * k * q[j] / r_ij**3
                    + np.exp(-r_ij / popt.kappa_d)
                    * (z[i] - z[j])
                    * k
                    * q[j]
                    / r_ij**2
                    / popt.kappa_d
                )

    return np.array([Ec_x, Ec_y, Ec_z])


def newEw(x, y, z, q, popt: popt):
    Ew_x = np.zeros(2)
    Ew_y = np.zeros(2)
    Ew_z = np.zeros(2)

    top_idx = 0
    bot_idx = 1

    Ew_x[top_idx] = 0.0
    Ew_y[top_idx] = 0.0
    Ew_z[top_idx] = 0.0

    r_ij = np.sqrt(
        np.square(x[bot_idx] - x[top_idx]) + np.square(y[bot_idx] - y[top_idx])
    )

    dz0 = 0.0004820578534045655

    # z_w = (1 - popt.z_w) * r_D_e
    # z_w = popt.z_w
    z_w = popt.z_w * r_D_e
    # z_w = z[top_idx] - 0.1 * dz0

    coeff = np.exp(
        -popt.a_w * np.square(r_ij)
        - 2 * popt.b_w * r_ij * (z[bot_idx] - z[top_idx] + z_w)
        - popt.c_w * np.square(z[bot_idx] - z[top_idx] + z_w)
    )

    if r_ij != 0:
        Ew_x[bot_idx] = -1 * (
            coeff
            * k
            * popt.q_w
            / r_D_e
            * (
                -2 * popt.a_w * (x[bot_idx] - x[top_idx])
                - 2
                * popt.b_w
                * (z[bot_idx] - z[top_idx] + z_w)
                * (x[bot_idx] - x[top_idx])
                / r_ij
            )
        )
        Ew_y[bot_idx] = -1 * (
            coeff
            * k
            * popt.q_w
            / r_D_e
            * (
                -2 * popt.a_w * (y[bot_idx] - y[top_idx])
                - 2
                * popt.b_w
                * (z[bot_idx] - z[top_idx] + z_w)
                * (y[bot_idx] - y[top_idx])
                / r_ij
            )
        )
        Ew_z[bot_idx] = -1 * (
            coeff
            * k
            * popt.q_w
            / r_D_e
            * (-2 * popt.b_w * r_ij - 2 * popt.c_w * (z[bot_idx] - z[top_idx] + z_w))
        )
    else:
        Ew_x[bot_idx] = 0.0
        Ew_y[bot_idx] = 0.0
        Ew_z[bot_idx] = -1 * (
            coeff
            * k
            * popt.q_w
            / r_D_e
            * (-2 * popt.c_w * (z[bot_idx] - z[top_idx] + z_w))
        )

    return np.array([Ew_x, Ew_y, Ew_z])


def Ew(x, y, z, q, popt: popt):
    N = x.shape[0]
    Ew_x = np.zeros(N)
    Ew_y = np.zeros(N)
    Ew_z = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                if i == 0:
                    r_i = np.sqrt(np.square(x[i]) + np.square(y[i]))

                    exp_coeff = -(
                        k
                        * popt.q_w
                        * np.exp(
                            -popt.a_w * np.square(r_i)
                            - 2 * popt.b_w * r_i * (z[i] - popt.z_w)
                            - popt.c_w * np.square(z[i] - popt.z_w)
                        )
                        / r_D_e
                    )

                    Ew_x[i] += exp_coeff * (
                        -2 * popt.a_w * x[i]
                        - 2 * popt.b_w * (z[i] - popt.z_w) * x[i] / r_i
                    )
                    Ew_y[i] += exp_coeff * (
                        -2 * popt.a_w * y[i]
                        - 2 * popt.b_w * (z[i] - popt.z_w) * y[i] / r_i
                    )
                    Ew_z[i] += exp_coeff * (
                        -2 * popt.b_w * r_i - 2 * popt.c_w * (z[i] - popt.z_w)
                    )
                else:
                    Ew_x[i] = 0.0
                    Ew_y[i] = 0.0
                    Ew_z[i] = 0.0

    return np.array([Ew_x, Ew_y, Ew_z])


def calculateAnalitE(x, y, z, q, _is_print=False):
    N = x.shape[0]
    E_x = np.zeros(N)
    E_y = np.zeros(N)
    E_z = np.zeros(N)

    E_c = newEc(x, y, z, q, POPT)
    E_w = newEw(x, y, z, q, POPT)

    if _is_print:
        print(f"c : {E_c},  w : {E_w}")

    E_x, E_y, E_z = E_c + E_w

    return E_x, E_y, E_z


def getWake(x, y, z, q):
    return newEw(x, y, z, q, POPT)
