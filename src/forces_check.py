import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calc.calculateAnalitE import calculateAnalitE
from calc.calculateCharge import Charge


k_B = 1.380649e-23  # Boltzmann constant (J/K)
m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
P = 0.25 * 18.131842  # Pressure (Pascal)
r_D_e = 0.0016622715189364113  # m

g = 9.8  # free fall acceleration (m/s^2)
rho = 1500  # mass density of dust particle
steps = 500_000  # number of dust particles time steps
T_p = 300  # Kinetic temperature of dust particles motion
E_x_trap = -1.0 * 1035598  # x-trap (kg/s^2)
E_y_trap = -1.0 * 1035598  # y-trap (kg/s^2)
dt = 5e-5  # integration step for dust particles dynamics, s
r_p = 4.445e-6  # Dust particle radius
m_p = 4.0 / 3.0 * np.pi * r_p**3 * rho


def main():
    folder_path = "data/forces_check/"
    data_path = os.path.join(folder_path, "data.txt")
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    with open(data_path, "w") as output:
        output.write("")

    X_TOP_INIT = 0.0
    X_BOT_INIT = 0.0
    Y_TOP_INIT = 0.0
    Y_BOT_INIT = 0.0
    Z_TOP_INIT = 0.0032248049595676263
    Z_BOT_INIT = 0.002742747106163061

    VX_TOP_INIT = 0.0
    VX_BOT_INIT = 0.0
    VY_TOP_INIT = 0.0
    VY_BOT_INIT = 0.0
    VZ_TOP_INIT = 0.0
    VZ_BOT_INIT = 0.0

    TOP_IDX = 0
    BOT_IDX = 1

    xs = np.array([X_TOP_INIT, X_BOT_INIT])
    ys = np.array([Y_TOP_INIT, Y_BOT_INIT])
    zs = np.array([Z_TOP_INIT, Z_BOT_INIT])
    vxs = np.array([VX_TOP_INIT, VX_BOT_INIT])
    vys = np.array([VY_TOP_INIT, VY_BOT_INIT])
    vzs = np.array([VZ_TOP_INIT, VZ_BOT_INIT])

    charge = Charge()
    qs = charge.calculateLinearCharge(xs / r_D_e, ys / r_D_e, zs / r_D_e)

    for rbot in np.linspace(0, 5 * r_D_e, 1_000):
        xs[BOT_IDX] = rbot

        E_x, E_y, E_z = calculateAnalitE(xs, ys, zs, qs, wake_coeff=1.0)

        fx = E_x * qs
        fy = E_y * qs
        fz = E_z * qs

        with open(data_path, "a") as output:
            output.write(
                f"{rbot}\t{fx[TOP_IDX]}\t{fy[TOP_IDX]}\t{fz[TOP_IDX]}\t{fx[BOT_IDX]}\t{fy[BOT_IDX]}\t{fz[BOT_IDX]}\n"
            )


if __name__ == "__main__":
    main()
