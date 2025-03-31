import os
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calc.calculateAnalitE import calculateAnalitE
from calc.calculateCharge import Charge
from calc.fit import fit


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


# 36 / 100

if len(sys.argv) > 1:
    folder_path = "data/dynamics/"
    file_path = os.path.join(folder_path, f"traj_{sys.argv[1]}.xyz")
    data_path = os.path.join(folder_path, f"data_{sys.argv[1]}.txt")
    Path(folder_path).mkdir(parents=True, exist_ok=True)

else:
    folder_path = "data/dynamics/"
    file_path = os.path.join(folder_path, "traj.xyz")
    data_path = os.path.join(folder_path, "data.txt")
    Path(folder_path).mkdir(parents=True, exist_ok=True)

with open(file_path, "w") as output:
    output.write("")

with open(data_path, "w") as output:
    output.write("")

# initial particles' parameters
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

# strange magic
charge = Charge()

# dynamics setup
timestamp = 0.0

xs = np.array([X_TOP_INIT, X_BOT_INIT])
ys = np.array([Y_TOP_INIT, Y_BOT_INIT])
zs = np.array([Z_TOP_INIT, Z_BOT_INIT])
vxs = np.array([VX_TOP_INIT, VX_BOT_INIT])
vys = np.array([VY_TOP_INIT, VY_BOT_INIT])
vzs = np.array([VZ_TOP_INIT, VZ_BOT_INIT])
qs = charge.calculateLinearCharge(xs / r_D_e, ys / r_D_e, zs / r_D_e)

E_x, E_y, E_z = calculateAnalitE(xs, ys, zs, qs)
E_trap_to_approximate = m_p * g / qs - E_z
E_0, alpha = fit(zs, E_trap_to_approximate)

for step in range(steps):
    # calculate random temperature-related force
    v_T = np.sqrt(k_B * T_p / m_i)
    gamma_p = 8.0 / 3.0 * np.sqrt(2.0 * np.pi) * r_p**2 * P / v_T / m_p
    s_p = np.sqrt(2 * k_B * T_p * m_p * gamma_p / dt)
    f_therm_p_x = np.random.normal(0, s_p, 2) - m_p * gamma_p * vxs
    f_therm_p_y = np.random.normal(0, s_p, 2) - m_p * gamma_p * vys
    f_therm_p_z = np.random.normal(0, s_p, 2) - m_p * gamma_p * vzs

    # calculate fields
    E_x, E_y, E_z = calculateAnalitE(xs, ys, zs, qs)
    E_z_trap = E_0 + alpha * zs

    # calculate accs
    axs = -E_x_trap * qs * xs / m_p + E_x * qs / m_p + f_therm_p_x / m_p
    ays = -E_y_trap * qs * ys / m_p + E_y * qs / m_p + f_therm_p_y / m_p
    azs = -g + E_z_trap * qs / m_p + E_z * qs / m_p + f_therm_p_z / m_p

    # Euler's method dynamics
    xs += vxs * dt + 0.5 * axs * np.square(dt)
    ys += vys * dt + 0.5 * ays * np.square(dt)
    zs += vzs * dt + 0.5 * azs * np.square(dt)
    vxs += axs * dt
    vys += ays * dt
    vzs += azs * dt

    if step % 200 == 0:
        with open(file_path, "a") as output:
            print(f"{step} out of {steps}")

            output.write(f"{2}\n{step}\n")
            output.write(
                f"Ar\t"
                f"{xs[TOP_IDX]}\t{ys[TOP_IDX]}\t{zs[TOP_IDX]}\t"
                f"{vxs[TOP_IDX]}\t{vys[TOP_IDX]}\t{vzs[TOP_IDX]}\n"
            )
            output.write(
                f"Xe\t"
                f"{xs[BOT_IDX]}\t{ys[BOT_IDX]}\t{zs[BOT_IDX]}\t"
                f"{vxs[BOT_IDX]}\t{vys[BOT_IDX]}\t{vzs[BOT_IDX]}\n"
            )

        with open(data_path, "a") as output:
            output.write(
                f"{step * dt}\t"
                f"{xs[TOP_IDX]}\t{ys[TOP_IDX]}\t{zs[TOP_IDX]}\t"
                f"{xs[BOT_IDX]}\t{ys[BOT_IDX]}\t{zs[BOT_IDX]}\t"
                f"{vxs[TOP_IDX]}\t{vys[TOP_IDX]}\t{vzs[TOP_IDX]}\t"
                f"{vxs[BOT_IDX]}\t{vys[BOT_IDX]}\t{vzs[BOT_IDX]}\n"
            )
