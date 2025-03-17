import os
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calc.calculateCharge import Charge
from calc.calculatePlasmaE import Plasma
from calc.calculateAnalitE import calculateAnalitE
from calc.calculateAnalitE import getWake
from calc.fit import fit


# В динамике все в СИ!!!!
# посмотреть на силы не в динамике, а просто двигаем частицы!
# и сравнить с целевым
# warnings.filterwarnings("ignore")

folder_path = "data/calibration/"
file_path = os.path.join(folder_path, "data.txt")
Path(folder_path).mkdir(parents=True, exist_ok=True)
with open(file_path, "w") as file:
    file.write("idx\tx\ty\tz\tEx_ref\tEy_ref\tEz_ref\tEw_x\tEw_y\tEw_z\n\n")

k_B = 1.380649e-23  # Boltzmann constant (J/K)
m_i = 6.6335209e-26  # Ar+ ions' mass (kg)
r_D_e = 0.0016622715189364113  # m
g = 9.8  # free fall acceleration (m/s^2)

rho = 1500  # mass density of dust particle
n_dust = 1_000_000  # number of dust particles time steps
T_p = 300  # Kinetic temperature of dust particles motion
E_x_trap = -1.0 * 1035598  # x-trap (kg/s^2)
E_y_trap = -1.0 * 1035598  # y-trap (kg/s^2)
d_t_p = 5e-5  # integration step for dust particles dynamics, s
N_p = 2  # number of dust particles
r_p = 4.445e-6  # Dust particle radius
P = 0.25 * 18.131842  # Pressure (Pascal)

X_TOP = 0.0
X_BOT = 0.0

Y_TOP = 0.0
Y_BOT = 0.0

Z_TOP = 0.0032248049595676263
Z_BOT = 0.002742747106163061

TOP_IDX = 0
BOT_IDX = 1

m_p = 4.0 / 3.0 * np.pi * r_p**3 * rho
n_output = 200

assert N_p == 2

# strange magic
plasma = Plasma()
charge = Charge()

# simulation BEGIN

timestamp = 0.0

x_top, y_top, z_top = X_TOP, Y_TOP, Z_TOP
x_bot, y_bot, z_bot = X_BOT, Y_BOT, Z_BOT

xs = np.array([x_top, x_bot])
ys = np.array([y_top, y_bot])
zs = np.array([z_top, z_bot])
qs = charge.calculateLinearCharge(xs / r_D_e, ys / r_D_e, zs / r_D_e)

E_x_analit, E_y_analit, E_z_analit = calculateAnalitE(xs, ys, zs, qs)
E_x_ref, E_y_ref, E_z_ref = plasma.calculatePlasmaE(
    xs / r_D_e, ys / r_D_e, zs / r_D_e, qs
)

# x \in [-r_D_e, +r_D_e]
# y \in [-r_D_e, +r_D_e]
# z \in [-dist / 2, +dist / 2]
xs_interval = np.linspace(-r_D_e, +r_D_e, 100)
ys_interval = np.linspace(-r_D_e, +r_D_e, 100)
zs_interval = np.linspace(-(z_top - z_bot) / 2, +(z_top - z_bot) / 2, 100)

file_path = os.path.join(folder_path, "data_with_fixed_yz.txt")
with open(file_path, "w") as file:
    file.write("idx\tx\ty\tz\tEx_ref\tEy_ref\tEz_ref\tEw_x\tEw_y\tEw_z\n\n")
for x in xs_interval:
    x_bot = x
    y_bot = Y_BOT
    z_bot = Z_BOT

    xs = np.array([x_top, x_bot])
    ys = np.array([y_top, y_bot])
    zs = np.array([z_top, z_bot])

    Ew_x_analit, Ew_y_analit, Ew_z_analit = getWake(
        xs,
        ys,
        -zs,
        qs,
    )
    E_x_ref, E_y_ref, E_z_ref = (
        plasma.calculatePlasmaE(
            xs / r_D_e,
            ys / r_D_e,
            zs / r_D_e,
            qs,
        )
        / qs
    )

    with open(file_path, "a") as output:
        output.write(
            f"{TOP_IDX}\t"
            f"{xs[TOP_IDX]}\t{ys[TOP_IDX]}\t{zs[TOP_IDX]}\t"
            f"{E_x_ref[TOP_IDX]}\t{E_y_ref[TOP_IDX]}\t{E_z_ref[TOP_IDX]}\t"
            f"{Ew_x_analit[TOP_IDX]}\t{Ew_y_analit[TOP_IDX]}\t{Ew_z_analit[TOP_IDX]}\n"
        )
        output.write(
            f"{BOT_IDX}\t"
            f"{xs[BOT_IDX]}\t{ys[BOT_IDX]}\t{zs[BOT_IDX]}\t"
            f"{E_x_ref[BOT_IDX]}\t{E_y_ref[BOT_IDX]}\t{E_z_ref[BOT_IDX]}\t"
            f"{Ew_x_analit[BOT_IDX]}\t{Ew_y_analit[BOT_IDX]}\t{Ew_z_analit[BOT_IDX]}\n"
        )

file_path = os.path.join(folder_path, "data_with_fixed_xz.txt")
with open(file_path, "w") as file:
    file.write("idx\tx\ty\tz\tEx_ref\tEy_ref\tEz_ref\tEw_x\tEw_y\tEw_z\n\n")
for y in ys_interval:
    x_bot = X_BOT
    y_bot = y
    z_bot = Z_BOT

    xs = np.array([x_top, x_bot])
    ys = np.array([y_top, y_bot])
    zs = np.array([z_top, z_bot])

    Ew_x_analit, Ew_y_analit, Ew_z_analit = getWake(
        xs,
        ys,
        -zs,
        qs,
    )
    E_x_ref, E_y_ref, E_z_ref = (
        plasma.calculatePlasmaE(
            xs / r_D_e,
            ys / r_D_e,
            zs / r_D_e,
            qs,
        )
        / qs
    )

    with open(file_path, "a") as output:
        output.write(
            f"{TOP_IDX}\t"
            f"{xs[TOP_IDX]}\t{ys[TOP_IDX]}\t{zs[TOP_IDX]}\t"
            f"{E_x_ref[TOP_IDX]}\t{E_y_ref[TOP_IDX]}\t{E_z_ref[TOP_IDX]}\t"
            f"{Ew_x_analit[TOP_IDX]}\t{Ew_y_analit[TOP_IDX]}\t{Ew_z_analit[TOP_IDX]}\n"
        )
        output.write(
            f"{BOT_IDX}\t"
            f"{xs[BOT_IDX]}\t{ys[BOT_IDX]}\t{zs[BOT_IDX]}\t"
            f"{E_x_ref[BOT_IDX]}\t{E_y_ref[BOT_IDX]}\t{E_z_ref[BOT_IDX]}\t"
            f"{Ew_x_analit[BOT_IDX]}\t{Ew_y_analit[BOT_IDX]}\t{Ew_z_analit[BOT_IDX]}\n"
        )

file_path = os.path.join(folder_path, "data_with_fixed_xy.txt")
with open(file_path, "w") as file:
    file.write("idx\tx\ty\tz\tEx_ref\tEy_ref\tEz_ref\tEw_x\tEw_y\tEw_z\n\n")
for z in zs_interval:
    x_bot = X_BOT
    y_bot = Y_BOT
    z_bot = z

    xs = np.array([x_top, x_bot])
    ys = np.array([y_top, y_bot])
    zs = np.array([z_top, z_bot])

    Ew_x_analit, Ew_y_analit, Ew_z_analit = getWake(
        xs,
        ys,
        -zs,
        qs,
    )
    E_x_ref, E_y_ref, E_z_ref = (
        plasma.calculatePlasmaE(
            xs / r_D_e,
            ys / r_D_e,
            zs / r_D_e,
            qs,
        )
        / qs
    )

    with open(file_path, "a") as output:
        output.write(
            f"{TOP_IDX}\t"
            f"{xs[TOP_IDX]}\t{ys[TOP_IDX]}\t{zs[TOP_IDX]}\t"
            f"{E_x_ref[TOP_IDX]}\t{E_y_ref[TOP_IDX]}\t{E_z_ref[TOP_IDX]}\t"
            f"{Ew_x_analit[TOP_IDX]}\t{Ew_y_analit[TOP_IDX]}\t{Ew_z_analit[TOP_IDX]}\n"
        )
        output.write(
            f"{BOT_IDX}\t"
            f"{xs[BOT_IDX]}\t{ys[BOT_IDX]}\t{zs[BOT_IDX]}\t"
            f"{E_x_ref[BOT_IDX]}\t{E_y_ref[BOT_IDX]}\t{E_z_ref[BOT_IDX]}\t"
            f"{Ew_x_analit[BOT_IDX]}\t{Ew_y_analit[BOT_IDX]}\t{Ew_z_analit[BOT_IDX]}\n"
        )
