import numpy as np

rp = 4.445e-6
rho = 1500
mp = 4.0 / 3.0 * np.pi * rp**3 * rho

trap = 1035598

q = 3.105494829523742e-15 + 3.105494829523742e-15

X_TOP_INIT = -0.00029653779187107016
Y_TOP_INIT = 0.0001694319765951752
Z_TOP_INIT = 0.0031863724551996647
X_BOT_INIT = -5.2753970014182775e-05
Y_BOT_INIT = -0.00011692510047717283
Z_BOT_INIT = 0.0027886618943559704
VX_TOP_INIT = 0.011999878468532177
VY_TOP_INIT = 0.02105825810399531
VZ_TOP_INIT = -8.139335036280951e-05
VX_BOT_INIT = -0.008222919729612912
VY_BOT_INIT = 0.0036492688136162696
VZ_BOT_INIT = 0.00012806075899425201

v_top = (VX_TOP_INIT**2 + VY_TOP_INIT**2) ** 0.5
v_bot = (VX_BOT_INIT**2 + VY_BOT_INIT**2) ** 0.5

r_top = (X_TOP_INIT**2 + Y_TOP_INIT**2) ** 0.5
r_bot = (X_BOT_INIT**2 + Y_BOT_INIT**2) ** 0.5

print(v_top / r_top, v_bot / r_bot)
print(r_top, r_bot)

print(
    np.arccos(
        (
            r_top**2
            + r_bot**2
            - (X_BOT_INIT - X_TOP_INIT) ** 2
            - (Y_TOP_INIT - Y_BOT_INIT) ** 2
        )
        / 2
        / r_top
        / r_bot
    ),
    80 / 57,
)
print(np.cos(1.6660635249883957))


print(
    0.00034152870552959104 / 0.00012827494094270423,
    0.00216405 / 0.000748201,
    0.00025532 / 0.00010521,
)
print(mp)

print(97 / 57)
