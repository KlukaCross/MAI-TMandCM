import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.integrate import odeint
import sympy as sp


def formY(y, t, fV, fOm):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fV(y1, y2, y3, y4), fOm(y1, y2, y3, y4)]
    return dydt


"""defining parameters"""
a = 10  # DO = OE = a
lAB = 5  # length l of line AB
sA = 1  # size of the A
mA = 8  # mass of A
mB = 2  # mass of B
g = 9.81  # g is g!
k = 15  # spring stiffness coefficient

# defining t as a symbol
t = sp.Symbol('t')

# defining x, phi, v=dx/dt, om=dphi/dt as functions of 't'
xA = sp.Function('x')(t)
phi = sp.Function('phi')(t)
VA = sp.Function('V')(t)
omB = sp.Function('om')(t)

"""constructing the Lagrange equations"""
# the squared velocity of the center of mass
VB2 = VA ** 2 + omB ** 2 * lAB ** 2 - 2 * VA * omB * lAB * sp.sin(phi)
# moment of inertia
JB = (mB * lAB ** 2) / 3
# kinetic energy
EkinB = (mB * VB2) / 2 + (JB * omB ** 2) / 2
EkinA = (mA * VA ** 2) / 2
Ekin = EkinA + EkinB
# potential energy
delta_x = sp.sqrt(a ** 2 + xA ** 2) - a
EpotStrings = k * delta_x ** 2
EpotA = -mA * g * xA
EpotB = -mB * g * (xA + lAB * sp.cos(phi))
Epot = EpotStrings + EpotA + EpotB

# Lagrange function
L = Ekin - Epot

# equations
ur1 = sp.diff(sp.diff(L, VA), t) - sp.diff(L, xA)
ur2 = sp.diff(sp.diff(L, omB), t) - sp.diff(L, phi)

# isolating second derivatives(dV/dt and dom/dt) using Kramer's method
a11 = ur1.coeff(sp.diff(VA, t), 1)
a12 = ur1.coeff(sp.diff(omB, t), 1)
a21 = ur2.coeff(sp.diff(VA, t), 1)
a22 = ur2.coeff(sp.diff(omB, t), 1)
b1 = -(ur1.coeff(sp.diff(VA, t), 0)).coeff(sp.diff(omB, t), 0).subs([(sp.diff(xA, t), VA), (sp.diff(phi, t), omB)])
b2 = -(ur2.coeff(sp.diff(VA, t), 0)).coeff(sp.diff(omB, t), 0).subs([(sp.diff(xA, t), VA), (sp.diff(phi, t), omB)])

det = a11 * a22 - a12 * a21
det1 = b1 * a22 - b2 * a12
det2 = a11 * b2 - b1 * a21

dVAdt = det1 / det
domBdt = det2 / det

"""constructing functions"""
# Constructing the system of differential equations
countOfFrames = 300
y0 = [0, sp.rad(90), 0, 0]  # x(0), phi(0), v(0), om(0)
T_start, T_stop = 0, 25
T = np.linspace(T_start, T_stop, countOfFrames)

fVA = sp.lambdify([xA, phi, VA, omB], dVAdt, "numpy")
fOmB = sp.lambdify([xA, phi, VA, omB], domBdt, "numpy")
sol = odeint(formY, y0, T, args=(fVA, fOmB))

# sol - our solution
# sol[:,0] - x
# sol[:,1] - phi
# sol[:,2] - v (dx/dt)
# sol[:,3] - om (dphi/dt)

XA_def = sp.lambdify(xA, xA)
XB_def = sp.lambdify([xA, phi], xA + lAB * sp.cos(phi))
YB_def = sp.lambdify(phi, lAB * sp.sin(phi))

XA = XA_def(sol[:, 0])
XB = XB_def(sol[:, 0], sol[:, 1])
YB = YB_def(sol[:, 1])

"""plotting"""

fig = plt.figure(figsize=(17, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(ylim=[-a, XA.max() + a], xlim=[min(-lAB, -a), max(lAB, a)])
ax1.set_xlabel('ось y')
ax1.set_ylabel('ось x')
ax1.invert_yaxis()

# plotting dots D and E
ax1.plot(-a, 0, marker='o', color='black')
ax1.plot(a, 0, marker='o', color='black')

# plotting lines between which A is located
ax1.plot([-sA / 2, -sA / 2], [XA.min(), XA.max()], linestyle='-.', color='black')
ax1.plot([sA / 2, sA / 2], [XA.min(), XA.max()], linestyle='-.', color='black')

# plotting initial positions

# plotting A
PA = ax1.add_patch(Rectangle(xy=[-sA / 2, XA[0] - sA / 2], width=sA, height=sA, color='g'))

# plotting B
PB, = ax1.plot(YB[0], XB[0], marker='o', color='r')

# plotting line AB
PAB, = ax1.plot([0, YB[0]], [XA[0], XB[0]], 'black')

# plotting lines DA and EA
PDA, = ax1.plot([-a, 0], [0, XA[0]], linestyle='--', color='m')
PEA, = ax1.plot([a, 0], [0, XA[0]], linestyle='--', color='m')

# plotting T-V and T-Om
ax2 = fig.add_subplot(4, 2, 2)
ax2.set(xlim=[T_start, T_stop], ylim=[min(sol[:, 2]), max(sol[:, 2])])
tv_x = [T[0]]
tv_y = [sol[:, 2][0]]
TV, = ax2.plot(tv_x, tv_y, '-')
ax2.set_xlabel('T')
ax2.set_ylabel('V')

ax3 = fig.add_subplot(4, 2, 4)
ax3.set(xlim=[T_start, T_stop], ylim=[min(sol[:, 3]), max(sol[:, 3])])
tom_x = [T[0]]
tom_y = [sol[:, 3][0]]
TOm, = ax3.plot(tom_x, tom_y, '-')
ax3.set_xlabel('T')
ax3.set_ylabel('Om')

plt.subplots_adjust(wspace=0.3, hspace=0.7)


# function for recounting the positions
def anima(i):
    PA.set(xy=[-sA / 2, XA[i] - sA / 2])
    PB.set_data(YB[i], XB[i])
    PAB.set_data([0, YB[i]], [XA[i], XB[i]])
    PDA.set_data([-a, 0], [0, XA[i]])
    PEA.set_data([a, 0], [0, XA[i]])

    tv_x.append(T[i])
    tv_y.append(sol[:, 2][i])
    tom_x.append(T[i])
    tom_y.append(sol[:, 3][i])
    TV.set_data(tv_x, tv_y)
    TOm.set_data(tom_x, tom_y)
    if i == countOfFrames - 1:
        tv_x.clear()
        tv_y.clear()
        tom_x.clear()
        tom_y.clear()
    return PAB, PDA, PEA, PA, PB, TV, TOm


# animation function
anim = FuncAnimation(fig, anima, frames=countOfFrames, interval=100, blit=True)

plt.show()
