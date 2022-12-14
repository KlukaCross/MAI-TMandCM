import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.integrate import odeint
import sympy as sp


def formY(y, t, fV):
    y1, y2 = y
    dydt = [y2, fV(y1, y2)]
    return dydt


"""defining parameters"""
a = 1  # DO = OE = a
lAB = 0.5  # length l of line AB
sA = 0.2  # size of the A
mA = 1  # mass of A
mB = 1  # mass of B
g = 9.81  # g is g!
k = 9.81  # spring stiffness coefficient
y0 = [2, 0]  # x(0), v(0)

# defining t as a symbol
t = sp.Symbol('t')

# defining x, phi, v=dx/dt, om=dphi/dt as functions of 't'
xA = sp.Function('x')(t)
phi = 0*t
VA = sp.Function('V')(t)
omB = 0*t

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
#ur2 = sp.diff(sp.diff(L, omB), t) - sp.diff(L, phi)
print(ur1)

# isolating second derivatives(dV/dt and dom/dt) using Kramer's method
a11 = ur1.coeff(sp.diff(VA, t), 1)
#a12 = ur1.coeff(sp.diff(omB, t), 1)
#a21 = ur2.coeff(sp.diff(VA, t), 1)
#a22 = ur2.coeff(sp.diff(omB, t), 1)
b1 = -(ur1.coeff(sp.diff(VA, t), 0)).subs(sp.diff(xA, t), VA)

#det = a11 * a22 - a12 * a21
#det1 = b1 * a22 - b2 * a12
#det2 = a11 * b2 - b1 * a21

dVAdt = b1 / a11
#domBdt = det2 / det
print(dVAdt)

"""constructing functions"""
# Constructing the system of differential equations
countOfFrames = 300
T_start, T_stop = 0, 25
T = np.linspace(T_start, T_stop, countOfFrames)

fVA = sp.lambdify([xA, VA], dVAdt, "numpy")
#fOmB = sp.lambdify([xA, phi, VA, omB], domBdt, "numpy")
sol = odeint(formY, y0, T, args=(fVA,))

# sol - our solution
# sol[:,0] - x
# sol[:,1] - v (dx/dt)

XA_def = sp.lambdify(xA, xA)
XB_def = sp.lambdify(xA, xA + lAB)
#YB_def = sp.lambdify(phi, lAB * sp.sin(phi))

XA = XA_def(sol[:, 0])
XB = XB_def(sol[:, 0])
#YB = YB_def(sol[:, 0])

"""plotting"""

fig = plt.figure(figsize=(17, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(ylim=[XA.min() - lAB, XA.max() + lAB], xlim=[min(-lAB, -a), max(lAB, a)])
ax1.set_xlabel('?????? y')
ax1.set_ylabel('?????? x')
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
PB, = ax1.plot(0, XB[0], marker='o', color='r')

# plotting line AB
PAB, = ax1.plot([0, 0], [XA[0], XB[0]], 'black')

# plotting lines DA and EA
PDA, = ax1.plot([-a, 0], [0, XA[0]], linestyle='--', color='m')
PEA, = ax1.plot([a, 0], [0, XA[0]], linestyle='--', color='m')

# plotting T-X and T-V
ax2 = fig.add_subplot(4, 2, 2)
ax2.set(xlim=[T_start, T_stop], ylim=[min(sol[:, 0]), max(sol[:, 0])])
tx_x = [T[0]]
tx_y = [sol[:, 0][0]]
TX, = ax2.plot(tx_x, tx_y, '-')
ax2.set_xlabel('T')
ax2.set_ylabel('X')


ax3 = fig.add_subplot(4, 2, 4)
ax3.set(xlim=[T_start, T_stop], ylim=[min(sol[:, 1]), max(sol[:, 1])])
tv_x = [T[0]]
tv_y = [sol[:, 1][0]]
TV, = ax3.plot(tv_x, tv_y, '-')
ax3.set_xlabel('T')
ax3.set_ylabel('V')

#ax3 = fig.add_subplot(4, 2, 4)
#ax3.set(xlim=[T_start, T_stop], ylim=[min(sol[:, 3]), max(sol[:, 3])])
#tom_x = [T[0]]
#tom_y = [sol[:, 3][0]]
#TOm, = ax3.plot(tom_x, tom_y, '-')
#ax3.set_xlabel('T')
#ax3.set_ylabel('Om')

plt.subplots_adjust(wspace=0.3, hspace=0.7)


# function for recounting the positions
def anima(i):
    PA.set(xy=[-sA / 2, XA[i] - sA / 2])
    PB.set_data(0, XB[i])
    PAB.set_data([0, 0], [XA[i], XB[i]])
    PDA.set_data([-a, 0], [0, XA[i]])
    PEA.set_data([a, 0], [0, XA[i]])

    tv_x.append(T[i])
    tv_y.append(sol[:, 1][i])
    tx_x.append(T[i])
    tx_y.append(sol[:, 0][i])
    TV.set_data(tv_x, tv_y)
    TX.set_data(tx_x, tx_y)
    if i == countOfFrames - 1:
        tv_x.clear()
        tv_y.clear()
        tx_x.clear()
        tx_y.clear()
    return PAB, PDA, PEA, PA, PB, TV, TX


# animation function
anim = FuncAnimation(fig, anima, frames=countOfFrames, interval=100, blit=True)

plt.show()
