import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import sympy as sp

"""defining parameters"""
# DO = OE = a
a = 10
# length l of line AB
lAB = 5
# size of the A
sA = 1

# defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

# defining s, phi
xA = 4*sp.cos(3*t)
phi = 4*sp.sin(t-10)

# Motion of the A
VmodA = sp.diff(xA, t)
WmodA = sp.diff(VmodA, t)

# Motion of the B
xB = xA + lAB * sp.sin(phi)
yB = lAB * sp.cos(phi)

VmodB = sp.sqrt(sp.diff(xB, t)**2 + sp.diff(yB, t)**2)
WmodB = sp.sqrt(sp.diff(xB, t, 2)**2 + sp.diff(yB, t, 2)**2)


"""constructing functions"""
countOfFrames = 200
T_start, T_stop = 0, 12
T = np.linspace(T_start, T_stop, countOfFrames)

XA_def = sp.lambdify(t, xA)
XB_def = sp.lambdify(t, xB)
YB_def = sp.lambdify(t, yB)
VmodB_def = sp.lambdify(t, VmodB)
WmodB_def = sp.lambdify(t, WmodB)

XA = XA_def(T)
XB = XB_def(T)
YB = YB_def(T)
VB = VmodB_def(T)
WB = WmodB_def(T)

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
ax1.plot([-sA/2, -sA/2], [XA.min(), XA.max()], linestyle='-.', color='black')
ax1.plot([sA/2, sA/2], [XA.min(), XA.max()], linestyle='-.', color='black')

# plotting initial positions

# plotting A
PA = ax1.add_patch(Rectangle(xy=[-sA/2, XA[0]-sA/2], width=sA, height=sA, color='g'))

# plotting B
PB, = ax1.plot(YB[0], XB[0], marker='o', color='r')

# plotting line AB
PAB, = ax1.plot([0, YB[0]], [XA[0], XB[0]], 'black')

# plotting lines DA and EA
PDA, = ax1.plot([-a, 0], [0, XA[0]], linestyle='--', color='m')
PEA, = ax1.plot([a, 0], [0, XA[0]], linestyle='--', color='m')

# plotting T-V and T-W
ax2 = fig.add_subplot(4, 2, 2)
ax2.set(xlim=[T_start, T_stop], ylim=[VB.min(), VB.max()])
tv_x = [T[0]]
tv_y = [VB[0]]
TV, = ax2.plot(tv_x, tv_y, '-')
ax2.set_xlabel('T')
ax2.set_ylabel('V')

ax3 = fig.add_subplot(4, 2, 4)
ax3.set(xlim=[T_start, T_stop], ylim=[WB.min(), WB.max()])
tw_x = [T[0]]
tw_y = [WB[0]]
TW, = ax3.plot(tv_x, tv_y, '-')

ax3.set_xlabel('T')
ax3.set_ylabel('W')

plt.subplots_adjust(wspace=0.3, hspace=0.7)


# function for recounting the positions
def anima(i):
    PA.set(xy=[-sA/2, XA[i]-sA/2])
    PB.set_data(YB[i], XB[i])
    PAB.set_data([0, YB[i]], [XA[i], XB[i]])
    PDA.set_data([-a, 0], [0, XA[i]])
    PEA.set_data([a, 0], [0, XA[i]])

    tv_x.append(T[i])
    tv_y.append(VB[i])
    tw_x.append(T[i])
    tw_y.append(WB[i])
    TV.set_data(tv_x, tv_y)
    TW.set_data(tw_x, tw_y)
    if i == countOfFrames-1:
        tv_x.clear()
        tv_y.clear()
        tw_x.clear()
        tw_y.clear()
    return PAB, PDA, PEA, PA, PB, TV, TW


# animation function
anim = FuncAnimation(fig, anima, frames=countOfFrames, interval=100, blit=True)

plt.show()
