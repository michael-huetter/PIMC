import numpy as np
import matplotlib.pyplot as plt

def x_dt(p,m):
    return p/m

def p_dt(x,m,w):
    return -m*w**2*x

def Euler(L,dt,x,p,m,w,x_arr,p_arr):
    x_arr.append(x)
    p_arr.append(p)
    for i in range(L):
        x_old = np.copy(x)
        x += x_dt(p,m) * dt
        p += p_dt(x_old,m,w) * dt
        x_arr.append(x)
        p_arr.append(p)
    return x,p,x_arr,p_arr

def Leapfrog(L,dt,x,p,m,w,x_arr,p_arr):
    x_arr.append(x)
    p_arr.append(p)
    p += dt/2 * p_dt(x,m,w)
    for l in range(L):
        x += dt * x_dt(p,m)
        x_arr.append(x)
        if l != L-1:
            p += dt *p_dt(x,m,w)
            p_arr.append(p)
        else:
            p += dt/2 * p_dt(x,m,w)
            p_arr.append(p)
    return x,p,x_arr,p_arr

def analytic_x_p(x,p,t,m,w):
    x_t = x * np.cos(w*t) + p/(m*w) *np.sin(w*t)
    p_t = p * np.cos(w*t) - m * w * x * np.sin(w*t)
    return x_t,p_t

#Startposition
x_0 = 2
p_0 = 2

L = 20
t = 6
t_list = np.linspace(0,t,100)
dt = t/L

w = 1
m = 1

x_Euler = []
p_Euler = []
x_Leapfrog = []
p_Leapfrog = []

x_E, p_E , x_Euler, p_Euler = Euler(L,dt,x_0,p_0,m,w,x_Euler,p_Euler)
x_L, p_L, x_Leapfrog, p_Leapfrog = Leapfrog(L,dt,x_0,p_0,m,w,x_Leapfrog,p_Leapfrog)
x_an,p_an = analytic_x_p(x_0,p_0,t_list,m,w)

plt.plot(x_Euler,p_Euler,label = "Euler-Verfahren")
plt.plot(x_Leapfrog,p_Leapfrog, label = "Leapfrog-Verfahren")
plt.plot(x_an,p_an, label = "Analytische Lösung",ls = "--")
plt.plot(x_0,p_0, "ro",label = "Startpunkt")
plt.title("Euler vs Leapfrog / Harmonischer Oszillator")
plt.xlabel(r"$x$")
plt.ylabel(r"$p$")
plt.legend()
plt.grid(linestyle="--")
plt.show()