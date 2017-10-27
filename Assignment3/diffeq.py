#!/usr/bin/python
import numpy as np
from matplotlib import rc
import matplotlib.pylab as plt
from sys import argv


"""Evaluates the values of x and v for a mass on a spring with t in the interval
[tmin, tmax). The initial value of x is x0, the initial value of v is v0, and
the step size is h. The values can be found using the explicit, implicit, or
symplectic Euler methods.
"""
def euler(tmin, tmax, x0, v0, h, method):
    assert method in ('explicit', 'implicit', 'symplectic'), \
    'Evaluate: Invalid method \'%s\'.' % method

    t = np.arange(tmin, tmax, h)
    x = np.zeros(len(t))
    v = np.zeros(len(t))

    x[0] = x0
    v[0] = v0

    if method == 'explicit':
        for i in range(1, len(t)):
            x[i] = x[i - 1] + h * v[i - 1]
            v[i] = v[i - 1] - h * x[i - 1]

    elif method == 'implicit':
        for i in range(1, len(t)):
            x[i] = (x[i - 1] + h * v[i - 1]) / (1 + h * h)
            v[i] = (v[i - 1] - h * x[i - 1]) / (1 + h * h)

    else:
        for i in range(1, len(t)):
            x[i] = x[i - 1] + h * v[i - 1]
            v[i] = v[i - 1] - h * x[i]

    return t, x, v


"""Finds the exact value of x at time t, where the initial value of x is x0 and
the initial value of v is v0.
"""
def x_exact(t, x0, v0):
    return x0 * np.cos(t) + v0 * np.sin(t)


"""Finds the exact value of v at time t, where the initial value of x is x0 and
the initial value of v is v0.
"""
def v_exact(t, x0, v0):
    return v0 * np.cos(t) - x0 * np.sin(t)


"""Sets up the plotting environment.
"""
def setup():
    rc('font', **{'family': 'CMU Serif', 'size': 11})
    rc('grid', **{'color': '0.9'})
    rc('lines', **{'markersize': 0.3})
    rc('legend', **{'loc': 'upper center', 'handlelength': 0.3})
    rc('savefig', **{'bbox': 'tight', 'dpi': 1000, 'format': 'png'})
    plt.figure(figsize=(3.5, 3.5))


"""Plots the position and velocity using the explicit Euler method.
"""
def plot_explicit_euler():
    t, x, v = euler(0, 15, 1, 2, 0.05, 'explicit')

    plt.clf()
    plt.plot(t, x, 'bo', label='Position')
    plt.plot(t, v, 'go', label='Velocity')
    plt.xlabel('Time')
    plt.ylabel('Position and Velocity')
    plt.xlim(0, 15)
    plt.ylim(-4, 4)
    plt.grid()
    plt.legend()
    plt.savefig('explicit_euler.png')


"""Plots the errors in the position and velocity obtained with the explicit
Euler method.
"""
def plot_explicit_euler_errors():
    t, x, v = euler(0, 15, 1, 2, 0.05, 'explicit')

    plt.clf()
    plt.plot(t, x_exact(t, 1, 2) - x, 'bo', label='Error in Position')
    plt.plot(t, v_exact(t, 1, 2) - v, 'go', label='Error in Velocity')
    plt.xlabel('Time')
    plt.ylabel('Errors in Position and Velocity')
    plt.xlim(0, 15)
    plt.ylim(-1, 1)
    plt.grid()
    plt.legend()
    plt.savefig('explicit_euler_errors.png')


"""Plots the energy using the explicit Euler method.
"""
def plot_explicit_euler_energy():
    t, x, v = euler(0, 15, 1, 2, 0.05, 'explicit')

    plt.clf()
    plt.plot(t, x**2 + v**2, 'bo')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.xlim(0, 15)
    plt.ylim(5, 11)
    plt.grid()
    plt.savefig('explicit_euler_energy.png')


"""Plots the maximum errors in the position and velocity obtained with the
explicit Euler method for different step sizes.
"""
def plot_explicit_euler_max_errors():
    h = np.arange(0.0002, 0.06, 0.0002)
    x_max_errors = np.zeros(len(h))
    for i in range(len(h)):
        t, x, _ = euler(0, 15, 1, 2, h[i], 'explicit')
        x_max_errors[i] = np.max(x_exact(t, 1, 2) - x)

    plt.clf()
    plt.plot(h, x_max_errors, 'bo')
    plt.xlabel('Step Size')
    plt.ylabel('Maximum Error in Position')
    plt.xlim(0, 0.06)
    plt.ylim(0, 0.8)
    plt.grid()
    plt.savefig('explicit_euler_max_errors.png')


"""Plots the postion and velocity using the implicit Euler method.
"""
def plot_implicit_euler():
    t, x, v = euler(0, 15, 1, 2, 0.05, 'implicit')

    plt.clf()
    plt.plot(t, x, 'bo', label='Position')
    plt.plot(t, v, 'go', label='Velocity')
    plt.xlabel('Time')
    plt.ylabel('Position and Velocity')
    plt.xlim(0, 15)
    plt.ylim(-3, 3)
    plt.grid()
    plt.legend()
    plt.savefig('implicit_euler.png')


"""Plots the errors in the postion and velocity obtained with the implicit Euler
method.
"""
def plot_implicit_euler_errors():
    t, x, v = euler(0, 15, 1, 2, 0.05, 'implicit')

    plt.clf()
    plt.plot(t, x_exact(t, 1, 2) - x, 'bo', label='Error in Position')
    plt.plot(t, v_exact(t, 1, 2) - v, 'go', label='Error in Velocity')
    plt.xlabel('Time')
    plt.ylabel('Errors in Position and Velocity')
    plt.xlim(0, 15)
    plt.ylim(-0.8, 0.8)
    plt.grid()
    plt.legend()
    plt.savefig('implicit_euler_errors.png')


"""Plots the energy using the implicit Euler method.
"""
def plot_implicit_euler_energy():
    t, x, v = euler(0, 15, 1, 2, 0.05, 'implicit')

    plt.clf()
    plt.plot(t, x**2 + v**2, 'bo')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.xlim(0, 15)
    plt.ylim(2, 5)
    plt.grid()
    plt.savefig('implicit_euler_energy.png')


"""Plots the maximum errors in the position and velocity obtained with the
implicit Euler method for different step sizes.
"""
def plot_implicit_euler_energy():
    h = np.arange(0.0002, 0.06, 0.0002)
    x_max_errors = np.zeros(len(h))
    for i in range(len(h)):
        t, x, _ = euler(0, 15, 1, 2, h[i], 'implicit')
        x_max_errors[i] = np.max(x_exact(t, 1, 2) - x)

    plt.clf()
    plt.plot(h, x_max_errors, 'bo')
    plt.xlabel('Step Size')
    plt.ylabel('Maximum Error in Position')
    plt.xlim(0, 0.06)
    plt.ylim(0, 0.8)
    plt.grid()
    plt.savefig('implicit_euler_max_errors.png')


"""Plots the position and velocity phase space using the explicit, implicit, and
symplectic Euler methods, as well as the analytic solution.
"""
def plot_phase_space():
    t, x_explicit, v_explicit = euler(0, 300, 1, 2, 0.05, 'explicit')
    _, x_implicit, v_implicit = euler(0, 300, 1, 2, 0.05, 'implicit')
    _, x_symplectic, v_symplectic = euler(0, 300, 1, 2, 0.05, 'symplectic')

    plt.clf()
    plt.plot(x_explicit, v_explicit, 'bo', label='Explicit')
    plt.plot(x_implicit, v_implicit, 'go', label='Implicit')
    plt.plot(x_symplectic, v_symplectic, 'yo', label='Symplectic')
    plt.plot(x_exact(t, 1, 2), v_exact(t, 1, 2), 'mo', label='Analytic')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xticks(range(-4, 5))
    plt.grid()
    plt.legend(
        bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure,
        mode='expand', ncol=4
        )
    plt.savefig('phase_space.png')


"""Plots the energy using the symplectic Euler method.
"""
def plot_symplectic_euler_energy():
    t, x_symplectic, v_symplectic = euler(0, 300, 1, 2, 0.05, 'symplectic')

    plt.clf()
    plt.plot(t, x_symplectic**2 + v_symplectic**2, 'bo')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.xlim(0, 15)
    plt.ylim(4.95, 5.25)
    plt.grid()
    plt.savefig('symplectic_euler_energy.png')


"""Plots the position using the symplectic Euler method, as well as the analytic
solution.
"""
def plot_symplectic_euler():
    t, x_symplectic, v_symplectic = euler(1000, 1015, 1, 2, 0.05, 'symplectic')

    plt.clf()
    plt.plot(t, x_symplectic, 'bo', label='Symplectic')
    plt.plot(t, x_exact(t, 1, 2), 'go', label='Analytic')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.xlim(1000, 1015)
    plt.ylim(-3, 3)
    plt.grid()
    plt.legend()
    plt.savefig('symplectic_euler.png')


if __name__ == '__main__':
    if len(argv) != 1:
        print "Usage: python diffeq.py [image_name]"
        return

    setup()
    if argv[1] == "explicit_euler":
        plot_explicit_euler()
    elif argv[1] == "explicit_euler_errors":
        plot_explicit_euler_errors()
    elif argv[1] == "explicit_euler_energy":
        plot_explicit_euler_energy()
    elif argv[1] == "explicit_euler_max_errors":
        plot_explicit_euler_max_errors()
    elif argv[1] == "implicit_euler":
        plot_implicit_euler()
    elif argv[1] == "implicit_euler_errors":
        plot_implicit_euler_errors()
    elif argv[1] == "implicit_euler_energy":
        plot_implicit_euler_energy()
    elif argv[1] == "implicit_euler_max_errors":
        plot_implicit_euler_max_errors()
    elif argv[1] == "phase_space":
        plot_phase_space()
    elif argv[1] == "symplectic_euler_energy":
        plot_symplectic_euler_energy()
    elif argv[1] == "symplectic_euler":
        plot_symplectic_euler()
    else:
        print "Error: Image %s not found" % argv[1]
