"""
metodo.py
Métodos numéricos para sistemas de EDOs de primer orden  Y' = f(t, Y).

Aquí Y es un VECTOR (para la lavadora: Y = [x, x']). Por eso K1, K2, K3 y K4
también son vectores: numpy hace las sumas y productos componente a componente.

Restricción de la guía: el método está escrito por nosotros. No se usa
scipy.integrate ni sympy.dsolve.
"""
import numpy as np


def rk4_sistema(f, t0, Y0, h, N):
    """
    Runge-Kutta de orden 4 para un sistema Y' = f(t, Y).

    Entradas:
        f  : función f(t, Y) que devuelve el vector Y' (numpy array)
        t0 : tiempo inicial [s]
        Y0 : vector de condiciones iniciales, p. ej. [x0, v0]
        h  : paso de malla [s]
        N  : número de pasos
    Salidas:
        t : arreglo de N+1 tiempos  t_0, t_1, ..., t_N
        Y : matriz (N+1) x (número de ecuaciones); la fila i es Y_i ≈ Y(t_i)
    """
    # 1. Reservar memoria para los nodos y la solución
    t = np.zeros(N + 1)
    Y = np.zeros((N + 1, len(Y0)))

    # 2. Condición inicial
    t[0] = t0
    Y[0] = Y0

    # 3. Avanzar paso a paso
    for i in range(N):
        K1 = f(t[i],         Y[i])                 # pendiente al inicio
        K2 = f(t[i] + h / 2, Y[i] + (h / 2) * K1)  # pendiente a la mitad (usando K1)
        K3 = f(t[i] + h / 2, Y[i] + (h / 2) * K2)  # pendiente a la mitad (usando K2)
        K4 = f(t[i] + h,     Y[i] + h * K3)        # pendiente al final (usando K3)

        Y[i + 1] = Y[i] + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4)  # promedio ponderado
        t[i + 1] = t0 + (i + 1) * h   # se calcula así (y no t[i] + h) para no acumular redondeo

    # 4. Devolver la malla y la solución
    return t, Y


def euler_sistema(f, t0, Y0, h, N):
    """
    Método de Euler para un sistema Y' = f(t, Y). Mismas entradas y salidas
    que rk4_sistema. Solo se usa para comparar (orden 1 contra orden 4).
    """
    # 1. Reservar memoria
    t = np.zeros(N + 1)
    Y = np.zeros((N + 1, len(Y0)))

    # 2. Condición inicial
    t[0] = t0
    Y[0] = Y0

    # 3. Avanzar paso a paso con UNA sola pendiente (la del inicio)
    for i in range(N):
        Y[i + 1] = Y[i] + h * f(t[i], Y[i])
        t[i + 1] = t0 + (i + 1) * h

    # 4. Devolver
    return t, Y
