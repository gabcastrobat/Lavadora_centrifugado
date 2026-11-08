"""
modelo.py
Modelo de la lavadora en centrifugado:

    M x'' + c x' + k x = F(t),    x(0) = x0,  x'(0) = v0

Se convierte en un sistema de primer orden con  y1 = x,  y2 = x':

    y1' = y2
    y2' = ( F(t) - c*y2 - k*y1 ) / M
"""
import math
import numpy as np


def f(t, Y, parametros):
    """
    Lado derecho del sistema  Y' = f(t, Y).

    t          : tiempo [s]
    Y          : vector [y1, y2] = [x, x']  ([m], [m/s])
    parametros : diccionario con
                   'M' masa total [kg]
                   'c' amortiguamiento [N·s/m]
                   'k' rigidez [N/m]
                   'F' función F(t) de la excitación [N]
    Devuelve el vector [y1', y2'] = [x', x''].
    """
    M = parametros['M']
    c = parametros['c']
    k = parametros['k']
    F = parametros['F']

    y1, y2 = Y[0], Y[1]

    dy1 = y2                              # x' = velocidad
    dy2 = (F(t) - c * y2 - k * y1) / M    # x'' = aceleración (2a ley de Newton despejada)

    return np.array([dy1, dy2])


def frecuencia_natural(M, k):
    """wn = sqrt(k/M) en rad/s."""
    return math.sqrt(k / M)


def frecuencia_natural_rpm(M, k):
    """Nn = 60*wn/(2*pi): la velocidad del tambor que produce resonancia, en rpm."""
    return 60 * frecuencia_natural(M, k) / (2 * math.pi)


def factor_amortiguamiento(M, c, k):
    """zeta = c / (2*sqrt(k*M)). zeta < 1: subamortiguado (oscila)."""
    return c / (2 * math.sqrt(k * M))


def rpm_a_rad_s(N):
    """w = 2*pi*N/60."""
    return 2 * math.pi * N / 60


if __name__ == "__main__":
    # Prueba rápida con los valores base
    M, c, k = 50, 316, 5e4
    print(f"wn   = {frecuencia_natural(M, k):.2f} rad/s")
    print(f"Nn   = {frecuencia_natural_rpm(M, k):.1f} rpm")
    print(f"zeta = {factor_amortiguamiento(M, c, k):.3f}")
