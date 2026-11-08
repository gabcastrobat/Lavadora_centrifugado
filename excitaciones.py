"""
excitaciones.py
Fuerzas de excitación F(t) [N] producidas por la masa desbalanceada m0
que gira a una distancia e del eje del tambor.

Todas son funciones de t que se pueden evaluar en CUALQUIER instante
(RK4 las evalúa en t_i, t_i + h/2 y t_i + h, no solo en los nodos).
"""
import math


def giro_constante(t, m0, e, w):
    """
    Tambor girando a velocidad angular constante w [rad/s].
    La componente vertical de la fuerza centrífuga es m0*e*w^2*sin(w t).
    """
    return m0 * e * w**2 * math.sin(w * t)


def rampa(t, m0, e, w_max, tr):
    """
    Arranque en rampa: w sube linealmente de 0 a w_max en tr segundos
    y después se queda constante.

    Con alpha = w_max/tr (aceleración angular [rad/s^2]):
      t <  tr:  w(t) = alpha*t,  theta(t) = alpha*t^2/2
                F = m0*e*( w^2*sin(theta) - alpha*cos(theta) )
                (el segundo término viene de la aceleración tangencial)
      t >= tr:  theta(t) = alpha*tr^2/2 + w_max*(t - tr)
                F = m0*e*w_max^2*sin(theta)
    """
    alpha = w_max / tr
    if t < tr:
        w = alpha * t
        theta = alpha * t**2 / 2
        return m0 * e * (w**2 * math.sin(theta) - alpha * math.cos(theta))
    else:
        theta = alpha * tr**2 / 2 + w_max * (t - tr)
        return m0 * e * w_max**2 * math.sin(theta)


def ninguna(t):
    """Respuesta libre: no hay fuerza externa."""
    return 0.0


def construir_F(tipo, m0, e, w, tr):
    """
    Devuelve una función F(t) de UNA sola variable, lista para meterla en el
    diccionario de parámetros del modelo.

    tipo : 'giro constante', 'rampa' o 'ninguna'
    w    : velocidad de giro [rad/s] (en la rampa es la velocidad final)
    tr   : tiempo de arranque [s] (solo se usa en la rampa)
    """
    if tipo == 'giro constante':
        return lambda t: giro_constante(t, m0, e, w)
    elif tipo == 'rampa':
        return lambda t: rampa(t, m0, e, w, tr)
    else:
        return ninguna


def angulo_tambor(tipo, t, w, tr):
    """
    Ángulo theta(t) [rad] de la masa desbalanceada. Solo se usa para dibujar
    la animación (el tambor girando); no entra en la ecuación.
    """
    if tipo == 'giro constante':
        return w * t
    elif tipo == 'rampa':
        alpha = w / tr
        if t < tr:
            return alpha * t**2 / 2
        return alpha * tr**2 / 2 + w * (t - tr)
    else:
        return 0.0
