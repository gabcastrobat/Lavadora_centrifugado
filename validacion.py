"""
validacion.py
Valida RK4 (y Euler) contra la solución analítica del caso de giro constante.

1. Resuelve el caso base (N = 300 rpm, x0 = v0 = 0, t en [0, 2] s) con RK4.
2. Grafica la solución numérica y la analítica juntas.
3. Calcula el error global e(h) = max_k |x_exacta(t_k) - x_k| para varios h.
4. Imprime la tabla h, e(h), e(2h)/e(h). RK4 -> razón ≈ 16; Euler -> razón ≈ 2.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from metodo import rk4_sistema, euler_sistema
from modelo import f, frecuencia_natural, factor_amortiguamiento, rpm_a_rad_s
from excitaciones import construir_F

# ---------------- Caso base ----------------
M, m0, e = 50, 1, 0.25
k, c = 5e4, 316
N_rpm = 300
x0, v0 = 0.0, 0.0
t0, tf = 0.0, 2.0

w = rpm_a_rad_s(N_rpm)


def solucion_analitica(t):
    """Solución exacta de M x'' + c x' + k x = m0 e w^2 sin(w t), con 0 < zeta < 1."""
    wn = frecuencia_natural(M, k)
    zeta = factor_amortiguamiento(M, c, k)
    F0 = m0 * e * w**2
    D = (k - M * w**2)**2 + (c * w)**2
    A = -F0 * c * w / D                     # parte particular (estado estable)
    B = F0 * (k - M * w**2) / D
    wd = wn * math.sqrt(1 - zeta**2)        # frecuencia amortiguada
    C1 = x0 - A                             # constantes de la parte homogénea (transitorio)
    C2 = (v0 + zeta * wn * C1 - B * w) / wd
    return (np.exp(-zeta * wn * t) * (C1 * np.cos(wd * t) + C2 * np.sin(wd * t))
            + A * np.cos(w * t) + B * np.sin(w * t))


def resolver(metodo, h):
    """Resuelve el caso base con el método dado y paso h. Devuelve t y x."""
    parametros = {'M': M, 'c': c, 'k': k,
                  'F': construir_F('giro constante', m0, e, w, None)}
    N = round((tf - t0) / h)                        # número de pasos
    t, Y = metodo(lambda t, Y: f(t, Y, parametros), t0, [x0, v0], h, N)
    return t, Y[:, 0]                               # columna 0 = x


def tabla_errores(metodo, nombre, lista_h):
    """Calcula e(h) para cada h e imprime la tabla con la razón e(2h)/e(h)."""
    print(f"\n{nombre}")
    print(f"{'h [s]':>10} | {'e(h) [m]':>12} | {'e(2h)/e(h)':>10}")
    print("-" * 38)
    errores = []
    for i, h in enumerate(lista_h):
        t, x = resolver(metodo, h)
        error = np.max(np.abs(solucion_analitica(t) - x))
        errores.append(error)
        razon = f"{errores[i-1] / error:10.2f}" if i > 0 else f"{'-':>10}"
        print(f"{h:>10.4f} | {error:>12.4e} | {razon}")
    # Orden estimado p = log2(razón) con los dos h más pequeños
    p = math.log2(errores[-2] / errores[-1])
    print(f"Orden estimado p = log2(razón) ≈ {p:.2f}")
    return errores


if __name__ == "__main__":
    wn = frecuencia_natural(M, k)
    zeta = factor_amortiguamiento(M, c, k)
    print(f"Caso base: N = {N_rpm} rpm (w = {w:.2f} rad/s), wn = {wn:.2f} rad/s, zeta = {zeta:.3f}")

    lista_h = [0.004, 0.002, 0.001, 0.0005]
    err_rk4 = tabla_errores(rk4_sistema, "RK4 (se espera razón ≈ 2^4 = 16)", lista_h)
    err_eul = tabla_errores(euler_sistema, "Euler (se espera razón ≈ 2^1 = 2)", lista_h)

    # ---------------- Gráficas ----------------
    h = 0.001
    t, x_rk4 = resolver(rk4_sistema, h)
    _, x_eul = resolver(euler_sistema, h)
    t_fino = np.linspace(t0, tf, 4001)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Validación: lavadora a 300 rpm (casi en resonancia)", fontweight="bold")

    ax1.plot(t_fino, 1000 * solucion_analitica(t_fino), 'k-', lw=2.5, label="Analítica")
    ax1.plot(t[::20], 1000 * x_rk4[::20], 'o', color='tab:red', ms=4, label=f"RK4 (h = {h} s, cada 20 nodos)")
    ax1.plot(t, 1000 * x_eul, '--', color='tab:blue', lw=1, label=f"Euler (h = {h} s)")
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("x [mm]")
    ax1.set_title("Solución numérica vs analítica")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    ax2.semilogy(t, np.abs(solucion_analitica(t) - x_rk4) + 1e-20, color='tab:red', label="|error| RK4")
    ax2.semilogy(t, np.abs(solucion_analitica(t) - x_eul) + 1e-20, color='tab:blue', label="|error| Euler")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("|x_exacta - x_k| [m]")
    ax2.set_title(f"Error puntual con h = {h} s (escala logarítmica)")
    ax2.legend(); ax2.grid(alpha=0.3, which="both")

    ax3.loglog(lista_h, err_rk4, 'o-', color='tab:red', label="RK4")
    ax3.loglog(lista_h, err_eul, 's-', color='tab:blue', label="Euler")
    hh = np.array(lista_h)
    ax3.loglog(hh, err_rk4[-1] * (hh / hh[-1])**4, 'k:', label="pendiente 4")
    ax3.loglog(hh, err_eul[-1] * (hh / hh[-1])**1, 'k--', label="pendiente 1")
    ax3.set_xlabel("h [s]"); ax3.set_ylabel("e(h) [m]")
    ax3.set_title("Error global contra h (log-log): la pendiente es el orden")
    ax3.legend(); ax3.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig("validacion.png", dpi=120)
    print("\nGráfica guardada en validacion.png")
    plt.show()
