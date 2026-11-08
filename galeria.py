"""
galeria.py
Gráficas de las pruebas más vistosas. Las guarda en la carpeta galeria/.

Ejecutar:  python galeria.py
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from metodo import rk4_sistema, euler_sistema
from modelo import f, frecuencia_natural, frecuencia_natural_rpm, factor_amortiguamiento, rpm_a_rad_s
from excitaciones import ninguna, construir_F, rampa
import validacion as V

M, m0, e, k, c = 50, 1, 0.25, 5e4, 316
wn = frecuencia_natural(M, k)
zeta = factor_amortiguamiento(M, c, k)
Nn = frecuencia_natural_rpm(M, k)
os.makedirs('galeria', exist_ok=True)

ROJO, AZUL, GRIS, VERDE, MORADO = '#d62728', '#1f77b4', '#555555', '#2ca02c', '#9467bd'


def resolver(F, x0=0.0, v0=0.0, h=0.001, tf=5.0, metodo=rk4_sistema, c=c):
    p = {'M': M, 'c': c, 'k': k, 'F': F}
    with np.errstate(all='ignore'):
        t, Y = metodo(lambda t, Y: f(t, Y, p), 0.0, [x0, v0], h, round(tf / h))
    return t, Y[:, 0], Y[:, 1]


def amplitud_teorica(w, c=c):
    return m0 * e * w**2 / math.sqrt((k - M * w**2)**2 + (c * w)**2)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join('galeria', nombre), dpi=130)
    plt.close(fig)
    print(f"  ✔ galeria/{nombre}")


# --------------------------------------------------------------------------
# 1. Curva de resonancia: simulación (puntos) vs teoría (líneas) para 3 valores de c
# --------------------------------------------------------------------------
def curva_resonancia():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    N_teo = np.linspace(20, 1400, 800)
    N_sim = [80, 160, 220, 260, 285, 300, 315, 340, 380, 450, 600, 850, 1100, 1400]
    for c_i, color in ((158, ROJO), (316, AZUL), (632, VERDE)):
        z = factor_amortiguamiento(M, c_i, k)
        X_teo = [1000 * amplitud_teorica(rpm_a_rad_s(N), c_i) for N in N_teo]
        ax.plot(N_teo, X_teo, color=color, lw=2, label=f"teoría  c = {c_i}  (ζ = {z:.2f})")
        X_sim = []
        for N in N_sim:
            tf = 3 / (z * wn) + 1.5                        # dejar que muera el transitorio
            t, x, _ = resolver(construir_F('giro constante', m0, e, rpm_a_rad_s(N), None),
                               tf=tf, h=0.0005, c=c_i)
            X_sim.append(1000 * np.max(np.abs(x[t > tf - 1.2])))
        ax.plot(N_sim, X_sim, 'o', color=color, mec='k', ms=6)
    ax.plot([], [], 'o', color='w', mec='k', label='simulado con NUESTRO RK4')
    ax.axvline(Nn, color=GRIS, ls=':', lw=1)
    ax.text(Nn - 12, 52, f"Nn = {Nn:.0f} rpm", color=GRIS, ha='right')
    ax.axhline(1000 * m0 * e / M, color=GRIS, ls='--', lw=1)
    ax.text(1400, 1000 * m0 * e / M - 2.5, "m0·e/M = 5 mm (límite a alta velocidad)", color=GRIS,
            fontsize=9, ha='right')
    ax.set_xlabel("Velocidad del tambor N [rpm]")
    ax.set_ylabel("Amplitud en estado estable X [mm]")
    ax.set_title("Curva de resonancia: cada punto es una simulación completa", fontweight='bold')
    ax.set_ylim(0, 56)
    ax.legend()
    ax.grid(alpha=0.3)
    guardar(fig, '1_curva_resonancia.png')


# --------------------------------------------------------------------------
# 2. Energía sin amortiguamiento: Euler la inventa, RK4 la conserva
# --------------------------------------------------------------------------
def energia():
    h, tf = 0.002, 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for metodo, nombre, color in ((euler_sistema, 'Euler', AZUL), (rk4_sistema, 'RK4', ROJO)):
        t, x, v = resolver(ninguna, x0=0.01, h=h, tf=tf, metodo=metodo, c=0)
        E = 0.5 * M * v**2 + 0.5 * k * x**2
        ax1.plot(t, E / E[0], color=color, lw=2, label=nombre)
        ax2.plot(1000 * x, 1000 * v, color=color, lw=1, label=nombre)
    n = np.arange(round(tf / h) + 1)
    ax1.plot(n * h, (1 + h**2 * wn**2) ** n, 'k:', lw=1.5, label='fórmula (1 + h²wn²)ⁿ')
    ax1.set_yscale('log')
    ax1.set_xlabel('t [s]'); ax1.set_ylabel('E(t) / E(0)')
    ax1.set_title('Energía con c = 0 y F = 0 (debería ser constante = 1)')
    ax1.legend(); ax1.grid(alpha=0.3, which='both')
    ax2.set_xlabel('x [mm]'); ax2.set_ylabel("x' [mm/s]")
    ax2.set_title('Plano de fase: RK4 = elipse cerrada · Euler = espiral')
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f"Sin rozamiento la energía se conserva… salvo que uses Euler (h = {h} s)",
                 fontweight='bold')
    guardar(fig, '2_energia_euler_vs_rk4.png')


# --------------------------------------------------------------------------
# 3. Frontera de estabilidad de Euler
# --------------------------------------------------------------------------
def estabilidad_euler():
    h_lim = 2 * zeta / wn
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, factor, color in zip(axs, (0.9, 1.0, 1.1), (VERDE, GRIS, ROJO)):
        h = factor * h_lim
        t, x, _ = resolver(ninguna, x0=0.01, h=h, tf=8, metodo=euler_sistema)
        ax.plot(t, 1000 * x, color=color, lw=1)
        estado = {0.9: 'decae ✔', 1.0: 'ni crece ni decae', 1.1: 'EXPLOTA ✘'}[factor]
        ax.set_title(f"Euler con h = {factor:.1f}·h_lím = {h:.5f} s → {estado}", fontsize=10)
        ax.set_ylabel('x [mm]'); ax.grid(alpha=0.3)
    axs[-1].set_xlabel('t [s]')
    fig.suptitle(f"Euler es estable solo si h < 2ζ/wn = {h_lim:.5f} s  (respuesta libre, x0 = 10 mm)",
                 fontweight='bold')
    guardar(fig, '3_estabilidad_euler.png')


# --------------------------------------------------------------------------
# 4. Arranque en rampa: cruzando la resonancia
# --------------------------------------------------------------------------
def arranque():
    w_max = rpm_a_rad_s(1400)
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, tr in zip(axs, (5, 20)):
        t, x, _ = resolver(lambda s: rampa(s, m0, e, w_max, tr), tf=tr + 5, h=0.0005)
        rpm = np.where(t < tr, 1400 * t / tr, 1400)
        ax.plot(t, 1000 * x, color=ROJO, lw=0.6, label='x(t) [mm]')
        ax.set_ylabel('x [mm]', color=ROJO)
        ax.set_ylim(-30, 30)
        ax2 = ax.twinx()
        ax2.plot(t, rpm, color=AZUL, lw=2, label='velocidad del tambor')
        ax2.axhline(Nn, color=GRIS, ls=':')
        ax2.set_ylabel('N [rpm]', color=AZUL)
        ax2.set_ylim(0, 1550)
        t_res = tr * Nn / 1400
        ax.axvspan(t_res - 0.3, t_res + 0.8, color='gold', alpha=0.3)
        ax.text(t_res + 0.9, 24, f"zona de resonancia\n(N ≈ {Nn:.0f} rpm)", fontsize=9)
        ax.set_title(f"Arranque en {tr} s → pico de {1000*np.max(np.abs(x)):.1f} mm", fontsize=10)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel('t [s]')
    fig.suptitle("Centrifugado real: la máquina SALTA al pasar por resonancia y luego se calma",
                 fontweight='bold')
    guardar(fig, '4_arranque_rampa.png')


# --------------------------------------------------------------------------
# 5. Decremento logarítmico: medir ζ a partir de la simulación
# --------------------------------------------------------------------------
def decremento():
    t, x, _ = resolver(ninguna, x0=0.01, h=0.0001, tf=1.6)
    ip = [i for i in range(1, len(x) - 1) if x[i - 1] < x[i] >= x[i + 1]]
    delta = np.mean(np.log(x[ip][:-1] / x[ip][1:]))
    z_med = delta / math.sqrt(4 * math.pi**2 + delta**2)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, 1000 * x, color=MORADO, lw=1.5, label='x(t) simulado (RK4)')
    env = 1000 * 0.01 * np.exp(-zeta * wn * t) / math.sqrt(1 - zeta**2)
    ax.plot(t, env, 'k--', lw=1, label='envolvente ± e^(−ζ wn t)')
    ax.plot(t, -env, 'k--', lw=1)
    ax.plot(t[ip], 1000 * x[ip], 'o', color=ROJO, ms=7, label='picos detectados')
    for a, b in zip(ip[:3], ip[1:4]):
        ax.annotate('', xy=(t[b], 1000 * x[b]), xytext=(t[a], 1000 * x[a]),
                    arrowprops=dict(arrowstyle='->', color=ROJO))
    ax.text(0.75, 6, f"δ = ln(xₙ / xₙ₊₁) = {delta:.4f}\nζ = δ / √(4π² + δ²) = {z_med:.4f}\n"
                     f"ζ real = c / (2√(kM)) = {zeta:.4f}",
            fontsize=11, family='monospace', bbox=dict(fc='white', ec=GRIS))
    ax.set_xlabel('t [s]'); ax.set_ylabel('x [mm]')
    ax.set_title("Midiendo el amortiguamiento como en un laboratorio: decremento logarítmico",
                 fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    guardar(fig, '5_decremento_logaritmico.png')


# --------------------------------------------------------------------------
# 6. Eficiencia: error contra trabajo (evaluaciones de f)
# --------------------------------------------------------------------------
def eficiencia():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for metodo, nombre, evals, color, hs in (
            (euler_sistema, 'Euler', 1, AZUL, [0.002, 0.001, 0.0005, 0.00025, 0.000125, 0.0000625]),
            (rk4_sistema, 'RK4', 4, ROJO, [0.008, 0.004, 0.002, 0.001, 0.0005])):
        trabajo, errores = [], []
        for h in hs:
            t, x = V.resolver(metodo, h)
            errores.append(1000 * np.max(np.abs(V.solucion_analitica(t) - x)))
            trabajo.append(evals * round(2 / h))
        ax.loglog(trabajo, errores, 'o-', color=color, lw=2, ms=7, label=f"{nombre} ({evals} eval. de f por paso)")
    ax.axhline(1e-3, color=GRIS, ls=':')
    ax.text(1100, 1.3e-3, "1 micra", color=GRIS)
    ax.set_xlabel('Trabajo = número total de evaluaciones de f')
    ax.set_ylabel('Error global máximo [mm]')
    ax.set_title("¿Vale la pena pagar 4 evaluaciones por paso?\n"
                 "RK4 baja de 1 micra con 2 000 evaluaciones; Euler sigue en 0,25 mm con 32 000",
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, which='both')
    guardar(fig, '6_eficiencia_error_vs_trabajo.png')


if __name__ == "__main__":
    print("Generando galería…")
    curva_resonancia()
    energia()
    estabilidad_euler()
    arranque()
    decremento()
    eficiencia()
    print("Listo: abre la carpeta galeria/")
