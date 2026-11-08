"""
pruebas.py
Batería de pruebas del simulador. Cada prueba compara lo que calcula NUESTRO
código contra algo que sabemos que es cierto (matemática o física).

Ejecutar:  python pruebas.py

Bloques:
  A. El método numérico (RK4 / Euler)
  B. La física de la lavadora
  C. La excitación en rampa
  D. El simulador (interfaz, robustez, reglas de la guía)
"""
import math
import random
import time

import matplotlib
matplotlib.use('Agg')            # sin ventanas: las pruebas no necesitan pantalla
import numpy as np

from metodo import rk4_sistema, euler_sistema
from modelo import f, frecuencia_natural, frecuencia_natural_rpm, factor_amortiguamiento, rpm_a_rad_s
from excitaciones import giro_constante, rampa, ninguna, construir_F, angulo_tambor
import simulador

# Valores base
M, m0, e, k, c = 50, 1, 0.25, 5e4, 316
wn = frecuencia_natural(M, k)
zeta = factor_amortiguamiento(M, c, k)
wd = wn * math.sqrt(1 - zeta**2)


# --------------------------------------------------------------------------
# Herramientas
# --------------------------------------------------------------------------
def resolver(F, x0=0.0, v0=0.0, h=0.001, tf=5.0, metodo=rk4_sistema, M=M, c=c, k=k):
    """Resuelve M x'' + c x' + k x = F(t) y devuelve t, x, v."""
    p = {'M': M, 'c': c, 'k': k, 'F': F}
    with np.errstate(all='ignore'):
        t, Y = metodo(lambda t, Y: f(t, Y, p), 0.0, [x0, v0], h, round(tf / h))
    return t, Y[:, 0], Y[:, 1]


def picos(x):
    """Índices de los máximos locales de x."""
    return [i for i in range(1, len(x) - 1) if x[i - 1] < x[i] >= x[i + 1]]


def amplitud_teorica(w, M=M, m0=m0, e=e, k=k, c=c):
    F0 = m0 * e * w**2
    return F0 / math.sqrt((k - M * w**2)**2 + (c * w)**2)


RESULTADOS = []   # (bloque, nombre, pasó, detalle, segundos)
PRUEBAS = []      # funciones de prueba, en el orden en que se definen


def prueba(bloque, nombre):
    """Decorador: registra la prueba, la ejecuta y guarda si pasó."""
    def envolver(func):
        def correr():
            t0 = time.time()
            try:
                ok, detalle = func()
            except Exception as err:          # una prueba que se cae cuenta como fallida
                ok, detalle = False, f"EXCEPCIÓN: {err!r}"
            RESULTADOS.append((bloque, nombre, ok, detalle, time.time() - t0))
        PRUEBAS.append(correr)
        return correr
    return envolver


# ==========================================================================
# A. MÉTODO NUMÉRICO
# ==========================================================================
@prueba('A', 'RK4 es EXACTO para y\' = 4t³ (integra polinomios de grado 3 como Simpson)')
def _():
    # Sistema escalar disfrazado de vector: Y = [y], y' = 4t^3, y(0)=0 -> y = t^4
    t, Y = rk4_sistema(lambda t, Y: np.array([4 * t**3]), 0.0, [0.0], 0.1, 20)
    err = np.max(np.abs(Y[:, 0] - t**4))
    return err < 1e-12, f"error máx = {err:.1e} (solo redondeo)"


@prueba('A', 'Un paso de RK4 en y\' = y reproduce Taylor: 1 + h + h²/2 + h³/6 + h⁴/24')
def _():
    h = 0.3
    _, Y = rk4_sistema(lambda t, Y: Y, 0.0, [1.0], h, 1)
    taylor = 1 + h + h**2 / 2 + h**3 / 6 + h**4 / 24
    return abs(Y[1, 0] - taylor) < 1e-15, f"RK4 = {Y[1,0]:.15f}, Taylor = {taylor:.15f}"


@prueba('A', 'Orden de RK4 ≈ 4 (razón del error al dividir h entre 2 ≈ 16)')
def _():
    import validacion as V
    errores = []
    for h in (0.002, 0.001, 0.0005):
        t, x = V.resolver(rk4_sistema, h)
        errores.append(np.max(np.abs(V.solucion_analitica(t) - x)))
    razones = [errores[i] / errores[i + 1] for i in range(2)]
    ok = all(15 < r < 17 for r in razones)
    return ok, "razones = " + ", ".join(f"{r:.2f}" for r in razones)


@prueba('A', 'Orden de Euler ≈ 1 (razón → 2 con h pequeño)')
def _():
    import validacion as V
    errores = []
    for h in (0.0005, 0.00025, 0.000125):
        t, x = V.resolver(euler_sistema, h)
        errores.append(np.max(np.abs(V.solucion_analitica(t) - x)))
    razones = [errores[i] / errores[i + 1] for i in range(2)]
    ok = all(1.9 < r < 2.3 for r in razones)
    return ok, "razones = " + ", ".join(f"{r:.2f}" for r in razones)


@prueba('A', 'Euler sin amortiguar GANA energía: E crece exactamente ×(1 + h²wn²) por paso')
def _():
    h, n = 0.002, 1000
    t, x, v = resolver(ninguna, x0=0.01, h=h, tf=n * h, metodo=euler_sistema, c=0)
    E = 0.5 * M * v**2 + 0.5 * k * x**2
    predicho = (1 + h**2 * wn**2) ** n
    real = E[-1] / E[0]
    return abs(real / predicho - 1) < 1e-9, f"E_final/E0 = {real:.4f} (fórmula: {predicho:.4f}) en {n*h:.0f} s"


@prueba('A', 'RK4 sin amortiguar casi conserva la energía (factor |R(ihw)|² por paso)')
def _():
    h, n = 0.002, 1000
    t, x, v = resolver(ninguna, x0=0.01, h=h, tf=n * h, c=0)
    E = 0.5 * M * v**2 + 0.5 * k * x**2
    z = h * wn
    factor = 1 - z**6 / 72 + z**8 / 576           # |1 + iz - z²/2 - iz³/6 + z⁴/24|²
    real, predicho = E[-1] / E[0], factor**n
    return abs(real / predicho - 1) < 1e-9, f"E_final/E0 = {real:.10f} (fórmula: {predicho:.10f})"


@prueba('A', 'Frontera de estabilidad de Euler en h = 2ζ/wn (justo abajo decae, justo arriba crece)')
def _():
    h_lim = 2 * zeta / wn
    _, x_abajo, _ = resolver(ninguna, x0=0.01, h=0.95 * h_lim, tf=20, metodo=euler_sistema)
    _, x_arriba, _ = resolver(ninguna, x0=0.01, h=1.05 * h_lim, tf=20, metodo=euler_sistema)
    a, b = np.max(np.abs(x_abajo[-500:])), np.max(np.abs(x_arriba[-500:]))
    return a < 0.01 < b, f"h_lim = {h_lim:.5f} s → |x| final: {1000*a:.3f} mm (0.95 h_lim) vs {1000*b:.1f} mm (1.05 h_lim)"


@prueba('A', 'RK4 no se desestabiliza con el h más grande del slider (0,01 s)')
def _():
    _, x, _ = resolver(ninguna, x0=0.01, h=0.01, tf=30)
    return np.all(np.isfinite(x)) and np.max(np.abs(x[-100:])) < 1e-4, \
        f"|x| final = {np.max(np.abs(x[-100:])):.1e} m (decae como debe)"


# ==========================================================================
# B. FÍSICA
# ==========================================================================
@prueba('B', 'Equilibrio: con x0 = v0 = 0 y sin fuerza, la máquina no se mueve')
def _():
    _, x, _ = resolver(ninguna, tf=5)
    return np.all(x == 0), "x(t) = 0 exacto en todos los nodos"


@prueba('B', 'Ley de Hooke: con fuerza constante de 500 N, x → F/k = 10 mm')
def _():
    _, x, _ = resolver(lambda t: 500.0, tf=10)
    return abs(x[-1] - 500 / k) < 1e-6, f"x final = {1000*x[-1]:.4f} mm"


@prueba('B', 'Periodo de la vibración libre = 2π/wd')
def _():
    t, x, _ = resolver(ninguna, x0=0.01, h=0.0001, tf=2)
    ip = picos(x)
    T_med = np.mean(np.diff(t[ip]))
    T_teo = 2 * math.pi / wd
    return abs(T_med / T_teo - 1) < 1e-3, f"medido {T_med:.5f} s, teórico {T_teo:.5f} s"


@prueba('B', 'Decremento logarítmico: de los picos se recupera ζ = 0,100')
def _():
    t, x, _ = resolver(ninguna, x0=0.01, h=0.0001, tf=2)
    ip = picos(x)
    delta = np.mean(np.log(x[ip][:-1] / x[ip][1:]))
    zeta_med = delta / math.sqrt(4 * math.pi**2 + delta**2)
    return abs(zeta_med - zeta) < 1e-3, f"δ = {delta:.4f} → ζ medido = {zeta_med:.4f} (real {zeta:.4f})"


@prueba('B', 'Amortiguamiento crítico (ζ = 1): vuelve al equilibrio SIN cruzarlo')
def _():
    c_crit = 2 * math.sqrt(k * M)
    t, x, _ = resolver(ninguna, x0=0.01, tf=1, c=c_crit)
    exacta = 0.01 * (1 + wn * t) * np.exp(-wn * t)
    return np.min(x) >= -1e-12 and np.max(np.abs(x - exacta)) < 1e-9, \
        f"c_crít = {c_crit:.0f} N·s/m, min x = {np.min(x):.1e}, error vs analítica = {np.max(np.abs(x-exacta)):.1e}"


@prueba('B', 'Superposición: x(x0,v0,F) = x(x0,v0,0) + x(0,0,F)')
def _():
    F = construir_F('giro constante', m0, e, rpm_a_rad_s(300), None)
    _, x_total, _ = resolver(F, x0=0.01, v0=-0.2)
    _, x_libre, _ = resolver(ninguna, x0=0.01, v0=-0.2)
    _, x_forz, _ = resolver(F)
    err = np.max(np.abs(x_total - (x_libre + x_forz)))
    return err < 1e-15, f"diferencia máx = {err:.1e} m (RK4 es lineal: solo redondeo)"


@prueba('B', 'Solo importa el producto m0·e: (m0=2, e=0,15) ≡ (m0=1,2, e=0,25)')
def _():
    w = rpm_a_rad_s(800)
    _, xa, _ = resolver(construir_F('giro constante', 2.0, 0.15, w, None))
    _, xb, _ = resolver(construir_F('giro constante', 1.2, 0.25, w, None))
    err = np.max(np.abs(xa - xb))
    return err < 1e-15, f"diferencia máx = {err:.1e} m"


@prueba('B', 'Curva de resonancia: amplitud simulada = F0/√D para 12 velocidades')
def _():
    peor = 0
    for N in (60, 150, 200, 250, 280, 300, 320, 350, 450, 700, 1000, 1400):
        w = rpm_a_rad_s(N)
        t, x, _ = resolver(construir_F('giro constante', m0, e, w, None), tf=6, h=0.0005)
        X_sim = np.max(np.abs(x[t > 4]))
        peor = max(peor, abs(X_sim / amplitud_teorica(w) - 1))
    return peor < 5e-3, f"peor diferencia relativa = {100*peor:.3f} %"


@prueba('B', 'La resonancia está en N ≈ Nn (302 rpm): el máximo de la curva cae ahí')
def _():
    Ns = np.arange(250, 351, 2)
    X = [amplitud_teorica(rpm_a_rad_s(N)) for N in Ns]
    N_pico = Ns[int(np.argmax(X))]
    N_teo = frecuencia_natural_rpm(M, k) / math.sqrt(1 - 2 * zeta**2)   # pico con desbalance
    return abs(N_pico - N_teo) <= 2, f"pico en {N_pico} rpm (fórmula Nn/√(1-2ζ²) = {N_teo:.1f} rpm, Nn = {frecuencia_natural_rpm(M,k):.1f})"


@prueba('B', 'En resonancia la respuesta va 90° atrasada respecto a la fuerza')
def _():
    w = wn
    F = construir_F('giro constante', m0, e, w, None)
    t, x, _ = resolver(F, tf=8, h=0.0005)
    T = 2 * math.pi / w
    sel = t >= t[-1] - 10 * T                      # 10 periodos completos de estado estable
    a = np.sum(x[sel] * np.sin(w * t[sel]))     # proyección sobre sin (∝ integral)
    b = np.sum(x[sel] * np.cos(w * t[sel]))     # proyección sobre cos
    fase = math.degrees(math.atan2(b, a))
    return abs(fase + 90) < 1, f"desfase medido = {fase:.2f}°"


@prueba('B', 'A muy alta velocidad la amplitud tiende a m0·e/M = 5 mm (por eso centrifugar rápido es seguro)')
def _():
    X = [1000 * amplitud_teorica(rpm_a_rad_s(N)) for N in (1400, 5000, 50000)]
    return abs(X[-1] - 5) < 0.01, "X(1400, 5000, 50000 rpm) = " + ", ".join(f"{x:.3f}" for x in X) + " mm"


@prueba('B', 'Más amortiguamiento → menos pico en resonancia (c ×2 ⇒ X ÷≈2)')
def _():
    w = wn
    X1 = amplitud_teorica(w, c=316)
    X2 = amplitud_teorica(w, c=632)
    t, x, _ = resolver(construir_F('giro constante', m0, e, w, None), tf=6, c=632)
    X2_sim = np.max(np.abs(x[t > 4]))
    return 1.9 < X1 / X2 < 2.1 and abs(X2_sim / X2 - 1) < 0.01, \
        f"X(c=316) = {1000*X1:.1f} mm, X(c=632) = {1000*X2:.1f} mm (simulado {1000*X2_sim:.1f})"


# ==========================================================================
# C. RAMPA
# ==========================================================================
@prueba('C', 'La fórmula de la rampa sale de Newton: F = −m0·d²(e·sin θ)/dt² (derivada numérica)')
def _():
    w_max, tr, d = rpm_a_rad_s(1400), 10.0, 1e-4
    peor = 0
    for t in np.linspace(0.3, 14.7, 40):
        y = lambda s: e * math.sin(angulo_tambor('rampa', s, w_max, tr))
        acel = (y(t + d) - 2 * y(t) + y(t - d)) / d**2
        F_newton = -m0 * acel
        F_codigo = rampa(t, m0, e, w_max, tr)
        peor = max(peor, abs(F_newton - F_codigo) / (m0 * e * w_max**2))
    return peor < 1e-3, f"diferencia máx = {100*peor:.4f} % de la fuerza máxima (40 instantes)"


@prueba('C', 'El ángulo θ(t) de la rampa es continuo en t = tr')
def _():
    w_max, tr = rpm_a_rad_s(1400), 10.0
    salto = abs(angulo_tambor('rampa', tr + 1e-9, w_max, tr) - angulo_tambor('rampa', tr - 1e-9, w_max, tr))
    return salto < 1e-6, f"salto de θ en tr = {salto:.1e} rad"


@prueba('C', 'Después de tr la rampa es un giro constante (|F| máx = m0·e·w²)')
def _():
    w_max, tr = rpm_a_rad_s(1400), 5.0
    ts = np.linspace(6, 8, 20001)
    Fmax = max(abs(rampa(t, m0, e, w_max, tr)) for t in ts)
    return abs(Fmax / (m0 * e * w_max**2) - 1) < 1e-4, f"|F| máx = {Fmax:.1f} N (teórico {m0*e*w_max**2:.1f} N)"


@prueba('C', 'Arranque a 1400 rpm: pasa por resonancia (pico) y termina en la amplitud estable')
def _():
    w_max, tr = rpm_a_rad_s(1400), 10.0
    t, x, _ = resolver(lambda s: rampa(s, m0, e, w_max, tr), tf=16, h=0.0005)
    i = np.argmax(np.abs(x))
    t_cruce = wn / (w_max / tr)                    # instante en que w(t) = wn
    X_final = np.max(np.abs(x[t > 14]))
    ok = t_cruce < t[i] < t_cruce + 1 and abs(X_final / amplitud_teorica(w_max) - 1) < 0.01
    return ok, (f"w = wn en t = {t_cruce:.2f} s, pico de {1000*abs(x[i]):.1f} mm en t = {t[i]:.2f} s, "
                f"final {1000*X_final:.2f} mm")


@prueba('C', 'Cruzar la resonancia RÁPIDO (tr = 5) da menos pico que cruzarla lento (tr = 30)')
def _():
    w_max = rpm_a_rad_s(1400)
    picos_tr = {}
    for tr in (5, 30):
        t, x, _ = resolver(lambda s: rampa(s, m0, e, w_max, tr), tf=tr + 3, h=0.0005)
        picos_tr[tr] = 1000 * np.max(np.abs(x))
    # Físicamente: cuanto MÁS lento se cruza la resonancia, más tiempo tiene para crecer
    return picos_tr[30] > picos_tr[5], \
        f"pico con tr=5: {picos_tr[5]:.1f} mm, con tr=30: {picos_tr[30]:.1f} mm → cruzar rápido es mejor"


# ==========================================================================
# D. SIMULADOR
# ==========================================================================
@prueba('D', 'Auditoría: ningún archivo del método usa scipy.integrate, odeint, solve_ivp ni dsolve')
def _():
    prohibido = ('scipy', 'odeint', 'solve_ivp', 'dsolve', 'sympy')
    culpables = []
    for archivo in ('metodo.py', 'modelo.py', 'excitaciones.py', 'simulador.py', 'validacion.py'):
        for n, linea in enumerate(open(archivo, encoding='utf-8'), 1):
            codigo = linea.split('#')[0]
            if ('import' in codigo) and any(p in codigo for p in prohibido):
                culpables.append(f"{archivo}:{n}")
    return not culpables, "limpio ✔" if not culpables else "encontrado en " + ", ".join(culpables)


@prueba('D', 'Valores derivados del caso base: wn = 31,62, Nn = 302, ζ = 0,100')
def _():
    ok = abs(wn - 31.62) < 0.01 and abs(frecuencia_natural_rpm(M, k) - 302.0) < 0.1 and abs(zeta - 0.0999) < 1e-3
    return ok, f"wn = {wn:.2f} rad/s, Nn = {frecuencia_natural_rpm(M,k):.1f} rpm, ζ = {zeta:.4f}"


@prueba('D', 'Interfaz: arranca, y el aviso de h aparece a 1400 rpm con h = 0,003')
def _():
    sim = simulador.Simulador()
    sin_aviso = sim.texto_aviso.get_text()
    sim.sliders['N'].set_val(1400)
    sim.sliders['h'].set_val(0.003)
    con_aviso = sim.texto_aviso.get_text()
    sim.sliders['h'].set_val(0.002)
    quitado = sim.texto_aviso.get_text()
    ok = 'muy grande' not in sin_aviso and 'muy grande' in con_aviso and 'muy grande' not in quitado
    return ok, "aviso aparece con h = 0,003 y desaparece con h = 0,002"


@prueba('D', 'Interfaz: Euler con h = 0,008 dispara el aviso de divergencia')
def _():
    sim = simulador.Simulador()
    sim.sliders['N'].set_val(1400)
    sim.sliders['tf'].set_val(30)
    sim.sliders['h'].set_val(0.008)
    sim.radio_met.set_active(1)
    return 'diverge' in sim.texto_aviso.get_text(), "aviso: «La solución numérica diverge»"


@prueba('D', 'Interfaz: «Restablecer» devuelve todo a los valores base')
def _():
    sim = simulador.Simulador()
    for s in sim.sliders.values():
        s.set_val(s.valmax)
    sim.radio_exc.set_active(2)
    sim.restablecer(None)
    ok = all(s.val == s.valinit for s in sim.sliders.values()) and sim.radio_exc.value_selected == 'giro constante'
    return ok, "11 sliders y 2 selectores de vuelta al inicio"


@prueba('D', 'Prueba de estrés: 300 combinaciones aleatorias de sliders → RK4 nunca diverge')
def _():
    random.seed(2026)
    fallos, diverge_euler = 0, 0
    for _ in range(300):
        p = {clave: random.uniform(vmin, vmax) for clave, _, vmin, vmax, _, _ in simulador.PARAMETROS}
        p['tf'] = random.uniform(1, 3)                       # cortito para que sea rápido
        tipo = random.choice(simulador.EXCITACIONES)
        _, x, _, _ = simulador.simular(p, tipo, 'RK4')
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1:
            fallos += 1
        _, xe, _, _ = simulador.simular(p, tipo, 'Euler')
        if not np.all(np.isfinite(xe)) or np.max(np.abs(xe)) > 1:
            diverge_euler += 1
    return fallos == 0, f"RK4: {fallos}/300 divergen · Euler: {diverge_euler}/300 se van a más de 1 m"


@prueba('D', 'Velocidad: el peor caso (tf = 30, h = 0,0005 → 60 000 pasos) tarda menos de 3 s')
def _():
    p = {clave: v0 for clave, _, _, _, v0, _ in simulador.PARAMETROS}
    p.update(tf=30, h=0.0005, N=1400)
    t0 = time.time()
    simulador.simular(p, 'rampa', 'RK4')
    dt = time.time() - t0
    return dt < 3, f"{dt:.2f} s"


# --------------------------------------------------------------------------
# Ejecutar todo y mostrar el reporte
# --------------------------------------------------------------------------
NOMBRES_BLOQUE = {'A': 'MÉTODO NUMÉRICO', 'B': 'FÍSICA DE LA LAVADORA',
                  'C': 'ARRANQUE EN RAMPA', 'D': 'SIMULADOR E INTERFAZ'}

if __name__ == "__main__":
    t_total = time.time()
    for correr in PRUEBAS:
        correr()
    bloque_actual = None
    for bloque, nombre, ok, detalle, dt in RESULTADOS:
        if bloque != bloque_actual:
            print(f"\n━━━ {bloque}. {NOMBRES_BLOQUE[bloque]} ━━━")
            bloque_actual = bloque
        print(f" {'✅' if ok else '❌'} {nombre}")
        print(f"      └─ {detalle}  ({dt:.2f} s)")
    n_ok = sum(r[2] for r in RESULTADOS)
    print(f"\n{'═'*60}\n  {n_ok}/{len(RESULTADOS)} pruebas pasaron  ·  {time.time()-t_total:.1f} s en total\n{'═'*60}")
