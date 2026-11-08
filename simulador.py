"""
simulador.py
Simulador interactivo de la lavadora en centrifugado.

    M x'' + c x' + k x = F(t)

Cada vez que se mueve un control se resuelve de nuevo la ecuación (con NUESTRO
RK4 o Euler, de metodo.py) y se actualizan las gráficas.

Ejecutar:  python simulador.py
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

from metodo import rk4_sistema, euler_sistema
from modelo import (f, frecuencia_natural, frecuencia_natural_rpm,
                    factor_amortiguamiento, rpm_a_rad_s)
from excitaciones import construir_F, angulo_tambor

# --------------------------------------------------------------------------
# Parámetros: (clave, etiqueta, mínimo, máximo, valor base, formato)
# --------------------------------------------------------------------------
PARAMETROS = [
    ('M',  'M  masa total [kg]',          30,     80,    50,    '%.1f'),
    ('m0', 'm0 masa desbalanceada [kg]',  0.1,    3,     1,     '%.2f'),
    ('e',  'e  excentricidad [m]',        0.15,   0.30,  0.25,  '%.3f'),
    ('k',  'k  rigidez [N/m]',            1e4,    1e5,   5e4,   '%.0f'),
    ('c',  'c  amortiguamiento [N·s/m]',  50,     2000,  316,   '%.0f'),
    ('N',  'N  velocidad de giro [rpm]',  0,      1400,  300,   '%.0f'),
    ('tr', 'tr tiempo de arranque [s]',   5,      30,    10,    '%.1f'),
    ('x0', 'x0 posición inicial [m]',     -0.02,  0.02,  0,     '%.4f'),
    ('v0', 'v0 velocidad inicial [m/s]',  -0.5,   0.5,   0,     '%.3f'),
    ('h',  'h  paso de malla [s]',        0.0005, 0.01,  0.001, '%.4f'),
    ('tf', 'tf tiempo de simulación [s]', 1,      30,    5,     '%.1f'),
]
EXCITACIONES = ('giro constante', 'rampa', 'ninguna')
METODOS = ('RK4', 'Euler')


# --------------------------------------------------------------------------
# Lógica (separada de la interfaz): resolver con unos parámetros dados
# --------------------------------------------------------------------------
def simular(p, tipo, metodo):
    """
    p      : diccionario con M, m0, e, k, c, N, tr, x0, v0, h, tf
    tipo   : 'giro constante', 'rampa' o 'ninguna'
    metodo : 'RK4' o 'Euler'
    Devuelve t, x, v, F (arreglos) y la función F(t).
    """
    w = rpm_a_rad_s(p['N'])
    F = construir_F(tipo, p['m0'], p['e'], w, p['tr'])
    parametros = {'M': p['M'], 'c': p['c'], 'k': p['k'], 'F': F}

    n_pasos = round(p['tf'] / p['h'])
    resolver = rk4_sistema if metodo == 'RK4' else euler_sistema

    with np.errstate(all='ignore'):          # si Euler diverge, que no llene la consola
        t, Y = resolver(lambda t, Y: f(t, Y, parametros), 0.0,
                        [p['x0'], p['v0']], p['h'], n_pasos)
    Fv = np.array([F(ti) for ti in t])
    return t, Y[:, 0], Y[:, 1], Fv


def amplitud_estable(p):
    """Amplitud teórica del estado estable a giro constante: X = F0/sqrt(D)."""
    w = rpm_a_rad_s(p['N'])
    F0 = p['m0'] * p['e'] * w**2
    D = (p['k'] - p['M'] * w**2)**2 + (p['c'] * w)**2
    return F0 / math.sqrt(D)


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
class Simulador:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 9.5))
        self.fig.canvas.manager.set_window_title("Simulador lavadora - centrifugado")
        self.fig.suptitle("Lavadora en centrifugado:  M x'' + c x' + k x = F(t)",
                          fontsize=14, fontweight='bold')

        # ---- Ejes de las gráficas ----
        self.ax_x = self.fig.add_axes([0.06, 0.64, 0.56, 0.28])
        self.ax_F = self.fig.add_axes([0.06, 0.41, 0.56, 0.17], sharex=self.ax_x)
        self.ax_fase = self.fig.add_axes([0.68, 0.56, 0.18, 0.36])

        self.linea_x, = self.ax_x.plot([], [], color='tab:red', lw=1.2, label='x(t)')
        self.linea_env, = self.ax_x.plot([], [], 'k--', lw=1,
                                         label='± amplitud estable teórica')
        self.linea_env2, = self.ax_x.plot([], [], 'k--', lw=1)
        self.ax_x.set_ylabel('x [mm]')
        self.ax_x.set_title('Gráfica 1: desplazamiento vertical de la máquina')
        self.ax_x.grid(alpha=0.3)
        self.ax_x.legend(loc='upper right', fontsize=8)
        plt.setp(self.ax_x.get_xticklabels(), visible=False)

        self.linea_F, = self.ax_F.plot([], [], color='tab:blue', lw=1, label='F(t)')
        self.ax_F.set_xlabel('t [s]')
        self.ax_F.set_ylabel('F [N]')
        self.ax_F.set_title('Gráfica 2: fuerza de excitación', fontsize=10)
        self.ax_F.grid(alpha=0.3)
        self.ax_F.legend(loc='upper right', fontsize=8)

        self.linea_fase, = self.ax_fase.plot([], [], color='tab:purple', lw=0.8)
        self.punto_ini, = self.ax_fase.plot([], [], 'go', ms=6, label='inicio')
        self.ax_fase.set_xlabel('x [mm]')
        self.ax_fase.set_ylabel("x' [mm/s]")
        self.ax_fase.set_title('Plano de fase', fontsize=10)
        self.ax_fase.grid(alpha=0.3)
        self.ax_fase.legend(loc='upper right', fontsize=8)

        # ---- Textos informativos ----
        self.texto_info = self.fig.text(0.665, 0.47, '', fontsize=9.5, family='monospace',
                                        va='top')
        self.texto_aviso = self.fig.text(0.06, 0.335, '', fontsize=10, color='red',
                                         fontweight='bold')

        # ---- Sliders (dos columnas) ----
        self.sliders = {}
        for i, (clave, etiqueta, vmin, vmax, v0, fmt) in enumerate(PARAMETROS):
            col, fila = i % 2, i // 2
            x_izq = 0.17 if col == 0 else 0.62
            y = 0.29 - fila * 0.045
            ax_s = self.fig.add_axes([x_izq, y, 0.25, 0.025])
            s = Slider(ax_s, etiqueta, vmin, vmax, valinit=v0, valfmt=fmt)
            s.label.set_fontsize(9)
            s.on_changed(self.actualizar)
            self.sliders[clave] = s

        # ---- Radio buttons ----
        ax_r1 = self.fig.add_axes([0.885, 0.76, 0.105, 0.14])
        ax_r1.set_title('Excitación', fontsize=9)
        self.radio_exc = RadioButtons(ax_r1, EXCITACIONES)
        self.radio_exc.on_clicked(self.actualizar)

        ax_r2 = self.fig.add_axes([0.885, 0.62, 0.105, 0.10])
        ax_r2.set_title('Método', fontsize=9)
        self.radio_met = RadioButtons(ax_r2, METODOS)
        self.radio_met.on_clicked(self.actualizar)

        # ---- Botones ----
        ax_b1 = self.fig.add_axes([0.885, 0.53, 0.105, 0.045])
        self.boton_reset = Button(ax_b1, 'Restablecer')
        self.boton_reset.on_clicked(self.restablecer)

        ax_b2 = self.fig.add_axes([0.885, 0.47, 0.105, 0.045])
        self.boton_anim = Button(ax_b2, 'Animar máquina')
        self.boton_anim.on_clicked(self.animar)

        self.anim = None
        self.actualizar(None)

    # ------------------------------------------------------------------
    def leer_parametros(self):
        """Lee el valor actual de todos los sliders en un diccionario."""
        return {clave: s.val for clave, s in self.sliders.items()}

    def actualizar(self, _):
        """Se llama cada vez que cambia un control: resuelve y redibuja."""
        p = self.leer_parametros()
        tipo = self.radio_exc.value_selected
        metodo = self.radio_met.value_selected

        t, x, v, Fv = simular(p, tipo, metodo)
        self.ultimo = (p, tipo, t, x)

        # Gráficas (x en mm)
        self.linea_x.set_data(t, 1000 * x)
        self.linea_x.set_label(f'x(t)  [{metodo}]')
        self.linea_F.set_data(t, Fv)
        self.linea_fase.set_data(1000 * x, 1000 * v)
        self.punto_ini.set_data([1000 * p['x0']], [1000 * p['v0']])

        # Envolvente teórica solo para giro constante
        if tipo == 'giro constante' and p['N'] > 0:
            X = 1000 * amplitud_estable(p)
            self.linea_env.set_data([0, p['tf']], [X, X])
            self.linea_env2.set_data([0, p['tf']], [-X, -X])
        else:
            X = None
            self.linea_env.set_data([], [])
            self.linea_env2.set_data([], [])
        self.ax_x.legend(loc='upper right', fontsize=8)

        for ax in (self.ax_x, self.ax_F, self.ax_fase):
            ax.relim()
            ax.autoscale_view()
        self.ax_x.set_xlim(0, p['tf'])

        # Texto con valores derivados
        wn = frecuencia_natural(p['M'], p['k'])
        Nn = frecuencia_natural_rpm(p['M'], p['k'])
        zeta = factor_amortiguamiento(p['M'], p['c'], p['k'])
        w = rpm_a_rad_s(p['N'])
        n_pasos = round(p['tf'] / p['h'])
        finito = np.all(np.isfinite(x))
        xmax = 1000 * np.max(np.abs(x)) if finito else float('inf')
        info = (f"wn   = {wn:7.2f} rad/s\n"
                f"Nn   = {Nn:7.1f} rpm\n"
                f"zeta = {zeta:7.3f}\n"
                f"w/wn = {w / wn:7.2f}\n"
                f"pasos= {n_pasos:7d}\n"
                f"|x|max = {xmax:9.2f} mm")
        if X is not None:
            info += f"\nX estable = {X:7.2f} mm"
        self.texto_info.set_text(info)

        # Avisos
        avisos = []
        if tipo != 'ninguna' and p['N'] > 0:
            h_max = (60 / p['N']) / 20          # 20 pasos por vuelta del tambor
            if p['h'] > h_max:
                avisos.append(f"⚠ h = {p['h']:.4f} s es muy grande: para {p['N']:.0f} rpm "
                              f"se necesita h ≤ {h_max:.4f} s (20 pasos por vuelta)")
        if not finito or xmax > 1000:
            avisos.append("⚠ La solución numérica diverge (inestabilidad): reduzca h")
        if tipo != 'ninguna' and abs(w / wn - 1) < 0.1 and p['N'] > 0:
            avisos.append("◆ Cerca de RESONANCIA (w ≈ wn)")
        self.texto_aviso.set_text("   ".join(avisos))

        self.fig.canvas.draw_idle()

    def restablecer(self, _):
        for s in self.sliders.values():
            s.reset()
        self.radio_exc.set_active(0)
        self.radio_met.set_active(0)

    # ------------------------------------------------------------------
    def animar(self, _):
        """Abre una ventana con la máquina subiendo y bajando sobre sus resortes."""
        p, tipo, t, x = self.ultimo
        if not np.all(np.isfinite(x)):
            return
        w = rpm_a_rad_s(p['N'])

        fig_a, ax = plt.subplots(figsize=(5, 6))
        fig_a.canvas.manager.set_window_title("Animación de la lavadora")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.3, 3.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.plot([-1.4, 1.4], [0, 0], 'k-', lw=3)        # piso

        xmax = np.max(np.abs(x))
        escala = 0.35 / xmax if xmax > 0 else 1.0       # exagera el movimiento para verlo
        y_base = 1.2                                    # altura de la máquina en equilibrio

        caja = Rectangle((-1.0, y_base), 2.0, 1.6, fc='#dde6f0', ec='k', lw=2)
        ax.add_patch(caja)
        tambor = Circle((0, y_base + 0.8), 0.6, fc='white', ec='gray', lw=2)
        ax.add_patch(tambor)
        masa, = ax.plot([], [], 'o', color='tab:red', ms=12)
        resorte_i, = ax.plot([], [], 'k-', lw=1.5)
        resorte_d, = ax.plot([], [], 'k-', lw=1.5)
        texto = ax.text(-1.4, 3.05, '', fontsize=9, family='monospace')
        ax.set_title("Movimiento vertical exagerado para que se vea (cámara lenta ×0.25)",
                     fontsize=9)

        def zigzag(x_c, y_top):
            """Dibuja un resorte en zigzag de y = 0 a y = y_top."""
            ys = np.linspace(0, y_top, 14)
            xs = x_c + 0.12 * np.array([0] + [(-1) ** j for j in range(12)] + [0])
            return xs, ys

        paso = max(1, round(0.01 / p['h']))             # 0.01 s de simulación por cuadro
        indices = np.arange(0, len(t), paso)

        def cuadro(j):
            i = indices[j]
            dy = escala * x[i]
            y0 = y_base + dy
            caja.set_y(y0)
            tambor.center = (0, y0 + 0.8)
            th = angulo_tambor(tipo, t[i], w, p['tr'])
            masa.set_data([0.45 * math.cos(th)], [y0 + 0.8 + 0.45 * math.sin(th)])
            resorte_i.set_data(*zigzag(-0.7, y0))
            resorte_d.set_data(*zigzag(0.7, y0))
            texto.set_text(f"t = {t[i]:5.2f} s   x = {1000 * x[i]:7.2f} mm")
            return caja, tambor, masa, resorte_i, resorte_d, texto

        self.anim = FuncAnimation(fig_a, cuadro, frames=len(indices),
                                  interval=40, blit=False, repeat=True)
        plt.show(block=False)


if __name__ == "__main__":
    sim = Simulador()
    plt.show()
