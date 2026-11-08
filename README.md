# Simulador de lavadora en centrifugado (Entrega 2 – Ecuaciones Diferenciales)

Modelo: `M x'' + c x' + k x = F(t)`, resuelto con **RK4 para sistemas programado por nosotros**
(sin `scipy.integrate` ni `sympy.dsolve`).

## Instalación

```bash
pip install numpy matplotlib
```

## Ejecución (en el orden de la revisión)

```bash
python validacion.py     # tabla de errores + gráfica numérica vs analítica (guarda validacion.png)
python simulador.py      # interfaz con sliders
python pruebas.py        # 32 pruebas automáticas (método, física, rampa, interfaz) ~20 s
python galeria.py        # 6 gráficas de las pruebas en la carpeta galeria/
```

## Archivos

| Archivo | Qué contiene |
|---|---|
| `metodo.py` | `rk4_sistema(f, t0, Y0, h, N)` y `euler_sistema(...)` |
| `modelo.py` | el sistema `f(t, Y, parametros)` y `wn`, `Nn`, `zeta` |
| `excitaciones.py` | `giro_constante`, `rampa`, `ninguna` y `construir_F` |
| `validacion.py` | solución analítica, gráficas y tabla de errores |
| `simulador.py` | interfaz (Slider, RadioButtons, botones, animación) |
| `pruebas.py` | 32 pruebas automáticas con ✅/❌ |
| `galeria.py` | genera las gráficas de `galeria/` |
| `GUIA_DEFENSA.md` | pseudocódigo, explicación y preguntas de práctica |

## Uso del simulador

- 11 sliders (M, m0, e, k, c, N, tr, x0, v0, h, tf), excitación (giro constante / rampa / ninguna) y método (RK4 / Euler).
- Gráfica 1: x(t) en mm (con la amplitud estable teórica en giro constante). Gráfica 2: F(t). Plano de fase.
- Texto con wn, Nn, zeta, w/wn, número de pasos y |x| máximo.
- Aviso rojo si h > (60/N)/20 (menos de 20 pasos por vuelta), si la solución diverge o si hay resonancia.
- Botón **Animar máquina**: la lavadora sobre sus resortes (movimiento exagerado, cámara lenta).

## Galería de pruebas

| | |
|---|---|
| ![](galeria/1_curva_resonancia.png) | ![](galeria/4_arranque_rampa.png) |
| ![](galeria/2_energia_euler_vs_rk4.png) | ![](galeria/3_estabilidad_euler.png) |
| ![](galeria/5_decremento_logaritmico.png) | ![](galeria/6_eficiencia_error_vs_trabajo.png) |
