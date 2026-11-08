# Guía de defensa – Simulador de la lavadora

Orden de la revisión (15 min): **método (2) → validación (3) → simulador (7) → preguntas (3)**.
Esta guía va en ese mismo orden. En cada parte encuentran: pseudocódigo → código → explicación → preguntas de práctica (con respuesta).

---

## PARTE 1 – `metodo.py` (2 min)

### Idea en una frase
RK4 avanza de `t_i` a `t_i + h` usando un **promedio ponderado de 4 pendientes**: una al inicio, dos a la mitad y una al final del paso. Como la ecuación es de 2º orden, primero la volvemos un sistema de 2 ecuaciones de 1er orden, y el mismo RK4 se aplica a un **vector** `Y = [x, x']`.

### Pseudocódigo ↔ código (línea a línea)

| # | Pseudocódigo | Código (`rk4_sistema`) |
|---|---|---|
| 1 | Crear arreglos t (N+1) y Y (N+1 × 2) | `t = np.zeros(N + 1)` / `Y = np.zeros((N + 1, len(Y0)))` |
| 2 | t₀ ← t0, Y₀ ← Y0 | `t[0] = t0` / `Y[0] = Y0` |
| 3 | Para i = 0, …, N−1: | `for i in range(N):` |
| 4 | K1 ← f(tᵢ, Yᵢ) | `K1 = f(t[i], Y[i])` |
| 5 | K2 ← f(tᵢ + h/2, Yᵢ + (h/2)·K1) | `K2 = f(t[i] + h/2, Y[i] + (h/2)*K1)` |
| 6 | K3 ← f(tᵢ + h/2, Yᵢ + (h/2)·K2) | `K3 = f(t[i] + h/2, Y[i] + (h/2)*K2)` |
| 7 | K4 ← f(tᵢ + h, Yᵢ + h·K3) | `K4 = f(t[i] + h, Y[i] + h*K3)` |
| 8 | Yᵢ₊₁ ← Yᵢ + (h/6)(K1 + 2K2 + 2K3 + K4) | `Y[i+1] = Y[i] + (h/6)*(K1 + 2*K2 + 2*K3 + K4)` |
| 9 | tᵢ₊₁ ← t0 + (i+1)·h | `t[i+1] = t0 + (i + 1)*h` |
| 10 | Devolver t, Y | `return t, Y` |

Euler es lo mismo pero las líneas 4–8 se cambian por una sola: `Yᵢ₊₁ ← Yᵢ + h·f(tᵢ, Yᵢ)`.

### Explicación sencilla
- **¿Por qué vectores?** `Y[i]` es `[x_i, v_i]`. `f` devuelve `[x', x'']`. Entonces `K1` es un vector de 2 componentes y `Y[i] + (h/2)*K1` mueve **posición y velocidad al mismo tiempo**. numpy hace la suma componente a componente.
- **¿Por qué `t0 + (i+1)*h` y no `t[i] + h`?** Sumar h miles de veces acumula error de redondeo; multiplicar no.
- **¿Por qué 1/6 y 2-2?** Son los pesos de la regla de Simpson: los puntos medios pesan el doble. La suma de pesos 1+2+2+1 = 6, por eso se divide entre 6.
- **Orden 4:** el error global es proporcional a h⁴. Si h se divide entre 2, el error se divide entre 2⁴ = 16.

### Preguntas de práctica
1. **"¿Qué es K2 físicamente?"** → Es la estimación de `[velocidad, aceleración]` en la mitad del paso, usando el punto al que llegaríamos avanzando medio paso con K1.
2. **"¿Cuántas veces evalúa f por paso RK4? ¿Y Euler?"** → 4 y 1. RK4 cuesta 4 veces más por paso, pero con el mismo h es muchísimo más preciso (ver tabla: 10⁻⁹ contra 10⁻³).
3. **"¿Por qué F(t) tiene que poder evaluarse en t + h/2?"** → Porque K2 y K3 evalúan f en `t_i + h/2`, que no es un nodo. Por eso F es una función de t y no un arreglo de valores.

---

## PARTE 2 – `modelo.py` y `excitaciones.py`

### Pseudocódigo del modelo
```
función f(t, Y, parametros):
    y1 ← Y[0]                      (x)
    y2 ← Y[1]                      (x')
    dy1 ← y2
    dy2 ← (F(t) − c·y2 − k·y1) / M
    devolver [dy1, dy2]
```
Sale de despejar x'' de `M x'' + c x' + k x = F(t)`.

- `wn = sqrt(k/M)` → 31,62 rad/s; `Nn = 60·wn/(2π)` → 302 rpm; `zeta = c/(2·sqrt(kM))` → 0,100.
- `parametros` es un diccionario `{M, c, k, F}`, y `F` es **una función** de t. En el método se usa `lambda t, Y: f(t, Y, parametros)` para que el método reciba una f de solo (t, Y), como en la guía.

### Excitaciones
- **Giro constante:** `F = m0·e·w²·sin(w t)`: la componente vertical de la fuerza centrífuga.
- **Rampa:** mientras arranca, `w(t) = α t` y `θ = α t²/2`. Aparece un término extra `−α cos θ` por la **aceleración tangencial** de la masa. Después de `tr` es igual al giro constante, pero con la fase `θ` acumulada.
- **Ninguna:** `F = 0` → vibración libre amortiguada (usar x0 o v0 ≠ 0 para ver algo).
- `construir_F(tipo, m0, e, w, tr)` devuelve una función de **una sola variable** t, con los parámetros "metidos adentro" (closure / lambda).
- `angulo_tambor` solo sirve para la animación.

### Preguntas de práctica
1. **"¿Por qué hay que pasar a sistema?"** → Los métodos de la guía son para `y' = f(t, y)` (1er orden). Con `y1 = x`, `y2 = x'` tenemos dos ecuaciones de 1er orden, y RK4 se aplica igual pero con vectores.
2. **"¿Qué pasa con F en la rampa en t = tr?"** → θ es continua (la fórmula de después suma `α tr²/2`). El término `−α cos θ` desaparece porque ya no hay aceleración angular, así que F tiene un pequeño salto, que es físico: la aceleración angular se corta de golpe.
3. **"¿Dónde está m0 en el modelo?"** → En F(t). M ya incluye a m0 (masa total), por eso se pide m0 < M.

---

## PARTE 3 – `validacion.py` (3 min)

### Qué hace
1. Resuelve el caso base (300 rpm, x0 = v0 = 0, t ∈ [0, 2] s).
2. Compara con la solución analítica `x = e^{−ζωn t}(C1 cos ωd t + C2 sin ωd t) + A cos ωt + B sin ωt`.
3. `e(h) = max_k |x_exacta(t_k) − x_k|` para h = 0,004; 0,002; 0,001 y 0,0005.

### Resultado real (corrido)
```
RK4 (se espera razón ≈ 2^4 = 16)
     h [s] |     e(h) [m] | e(2h)/e(h)
    0.0040 |   5.1550e-07 |          -
    0.0020 |   3.2145e-08 |      16.04
    0.0010 |   2.0066e-09 |      16.02
    0.0005 |   1.2531e-10 |      16.01
Orden estimado p ≈ 4.00

Euler (se espera razón ≈ 2^1 = 2)
    0.0040 |   3.5458e-02 |          -
    0.0020 |   1.0881e-02 |       3.26
    0.0010 |   4.5064e-03 |       2.41
    0.0005 |   2.0722e-03 |       2.17
Orden estimado p ≈ 1.12
```

### Cómo leerlo (lo que deben decir)
- RK4: la razón ≈ 16 = 2⁴ → **orden 4 confirmado**. El orden se calcula con `p = log₂(razón)`.
- Euler: la razón **va bajando hacia 2** (3,26 → 2,41 → 2,17). Con h grandes todavía no está en la "zona asintótica" (h es grande comparado con el periodo de 0,2 s y el sistema está casi en resonancia, lo que amplifica el error). Al reducir h se acerca a 2 → orden 1.
- La gráfica log-log: las pendientes de las rectas son el orden (4 para RK4, 1 para Euler).
- Con h = 0,001, el error de RK4 es ~2·10⁻⁹ m (nanómetros) y el de Euler ~4,5 mm. Euler incluso **exagera la amplitud** (se ve en la gráfica 1).

### Preguntas de práctica
1. **"¿Por qué validan a 300 rpm?"** → Es el caso base y está casi en resonancia (w/wn = 0,99), que es el caso más exigente: amplitud grande (24,8 mm) y transitorio largo.
2. **"¿Por qué la razón tiende a 16?"** → `e(h) ≈ C·h⁴` ⇒ `e(2h)/e(h) = C(2h)⁴ / (C h⁴) = 16`.
3. **"¿Qué significan C1 y C2?"** → Las constantes de la parte homogénea (transitorio), ajustadas para que se cumplan x(0) = x0 y x'(0) = v0. A y B son la parte particular (estado estable).

---

## PARTE 4 – `simulador.py` (7 min de demo)

### Estructura
- `simular(p, tipo, metodo)` = **lógica pura**: arma F, arma el diccionario, calcula `n_pasos = round(tf/h)` y llama a `rk4_sistema` o `euler_sistema`.
- `Simulador.actualizar()` se conecta a **todos** los sliders y radios con `on_changed` / `on_clicked`: lee los valores, resuelve otra vez y actualiza las líneas con `set_data`.
- Aviso: `h_max = (60/N)/20`, porque una vuelta dura 60/N segundos y queremos al menos 20 pasos por vuelta.

### Guion de demo sugerido (≈ 7 min)
1. **Caso base (300 rpm):** "w/wn = 0,99 → resonancia. La amplitud crece hasta la línea punteada, que es la amplitud estable teórica X = F0/√D = 24,8 mm. Coincide."
2. **Subir N a 1000–1400 rpm:** la amplitud **baja** a ~5 mm. Esto pasa porque, muy por encima de la resonancia, X → m0·e/M = 5 mm. Por eso las lavadoras centrifugan a alta velocidad sin saltar.
3. **Bajar N a 100 rpm:** X ≈ 0,6 mm, porque la fuerza es pequeña (∝ w²).
4. **Rampa, N = 1400, tr = 10, tf = 15–30:** la máquina **pasa por la resonancia** (w = wn en t ≈ 2,2 s y el pico de ~24 mm llega en t ≈ 2,5 s, con un pequeño retraso) y después se calma a ~5,2 mm. Así funciona el centrifugado real.
5. **Subir c (amortiguamiento):** el pico de resonancia baja. Con c = 2000 → ζ ≈ 0,63.
6. **Cambiar M o k:** Nn cambia en el texto. Si Nn se mueve, la resonancia aparece en otro N.
7. **Ninguna + x0 = 0,02:** vibración libre amortiguada, una espiral hacia el origen en el plano de fase.
8. **h y método:** con N = 1400 y h = 0,008 aparece el **aviso rojo**. Al cambiar a Euler la solución **explota** (aviso de divergencia). Volver a RK4 o bajar h.
9. **Animar máquina:** la lavadora sobre sus resortes con el tambor girando.

### Preguntas de práctica
1. **"¿Por qué Euler explota con h = 0,008 y RK4 no?"** → Para la respuesta libre, Euler multiplica cada paso por un factor de módulo `√(1 − 2hζωn + h²ωn²)`, que es mayor que 1 si `h > 2ζ/ωn ≈ 0,0063 s`. Entonces el error crece en cada paso. La región de estabilidad de RK4 es mucho más grande (para este problema h·ωn ≈ 0,25 está muy dentro).
2. **"¿Qué pasa si `tf/h` no es entero?"** → Se usa `round(tf/h)`, así que el último tiempo queda a lo sumo h/2 del tf pedido. El número exacto de pasos aparece en pantalla.
3. **"¿Dónde se resuelve la ecuación cuando muevo un slider?"** → `on_changed(self.actualizar)` → `simular(...)` → `rk4_sistema(...)`. No se usa scipy en ninguna parte.

---

## Preguntas sueltas que pueden caer (3 min)

- **¿Qué es ζ < 1?** Subamortiguado: oscila mientras decae. ζ = 1 es crítico y ζ > 1 sobreamortiguado.
- **¿Qué pasa exactamente en w = wn?** La amplitud es máxima y vale ≈ F0/(c·w). Solo el amortiguamiento la limita, por eso c es clave.
- **¿Por qué x se mide desde el equilibrio estático?** Porque así el peso Mg y la deflexión estática del resorte se cancelan y no aparecen en la ecuación.
- **¿Cuántos pasos hace el peor caso?** tf = 30, h = 0,0005 → 60 000 pasos ≈ 0,85 s en Python. Por eso el slider puede tardar un poco.
