# Censo de subgrafos de 3 nodos y 2 aristas en CPU y GPU

Prototipo de tesis sobre minería de subgrafos frecuentes (FSM) acelerada en GPU.

El programa cuenta todos los subgrafos conexos de 3 nodos y 2 aristas de un grafo etiquetado no dirigido. Esos subgrafos son siempre caminos `a – c – b`. Luego los agrupa por patrón de etiquetas, con **conteo no inducido**.

El repositorio incluye:

- una versión secuencial en Python, que sirve como referencia,
- una versión paralela en CUDA (Numba) que trabaja sobre el grafo en formato CSR,
- una demo que ejecuta varios experimentos y compara ambas versiones, incluido un grafo real de millones de nodos.

## Contenido

| Archivo | Descripción |
|---|---|
| `demo.py` | Demo completa: un solo comando ejecuta todos los experimentos y genera el informe |
| `censo_3n2a.py` | Versión secuencial (CPU) con dos estrategias: exploración y centro. Incluye generadores de grafos, lectura de archivos gSpan y dibujo |
| `censo_3n2a_gpu.py` | Versión paralela (GPU) sobre CSR: 3 kernels × 3 modos de conteo, con verificación contra la CPU |
| `datos_reales.py` | Descarga grafos de SNAP, los convierte a CSR, les asigna etiquetas y los guarda en caché |
| `requirements.txt` | Dependencias de Python. `numba-cuda` se instala aparte, según tu driver |

## Requisitos

- **GPU NVIDIA con soporte CUDA** para la parte paralela. Sin GPU, la demo ejecuta solo la parte de CPU.
- **Driver NVIDIA reciente.** En WSL2 el driver se instala en Windows, no dentro de Linux.
- **Python 3.10 a 3.13.**
- **Memoria para el grafo real por defecto (LiveJournal):** unos 8 GB de RAM, ~1 GB de memoria de GPU y ~1 GB de disco para la descarga y la caché.
- **Para Orkut** (opcional, 3,5 veces más grande): se recomiendan 16 GB de RAM y ~3 GB de memoria de GPU.

Entorno donde se desarrolló y probó: WSL2 con Ubuntu, Python 3.13, NVIDIA RTX 5060 (8 GB) y driver 610.57 (CUDA 13.3).

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
```

### 2. Crear un entorno e instalar las dependencias

Con conda o mamba (recomendado):

```bash
conda create -n censo-gpu python=3.12 -y
conda activate censo-gpu
pip install -r requirements.txt
```

También funciona con `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Instalar `numba-cuda` según la versión de CUDA del driver

Primero mira la versión que muestra `nvidia-smi`, arriba a la derecha (`CUDA Version`):

```bash
nvidia-smi
```

Luego instala el paquete que corresponda:

```bash
pip install "numba-cuda[cu12]"
```

Usa el comando anterior si tu driver muestra CUDA 12.x, y este si muestra CUDA 13.x:

```bash
pip install "numba-cuda[cu13]"
```

Estos paquetes traen las librerías de CUDA que Numba necesita, así que no hace falta instalar el CUDA Toolkit completo. Las GPUs de la serie RTX 50 necesitan un driver con CUDA 12.8 o superior.

Si ya tienes un entorno de RAPIDS, `numba-cuda` viene incluido. En ese caso basta con `pip install -r requirements.txt`.

### 4. Verificar que Numba ve la GPU

```bash
python -c "from numba import cuda; print(cuda.is_available()); cuda.detect()"
```

Debe imprimir `True` y listar la GPU.

## Ejecutar la demo

```bash
python demo.py
```

La primera vez descarga el grafo de LiveJournal desde SNAP y lo guarda en caché en `datasets/`. Las siguientes ejecuciones lo cargan en segundos.

El tiempo total depende de la máquina. Una estimación es de unos 5 a 10 minutos, más la descarga inicial. La mayor parte del tiempo se va en la versión de CPU en Python y en leer el grafo real la primera vez.

### Opciones

| Opción | Efecto |
|---|---|
| `--rapido` | Grafos más chicos y sin grafo real. Sirve para comprobar que todo funciona en unos 2 minutos |
| `--dataset livejournal` / `orkut` / `twitter` / `ninguno` | Grafo real para la parte solo GPU. Por defecto `livejournal` |
| `--sin-gpu` | Ejecuta solo la parte de CPU |
| `--reps N` | Repeticiones por medición (por defecto 3) |
| `--etiquetas-nodo N`, `--etiquetas-arista N` | Etiquetas sintéticas del grafo real (por defecto 5 y 2) |
| `--trabajo-por-lanzamiento N` | Máximo de caminos por lanzamiento de kernel (por defecto 1.000.000.000) |
| `--salida`, `--datos` | Carpetas de resultados y de datasets |

Ejemplos:

```bash
python demo.py --rapido
python demo.py --dataset orkut
python demo.py --sin-gpu
```

### Archivos generados

Cada ejecución crea una carpeta `resultados/demo_<fecha>_<hora>/` con:

- `resumen.md`: el informe completo con todas las tablas.
- `resultados.json`: todos los datos medidos, para analizarlos después.
- `cpu_vs_gpu.png`: gráfico de tiempos CPU vs GPU en escala logarítmica.

## Qué hace la demo

0. **Entorno.** Muestra las versiones de Python y las librerías, la CPU, la RAM y la GPU detectada.
1. **Validación de correctitud.** Cuenta grafos chicos con los 2 métodos de CPU y las 9 configuraciones de GPU, y comprueba que todos los censos sean idénticos. Además, la GPU emite el identificador de cada camino (el par de IDs de sus dos aristas) y se verifica que el conjunto sea igual al de la CPU, sin duplicados.
2. **Estrategias secuenciales.** Compara exploración contra centro en tiempo, memoria y trabajo descartado.
3. **CPU vs GPU sobre el mismo grafo.** Usa un grafo libre de escala (Barabási–Albert), uno aleatorio denso y uno completo. Muestra la tabla de las 9 configuraciones GPU y la aceleración contra el mejor método de CPU.
4. **GPU a gran escala.** Grafo completo de 1000 nodos (498.501.000 caminos). El tiempo de CPU se estima con el rendimiento medido en el experimento 3.
5. **Grafo real masivo (solo GPU).** com-LiveJournal de SNAP: 3.997.962 nodos y 34.681.189 aristas. El tiempo de CPU se estima.
6. **Resumen.** Tabla consolidada y gráfico.

## Cómo funciona

### El problema

Todo subgrafo conexo de 3 nodos y 2 aristas es un camino `a – c – b`, con un nodo central `c`. El patrón se identifica por la etiqueta del centro y sus dos ramas. Cada rama es la etiqueta de la arista más la del nodo extremo. Las ramas se ordenan, así que `A–C–B` y `B–C–A` son el mismo patrón.

En pantalla, un patrón se escribe así:

```
O -[d]- C -[s]- C
```

Las mayúsculas son etiquetas de nodo, las minúsculas entre corchetes son etiquetas de arista, y el centro va en el medio.

El total de caminos siempre debe ser `Σ C(grado, 2)` sobre todos los nodos, y ambos programas lo verifican en cada ejecución.

**Conteo no inducido:** un camino cuenta aunque sus extremos también estén unidos, así que un triángulo aporta 3 caminos. El conteo inducido se obtiene restando 3 × (número de triángulos).

### Estrategias secuenciales (`censo_3n2a.py`)

- **Exploración.** Recorre desde cada nodo, da un paso por una arista y luego otro. Así encuentra cada camino 4 veces. Para no contarlo de más, guarda el ID de cada camino ya visto en un conjunto hash y descarta los repetidos. Siempre descarta el 75% de lo que genera, y la memoria crece con el número de caminos.
- **Centro.** Cada camino tiene un único nodo central. Cada nodo forma los pares `(i, j)` de sus vecinos con `i < j`, de modo que cada camino aparece exactamente una vez. No necesita recordar lo visto y su memoria es constante.

### Versión paralela (`censo_3n2a_gpu.py`)

Usa la regla del centro, que no necesita estructuras compartidas entre hilos. Recibe el grafo en CSR: `offsets`, `indices` y etiquetas como enteros. Cada patrón se codifica como un entero único, sin colisiones.

**Kernels** (cómo se reparte el trabajo):

| Kernel | Cada hilo procesa | Comentario |
|---|---|---|
| `nodo` | Un nodo y todos sus pares de vecinos | Simple, pero un hub deja a un hilo con casi todo el trabajo |
| `entrada` | Una entrada del CSR (un vecino `i` y todos los `j > i`) | Reparto más fino |
| `balanceado` | Un bloque fijo de 64 caminos, ubicado con prefix sum, búsqueda binaria y "desrankeo" del par inicial | Balance casi perfecto |

**Modos de conteo:**

- `global`: un histograma en memoria global, con sumas atómicas.
- `shared`: un histograma por bloque en memoria compartida, que se vuelca al final. Solo se usa si hay como máximo 2048 patrones posibles.
- `hash`: una tabla hash en GPU con `atomicCAS`. Sirve para cualquier número de etiquetas.

En los tres modos, cada hilo agrupa los patrones repetidos consecutivos antes de la operación atómica.

**Lanzamientos por lotes.** El trabajo se divide en varios lanzamientos de kernel de hasta 1.000 millones de caminos cada uno. En GPUs que también controlan la pantalla, sobre todo en Windows y WSL2, el driver puede abortar los kernels muy largos (TDR).

### Métricas de la tabla GPU

| Columna | Significado |
|---|---|
| Hilos | Hilos lanzados (bloques × hilos por bloque) |
| Lanz. | Número de lanzamientos de kernel |
| H2D | Copia de datos a la GPU y reserva de memoria |
| Kernel (cálculo) | El conteo en sí |
| D2H | Traer los resultados y traducir los enteros a patrones con etiquetas |
| Total | H2D + cálculo + D2H. No incluye la compilación JIT, que se paga en la ejecución de calentamiento |
| Desbal | Trabajo del hilo más cargado / trabajo promedio por hilo. 1,00 = reparto perfecto. Es un modelo analítico que cuenta también los hilos sin trabajo |
| OK | El censo coincide con la referencia y el total es `Σ C(grado, 2)` |

Criterio de medición: la CPU usa el mejor de N repeticiones y la GPU el promedio de N tras una ejecución de calentamiento. Ese criterio no favorece a la GPU.

## Uso individual de los programas

Versión secuencial:

```bash
python censo_3n2a.py --grafo ba --n 20000 --k 3 --comparar --memoria
python censo_3n2a.py --grafo ejemplo --dibujar --json
python censo_3n2a.py --archivo ejemplos/moleculas.txt --dibujar
```

Versión GPU:

```bash
python censo_3n2a_gpu.py --grafo ba --n 50000 --k 3 --reps 3 --verificar-ids
python censo_3n2a_gpu.py --grafo completo --n 1000 --sin-secuencial
python censo_3n2a_gpu.py --csr grafo.npz --kernel balanceado --conteo hash
```

Con `--help` se ven todas las opciones. Los generadores disponibles son `ejemplo`, `triangulo`, `estrella`, `completo`, `aleatorio` y `ba`.

El formato CSR para `--csr` es un `.npz` con estos arreglos:

- `offsets` (n + 1),
- `indices` (cada arista en ambos sentidos),
- `etiqueta_nodo`,
- y opcionalmente `etiqueta_arista`, `nombres_nodo` y `nombres_arista`.

## Datos reales

| Opción | Grafo | Nodos | Aristas | Fuente |
|---|---|---|---|---|
| `livejournal` (por defecto) | com-LiveJournal | 3.997.962 | 34.681.189 | [SNAP](https://snap.stanford.edu/data/com-LiveJournal.html) |
| `orkut` | com-Orkut | 3.072.441 | 117.185.083 | [SNAP](https://snap.stanford.edu/data/com-Orkut.html) |
| `twitter` | ego-Twitter (se trata como no dirigido) | — | — | [SNAP](https://snap.stanford.edu/data/ego-Twitter.html) |

Los grafos de SNAP no traen etiquetas. La demo asigna etiquetas uniformes al azar con una semilla fija, por defecto 5 de nodo y 2 de arista. Cada arista tiene la misma etiqueta en ambos sentidos. Se eliminan los lazos y las aristas repetidas, y los IDs de nodo se renumeran de 0 a n − 1.

Si la descarga falla (por ejemplo, por un proxy), descarga el `.gz` a mano desde la página de SNAP y déjalo en `datasets/` con su nombre original:

- `com-lj.ungraph.txt.gz`
- `com-orkut.ungraph.txt.gz`
- `twitter_combined.txt.gz`

Cita de los datos: Jure Leskovec y Andrej Krevl, *SNAP Datasets: Stanford Large Network Dataset Collection*, http://snap.stanford.edu/data, 2014.

## Solución de problemas

- **`cuda.is_available()` devuelve `False`:** revisa que `nvidia-smi` funcione y que el extra de `numba-cuda` (`cu12` o `cu13`) coincida con la versión de CUDA del driver.
- **Error `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`:** el driver es más antiguo que las librerías de CUDA instaladas. Actualiza el driver de NVIDIA.
- **La pantalla se congela o aparece `CUDA_ERROR_LAUNCH_TIMEOUT`:** reduce el trabajo por lanzamiento, por ejemplo con `python demo.py --trabajo-por-lanzamiento 200000000`.
- **`MemoryError` o el proceso muere al cargar Orkut:** no hay RAM suficiente; usa `livejournal`. En WSL2, la RAM disponible se configura en `.wslconfig`.
- **Advertencias `NumbaPerformanceWarning`:** aparecen con grafos chicos y son inofensivas. Se pueden ocultar con `NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0`.
- **Sin GPU disponible:** la demo continúa con la parte de CPU e indica el motivo en la sección de entorno.

## Limitaciones conocidas

- La referencia de CPU es Python puro, así que la aceleración frente a una CPU compilada sería menor.
- La construcción de los grafos sintéticos es secuencial en Python.
- El soporte por transacción (para bases de varios grafos) solo existe en la versión secuencial.
