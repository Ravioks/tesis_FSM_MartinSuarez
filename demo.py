import argparse
import gc
import json
import math
import os
import platform
import random
import time
import traceback
import tracemalloc
from datetime import datetime

import numpy as np

import censo_3n2a as sec

try:
    from numba import cuda
    import censo_3n2a_gpu as gpu
    ERROR_GPU = None
except Exception as e:
    cuda = None
    gpu = None
    ERROR_GPU = f"{type(e).__name__}: {e}"

ESCALAS = {
    "completa": {
        "validacion": [("ejemplo", {}), ("triangulo", {}), ("estrella", {"n": 50}), ("ba", {"n": 300, "k": 2}),
                       ("aleatorio", {"n": 200, "m": 600, "L": 4, "E": 3})],
        "estrategias": [("estrella", {"n": 1000}), ("aleatorio", {"n": 10000, "m": 50000}), ("ba", {"n": 20000, "k": 3})],
        "cpu_gpu": [("ba", {"n": 50000, "k": 3}), ("aleatorio", {"n": 1000, "m": 49950}), ("completo", {"n": 300})],
        "gran_escala": [("completo", {"n": 1000})],
        "dataset": "livejournal",
    },
    "rapida": {
        "validacion": [("ejemplo", {}), ("triangulo", {}), ("ba", {"n": 300, "k": 2})],
        "estrategias": [("aleatorio", {"n": 3000, "m": 15000}), ("ba", {"n": 5000, "k": 3})],
        "cpu_gpu": [("ba", {"n": 10000, "k": 3}), ("completo", {"n": 150})],
        "gran_escala": [("completo", {"n": 500})],
        "dataset": "ninguno",
    },
}


def num(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def veces(x):
    if x is None:
        return "-"
    return f"{num(x, 2 if x < 1 else 1 if x < 100 else 0)}x"


def mb(x):
    return f"{num(x, 2)} MB"


class Informe:
    def __init__(self):
        self.md = []
        self.datos = {}

    def titulo(self, texto):
        linea = "=" * 78
        print(f"\n{linea}\n {texto}\n{linea}")
        self.md.append(f"\n## {texto}\n")

    def texto(self, t, consola=True):
        if consola:
            print(t)
        self.md.append(t + "\n")

    def tabla(self, encabezados, filas, consola=True):
        celdas = [[str(c) for c in f] for f in filas]
        anchos = [max(len(str(e)), *(len(f[i]) for f in celdas)) if celdas else len(str(e)) for i, e in enumerate(encabezados)]
        if consola:
            print("  ".join(str(e).ljust(a) if i == 0 else str(e).rjust(a) for i, (e, a) in enumerate(zip(encabezados, anchos))))
            print("  ".join("-" * a for a in anchos))
            for f in celdas:
                print("  ".join(c.ljust(a) if i == 0 else c.rjust(a) for i, (c, a) in enumerate(zip(f, anchos))))
        self.md.append("| " + " | ".join(str(e) for e in encabezados) + " |")
        self.md.append("|" + "|".join(["---"] + ["---:"] * (len(encabezados) - 1)) + "|")
        for f in celdas:
            self.md.append("| " + " | ".join(f) + " |")
        self.md.append("")


def crear_grafo(tipo, n=1000, m=5000, k=3, L=3, E=2, semilla=42):
    rng = random.Random(semilla)
    en, ea = sec.alfabeto(L, True), sec.alfabeto(E, False)
    if tipo == "ejemplo":
        return sec.generar_ejemplo()
    if tipo == "triangulo":
        return sec.generar_triangulo()
    if tipo == "estrella":
        return sec.generar_estrella(n)
    if tipo == "completo":
        return sec.generar_completo(n, rng, en, ea)
    if tipo == "aleatorio":
        return sec.generar_aleatorio(n, m, rng, en, ea)
    return sec.generar_ba(n, k, rng, en, ea)


def caminos(g):
    return sum(math.comb(len(a), 2) for a in g.adyacencia.values())


def cronometrar(fn, *args):
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    try:
        resultado = fn(*args)
    finally:
        t = time.perf_counter() - t0
        gc.enable()
    return resultado, t


def medir_metodo(g, metodo, reps):
    tiempos = []
    censo, info = {}, {}
    for _ in range(reps):
        (censo, info), t = cronometrar(sec.METODOS[metodo], g)
        tiempos.append(t)
    return censo, info, min(tiempos)


def memoria_pico(g, metodo):
    gc.collect()
    tracemalloc.start()
    sec.METODOS[metodo](g)
    pico = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return pico / 2**20


def detectar_gpu(sin_gpu):
    if sin_gpu:
        return False, "desactivada con --sin-gpu"
    if gpu is None:
        return False, f"Numba / numba-cuda no disponible ({ERROR_GPU})"
    try:
        if not cuda.is_available():
            return False, "Numba no detecta una GPU CUDA (revisa el driver y la instalación de numba-cuda)"
        cuda.to_device(np.zeros(1))
        cuda.synchronize()
        return True, ""
    except Exception as e:
        return False, f"error al inicializar CUDA: {e}"


def describir_entorno(usar_gpu):
    d = {"python": platform.python_version(), "sistema": platform.platform(), "numpy": np.__version__}
    for modulo, clave in (("numba", "numba"), ("numba_cuda", "numba_cuda"), ("pandas", "pandas")):
        try:
            d[clave] = __import__(modulo).__version__
        except Exception:
            d[clave] = "no instalado"
    try:
        with open("/proc/cpuinfo") as f:
            d["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        d["cpu"] = platform.processor() or "desconocida"
    try:
        d["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        d["ram_gb"] = None
    if usar_gpu:
        try:
            dev = cuda.get_current_device()
            nombre = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
            d["gpu"] = nombre
            d["capacidad_computo"] = ".".join(str(x) for x in dev.compute_capability)
        except Exception:
            d["gpu"] = "desconocida"
        libre, total = gpu.memoria_gpu()
        if total:
            d["vram_total_gb"] = round(total / 2**30, 2)
            d["vram_libre_gb"] = round(libre / 2**30, 2)
        try:
            v = cuda.cudadrv.driver.driver.get_version()
            d["cuda_driver"] = f"{v[0]}.{v[1]}"
        except Exception:
            pass
    return d


def mejor_fila(filas):
    validas = [f for f in filas if not f.get("omitida")]
    return min(validas, key=lambda f: f["t_total_s"]) if validas else None


def tabla_gpu_md(inf, filas):
    encabezados = ["Kernel", "Conteo", "Hilos", "Lanz.", "H2D", "Kernel (cálculo)", "D2H", "Total", "Desbal", "OK"]
    datos = []
    for f in filas:
        if f.get("omitida"):
            datos.append([f["kernel"], f["conteo"], "-", "-", "-", "omitida", "-", "-", "-", f["omitida"]])
            continue
        datos.append([f["kernel"], f["conteo"], num(f["hilos"]), f["lanzamientos"], sec.seg(f["t_h2d_s"]),
                      sec.seg(f["t_kernel_s"]), sec.seg(f["t_d2h_s"]), sec.seg(f["t_total_s"]),
                      num(f["desbalance"], 2), "sí" if f["correcto"] else "NO"])
    inf.tabla(encabezados, datos)


def exp_validacion(inf, escala, usar_gpu, p):
    inf.titulo("1. Validación de correctitud (CPU vs GPU, camino por camino)")
    inf.texto("Cada grafo se cuenta con los 2 métodos de CPU y con todas las configuraciones GPU válidas.\n"
              "Además, la GPU emite el ID (par de aristas) de cada camino y se compara con el conjunto de la CPU.")
    filas, datos = [], []
    for tipo, kw in escala["validacion"]:
        g = crear_grafo(tipo, **kw)
        W = caminos(g)
        ce, _ = sec.censo_exploracion(g)
        cc, _ = sec.censo_centro(g)
        cpu_ok = ce == cc and sum(cc.values()) == W
        gpu_txt, ids_txt = "-", "-"
        if usar_gpu:
            csr = gpu.grafo_a_csr(g)
            prep = gpu.preparar(csr)
            ok = total = 0
            for kernel in gpu.KERNELS:
                for conteo in gpu.MODOS:
                    if not gpu.config_valida(conteo, prep["P"]):
                        continue
                    r = gpu.ejecutar_gpu(csr, prep, kernel, conteo, p.bloque, p.max_bloques, p.chunk,
                                         trabajo_max=p.trabajo_por_lanzamiento)
                    gpu.liberar_gpu()
                    total += 1
                    ok += r["censo"] == cc
            v = gpu.verificar_ids(csr, prep, g, p.bloque, p.max_bloques, p.chunk, p.trabajo_por_lanzamiento)
            gpu_txt = f"{ok}/{total}"
            ids_txt = f"{v['estado']} ({v.get('duplicados_gpu', 0)} duplicados)" if "duplicados_gpu" in v else v["estado"]
        filas.append([g.nombre, num(len(g.etiqueta_nodo)), num(len(g.aristas)), num(W), len(cc),
                      "sí" if cpu_ok else "NO", gpu_txt, ids_txt])
        datos.append({"grafo": g.nombre, "caminos": W, "patrones": len(cc), "cpu_ok": cpu_ok,
                      "gpu_configs_ok": gpu_txt, "ids": ids_txt})
    inf.tabla(["Grafo", "Nodos", "Aristas", "Caminos", "Patrones", "CPU expl = centro", "GPU configs OK", "IDs GPU = CPU"], filas)
    inf.datos["validacion"] = datos


def exp_estrategias(inf, escala, p, rendimientos):
    inf.titulo("2. Estrategias secuenciales: exploración (hash de IDs) vs centro (regla de pertenencia)")
    inf.texto(f"Tiempo = mejor de {p.reps} repeticiones. La memoria se mide aparte con tracemalloc.")
    filas, datos = [], []
    for tipo, kw in escala["estrategias"]:
        g = crear_grafo(tipo, **kw)
        W = caminos(g)
        ce, ie, te = medir_metodo(g, "exploracion", p.reps)
        cc, _, tc = medir_metodo(g, "centro", p.reps)
        me, mc = memoria_pico(g, "exploracion"), memoria_pico(g, "centro")
        rendimientos.append(W / tc if tc > 0 else 0)
        pct = 100 * ie["descartados"] / ie["generados"] if ie["generados"] else 0
        filas.append([g.nombre, num(W), sec.seg(te), sec.seg(tc), veces(te / tc if tc else None), mb(me), mb(mc),
                      f"{num(ie['descartados'])} ({num(pct, 1)}%)", "sí" if ce == cc else "NO"])
        datos.append({"grafo": g.nombre, "caminos": W, "t_exploracion_s": te, "t_centro_s": tc,
                      "memoria_exploracion_mb": me, "memoria_centro_mb": mc,
                      "generados": ie["generados"], "descartados": ie["descartados"], "iguales": ce == cc})
        print(f"  listo: {g.nombre}")
    inf.tabla(["Grafo", "Caminos", "Exploración", "Centro", "Aceleración", "Mem. exploración", "Mem. centro",
               "Descartados (expl.)", "Mismo censo"], filas)
    inf.datos["estrategias"] = datos


def exp_cpu_gpu(inf, escala, usar_gpu, p, rendimientos, resumen):
    inf.titulo("3. CPU vs GPU sobre el mismo grafo")
    inf.texto(f"CPU = método centro (el mejor secuencial), mejor de {p.reps} (1 si hay más de 5 M de caminos). "
              f"GPU = promedio de {p.reps} tras 1 ejecución de calentamiento; Total = H2D + cálculo + D2H.")
    filas, datos = [], []
    for tipo, kw in escala["cpu_gpu"]:
        g = crear_grafo(tipo, **kw)
        W = caminos(g)
        cc, _, tc = medir_metodo(g, "centro", p.reps if W <= 5_000_000 else 1)
        rendimientos.append(W / tc if tc > 0 else 0)
        fila_res = {"grafo": g.nombre, "caminos": W, "cpu_s": tc, "cpu_estimado": False, "gpu_s": None, "config": None}
        if not usar_gpu:
            filas.append([g.nombre, num(W), sec.seg(tc), "-", "-", "-", "-", "-"])
            datos.append({"grafo": g.nombre, "caminos": W, "t_cpu_centro_s": tc})
            resumen.append(fila_res)
            continue
        csr = gpu.grafo_a_csr(g)
        prep = gpu.preparar(csr)
        filas_gpu, _ = gpu.medir_gpu(csr, prep, gpu.KERNELS, list(gpu.MODOS), p.bloque, p.max_bloques, p.chunk,
                                     p.reps, censo_ref=cc, t_cpu={"centro": tc}, trabajo_max=p.trabajo_por_lanzamiento)
        inf.texto(f"\n{g.nombre}: {num(W)} caminos, CPU centro {sec.seg(tc)}")
        tabla_gpu_md(inf, filas_gpu)
        mejor = mejor_fila(filas_gpu)
        todos_ok = all(f["correcto"] for f in filas_gpu if not f.get("omitida"))
        filas.append([g.nombre, num(W), sec.seg(tc), f"{mejor['kernel']} + {mejor['conteo']}", sec.seg(mejor["t_kernel_s"]),
                      sec.seg(mejor["t_total_s"]), veces(tc / mejor["t_total_s"]), "sí" if todos_ok else "NO"])
        fila_res.update({"gpu_s": mejor["t_total_s"], "config": f"{mejor['kernel']} + {mejor['conteo']}"})
        resumen.append(fila_res)
        datos.append({"grafo": g.nombre, "caminos": W, "t_cpu_centro_s": tc, "gpu": filas_gpu, "todos_correctos": todos_ok})
    inf.texto("\nResumen del experimento 3:")
    inf.tabla(["Grafo", "Caminos", "CPU centro", "Mejor GPU", "Cálculo GPU", "Total GPU", "Aceleración", "Censo = CPU"], filas)
    inf.datos["cpu_gpu"] = datos


def estimar_cpu(W, rendimientos):
    r = max(rendimientos) if rendimientos else 0
    return W / r if r > 0 else None


def exp_gran_escala(inf, escala, p, rendimientos, resumen):
    inf.titulo("4. GPU a gran escala (sin CPU: el tiempo de CPU se estima)")
    inf.texto("El tiempo de CPU se estima con el mejor rendimiento de 'centro' medido en esta máquina (caminos/s).")
    datos = []
    for tipo, kw in escala["gran_escala"]:
        t0 = time.perf_counter()
        g = crear_grafo(tipo, **kw)
        csr = gpu.grafo_a_csr(g)
        t_construccion = time.perf_counter() - t0
        del g
        prep = gpu.preparar(csr)
        W = prep["W"]
        filas_gpu, censo = gpu.medir_gpu(csr, prep, gpu.KERNELS, list(gpu.MODOS), p.bloque, p.max_bloques, p.chunk,
                                         p.reps, trabajo_max=p.trabajo_por_lanzamiento)
        t_cpu = estimar_cpu(W, rendimientos)
        inf.texto(f"\n{csr.nombre}: {num(W)} caminos | construcción del grafo (Python): {sec.seg(t_construccion)} | "
                  f"CPU centro estimado: {sec.seg(t_cpu) if t_cpu else '-'}")
        tabla_gpu_md(inf, filas_gpu)
        mejor = mejor_fila(filas_gpu)
        if mejor:
            resumen.append({"grafo": csr.nombre, "caminos": W, "cpu_s": t_cpu, "cpu_estimado": True,
                            "gpu_s": mejor["t_total_s"], "config": f"{mejor['kernel']} + {mejor['conteo']}"})
        datos.append({"grafo": csr.nombre, "caminos": W, "t_cpu_estimado_s": t_cpu, "gpu": filas_gpu})
    inf.datos["gran_escala"] = datos


def exp_real(inf, clave, p, rendimientos, resumen):
    import datos_reales as dr

    info = dr.DATASETS[clave]
    inf.titulo(f"5. Grafo real masivo: {info['nombre']} (SNAP), solo GPU")
    inf.texto(f"Fuente: {info['pagina']}\n"
              f"Los grafos de SNAP no traen etiquetas: se asignan {p.etiquetas_nodo} etiquetas de nodo y "
              f"{p.etiquetas_arista} de arista al azar (uniformes, semilla {p.semilla}).")
    csr, tiempos, meta = dr.cargar_dataset(clave, p.etiquetas_nodo, p.etiquetas_arista, p.semilla, p.datos)
    for k, v in tiempos.items():
        print(f"  {k.replace('_s', '').replace('_', ' ')}: {sec.seg(v)}")
        inf.md.append(f"- {k.replace('_s', '').replace('_', ' ')}: {sec.seg(v)}")
    inf.md.append("")
    prep = gpu.preparar(csr)
    W = prep["W"]
    m = csr.nnz // 2
    dmax = int(prep["grados"].max()) if csr.n else 0
    coincide = ""
    if info["nodos"]:
        coincide = " (coincide con SNAP)" if (csr.n, m) == (info["nodos"], info["aristas"]) else \
            f" (SNAP reporta {num(info['nodos'])} nodos y {num(info['aristas'])} aristas)"
    inf.texto(f"Nodos: {num(csr.n)} | aristas: {num(m)}{coincide}\n"
              f"Grado máximo: {num(dmax)} | caminos W = Σ C(d,2): {num(W)} | espacio de patrones P: {num(prep['P'])}")
    hilos_nodo = gpu.hilos_teoricos(csr, prep, "nodo", p.bloque, p.max_bloques, p.chunk)
    des_nodo = gpu.desbalance(csr, prep, "nodo", hilos_nodo, p.chunk)
    inf.texto(f"Kernel 'nodo' omitido: un solo hilo tendría que procesar C({num(dmax)}, 2) = {num(math.comb(dmax, 2))} "
              f"caminos (desbalance teórico {num(des_nodo, 0)}). Se usan 'entrada' y 'balanceado'.")
    libre, total = gpu.memoria_gpu()
    if libre:
        inf.texto(f"Memoria GPU libre: {num(libre / 2**30, 2)} GB de {num(total / 2**30, 2)} GB")
    reps = min(p.reps, 2)
    filas_gpu, censo = gpu.medir_gpu(csr, prep, ["entrada", "balanceado"], list(gpu.MODOS), p.bloque, p.max_bloques,
                                     p.chunk, reps, trabajo_max=p.trabajo_por_lanzamiento)
    tabla_gpu_md(inf, filas_gpu)
    t_cpu = estimar_cpu(W, rendimientos)
    mejor = mejor_fila(filas_gpu)
    if mejor:
        tasa = mejor["caminos_por_s_kernel"]
        inf.texto(f"Mejor configuración: {mejor['kernel']} + {mejor['conteo']}, {sec.seg(mejor['t_total_s'])} en total"
                  + (f" ({num(tasa / 1e9, 1)} mil millones de caminos/s en el cálculo)." if tasa else "."))
        if t_cpu:
            horas = f" ({num(t_cpu / 3600, 1)} horas)" if t_cpu >= 3600 else ""
            inf.texto(f"CPU centro estimado para este grafo: {sec.seg(t_cpu)}{horas}.")
        resumen.append({"grafo": info["nombre"], "caminos": W, "cpu_s": t_cpu, "cpu_estimado": True,
                        "gpu_s": mejor["t_total_s"], "config": f"{mejor['kernel']} + {mejor['conteo']}"})
    if censo:
        total_censo = sum(censo.values())
        top = sorted(censo.items(), key=lambda x: -x[1])[:5]
        inf.texto(f"Patrones distintos: {len(censo)} | total = W: {'sí' if total_censo == W else 'NO'} | top 5:")
        inf.tabla(["Patrón", "Frecuencia", "%"],
                  [[sec.formato_patron(k), num(v), f"{num(100 * v / total_censo, 2)}%"] for k, v in top])
    inf.datos["grafo_real"] = {"dataset": clave, "nombre": info["nombre"], "nodos": csr.n, "aristas": m,
                               "grado_max": dmax, "caminos": W, "P": prep["P"], "tiempos_carga_s": tiempos,
                               "desbalance_teorico_nodo": des_nodo, "t_cpu_estimado_s": t_cpu, "gpu": filas_gpu,
                               "cache": meta.get("cache")}


def graficar(resumen, ruta):
    filas = [r for r in resumen if r.get("gpu_s") and r.get("cpu_s")]
    if not filas:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return False
    x = np.arange(len(filas))
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(filas)), 5.5))
    barras = ax.bar(x - ancho / 2, [r["cpu_s"] for r in filas], ancho, color="#8c8c8c", label="CPU (centro, Python)")
    for b, r in zip(barras, filas):
        if r["cpu_estimado"]:
            b.set_hatch("//")
            b.set_alpha(0.55)
    ax.bar(x + ancho / 2, [r["gpu_s"] for r in filas], ancho, color="#2a9d8f", label="GPU (mejor configuración)")
    for i, r in enumerate(filas):
        ax.annotate(veces(r["cpu_s"] / r["gpu_s"]), (x[i] + ancho / 2, r["gpu_s"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Tiempo (s, escala logarítmica)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['grafo']}\n{num(r['caminos'])} caminos" for r in filas], fontsize=8)
    manejadores, _ = ax.get_legend_handles_labels()
    manejadores.append(Patch(facecolor="#8c8c8c", alpha=0.55, hatch="//", label="CPU estimado"))
    ax.legend(handles=manejadores, loc="upper left", fontsize=9)
    ax.set_title("Censo de subgrafos 3 nodos / 2 aristas: CPU vs GPU")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return True


def ejecutar(nombre, fn, inf, errores, *args):
    try:
        fn(inf, *args)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        errores.append({"experimento": nombre, "error": f"{type(e).__name__}: {e}"})
        inf.texto(f"\n[ERROR] {nombre}: {type(e).__name__}: {e}\n(se continúa con el siguiente experimento)")
        traceback.print_exc()


def crear_parser():
    p = argparse.ArgumentParser(description="Demo: censo de subgrafos de 3 nodos y 2 aristas en CPU y GPU")
    p.add_argument("--rapido", action="store_true", help="Grafos más chicos y sin grafo real (unos 2 minutos)")
    p.add_argument("--dataset", choices=["livejournal", "orkut", "twitter", "ninguno"], default=None,
                   help="Grafo real de SNAP para la parte solo GPU (por defecto livejournal; ninguno con --rapido)")
    p.add_argument("--sin-gpu", action="store_true", help="Ejecuta solo la parte de CPU")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--etiquetas-nodo", type=int, default=5, help="Etiquetas de nodo para el grafo real")
    p.add_argument("--etiquetas-arista", type=int, default=2, help="Etiquetas de arista para el grafo real")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="resultados")
    p.add_argument("--datos", default="datasets", help="Carpeta para descargas y caché de grafos reales")
    p.add_argument("--bloque", type=int, default=256)
    p.add_argument("--max-bloques", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--trabajo-por-lanzamiento", type=int, default=1_000_000_000)
    return p


def main():
    p = crear_parser().parse_args()
    p.reps = max(1, p.reps)
    escala = ESCALAS["rapida" if p.rapido else "completa"]
    dataset = p.dataset or escala["dataset"]
    carpeta = os.path.join(p.salida, "demo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(carpeta, exist_ok=True)
    inicio = time.perf_counter()

    inf = Informe()
    print("=" * 78)
    print(" DEMO — Censo de subgrafos de 3 nodos y 2 aristas (conteo no inducido), CPU vs GPU")
    print("=" * 78)
    inf.md.append("# Demo: censo de subgrafos de 3 nodos y 2 aristas (CPU vs GPU)\n")
    inf.md.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')} | escala: {'rápida' if p.rapido else 'completa'}"
                  f" | repeticiones: {p.reps}\n")

    usar_gpu, motivo = detectar_gpu(p.sin_gpu)
    inf.titulo("0. Entorno")
    entorno = describir_entorno(usar_gpu)
    for k, v in entorno.items():
        print(f"  {k}: {v}")
        inf.md.append(f"- {k}: {v}")
    inf.md.append("")
    if not usar_gpu:
        inf.texto(f"  GPU: no se usará ({motivo}). Se ejecuta solo la parte de CPU.")
    inf.datos["entorno"] = entorno

    errores, rendimientos, resumen = [], [], []
    ejecutar("validación", exp_validacion, inf, errores, escala, usar_gpu, p)
    ejecutar("estrategias", exp_estrategias, inf, errores, escala, p, rendimientos)
    ejecutar("cpu_gpu", exp_cpu_gpu, inf, errores, escala, usar_gpu, p, rendimientos, resumen)
    if usar_gpu:
        ejecutar("gran_escala", exp_gran_escala, inf, errores, escala, p, rendimientos, resumen)
        if dataset != "ninguno":
            ejecutar("grafo_real", exp_real, inf, errores, dataset, p, rendimientos, resumen)

    inf.titulo("6. Resumen")
    filas = []
    for r in resumen:
        cpu_txt = "-" if r["cpu_s"] is None else sec.seg(r["cpu_s"]) + (" (estimado)" if r["cpu_estimado"] else "")
        acel = veces(r["cpu_s"] / r["gpu_s"]) if r.get("gpu_s") and r.get("cpu_s") else "-"
        filas.append([r["grafo"], num(r["caminos"]), cpu_txt, r["config"] or "-",
                      sec.seg(r["gpu_s"]) if r.get("gpu_s") else "-", acel])
    if filas:
        inf.tabla(["Grafo", "Caminos", "CPU centro", "Mejor GPU", "Total GPU", "Aceleración"], filas)
    inf.texto("La aceleración compara contra Python puro; contra una CPU compilada la diferencia sería menor.")
    inf.datos["resumen"] = resumen
    inf.datos["errores"] = errores

    duracion = time.perf_counter() - inicio
    inf.texto(f"\nDuración total de la demo: {sec.seg(duracion)}")
    if errores:
        inf.texto(f"Experimentos con error: {', '.join(e['experimento'] for e in errores)}")

    ruta_png = os.path.join(carpeta, "cpu_vs_gpu.png")
    if graficar(resumen, ruta_png):
        inf.md.append("\n![CPU vs GPU](cpu_vs_gpu.png)\n")
    else:
        ruta_png = None
    with open(os.path.join(carpeta, "resumen.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(inf.md))
    with open(os.path.join(carpeta, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(inf.datos, f, ensure_ascii=False, indent=2, default=str)
    print("\nArchivos generados:")
    for nombre in ["resumen.md", "resultados.json"] + (["cpu_vs_gpu.png"] if ruta_png else []):
        print(f"  {os.path.join(carpeta, nombre)}")


if __name__ == "__main__":
    main()
