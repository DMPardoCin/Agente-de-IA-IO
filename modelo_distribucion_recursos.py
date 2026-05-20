"""
=============================================================
MODELO DE DISTRIBUCIÓN DE RECURSOS EN REDES (CPM)
Investigación de Operaciones — Técnicas de Planeación de Redes
Grupo 9/10: Distribución de Recursos

NOVEDAD: Integración con IA local (llamafile Mistral-7B-Instruct-v0.2)
         para análisis y recomendaciones sobre los resultados CPM.
=============================================================
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import json
import textwrap
import urllib.request
import urllib.error

# ──────────────────────────────────────────────
# 1. DATOS DEL PROYECTO
# ──────────────────────────────────────────────

ACTIVIDADES = {
    "A": {"nombre": "Excavación",         "duracion": 3, "recursos": 4, "predecesoras": []},
    "B": {"nombre": "Cimentación",        "duracion": 5, "recursos": 6, "predecesoras": ["A"]},
    "C": {"nombre": "Estructura metálica","duracion": 4, "recursos": 8, "predecesoras": ["B"]},
    "D": {"nombre": "Inst. eléctrica",    "duracion": 3, "recursos": 3, "predecesoras": ["B"]},
    "E": {"nombre": "Inst. hidráulica",   "duracion": 2, "recursos": 3, "predecesoras": ["B"]},
    "F": {"nombre": "Techado",            "duracion": 3, "recursos": 5, "predecesoras": ["C"]},
    "G": {"nombre": "Acabados",           "duracion": 4, "recursos": 6, "predecesoras": ["D", "E", "F"]},
    "H": {"nombre": "Equipamiento",       "duracion": 2, "recursos": 4, "predecesoras": ["G"]},
}
RECURSOS_DISPONIBLES = 12  # trabajadores por día

# ──────────────────────────────────────────────
# 2. CONSTRUCCIÓN DE LA RED
# ──────────────────────────────────────────────

def construir_red(actividades):
    G = nx.DiGraph()
    for act_id, datos in actividades.items():
        G.add_node(act_id, **datos)
    for act_id, datos in actividades.items():
        for pred in datos["predecesoras"]:
            G.add_edge(pred, act_id)
    return G

# ──────────────────────────────────────────────
# 3. ALGORITMO CPM
# ──────────────────────────────────────────────

def calcular_cpm(G, actividades):
    orden = list(nx.topological_sort(G))
    ES, EF = {}, {}

    for act in orden:
        preds = list(G.predecessors(act))
        ES[act] = max((EF[p] for p in preds), default=0)
        EF[act] = ES[act] + actividades[act]["duracion"]

    T_star = max(EF.values())

    LF, LS = {}, {}
    for act in reversed(orden):
        succs = list(G.successors(act))
        LF[act] = min((LS[s] for s in succs), default=T_star)
        LS[act] = LF[act] - actividades[act]["duracion"]

    TF = {act: LS[act] - ES[act] for act in orden}

    resultados = []
    for act in orden:
        resultados.append({
            "Actividad":   act,
            "Nombre":      actividades[act]["nombre"],
            "Duración":    actividades[act]["duracion"],
            "Recursos":    actividades[act]["recursos"],
            "ES":          ES[act],
            "EF":          EF[act],
            "LS":          LS[act],
            "LF":          LF[act],
            "Holgura (TF)": TF[act],
            "Crítica":     TF[act] == 0,
        })

    df = pd.DataFrame(resultados)
    return df, T_star, ES, EF, LS, LF, TF

# ──────────────────────────────────────────────
# 4. RUTA CRÍTICA
# ──────────────────────────────────────────────

def obtener_ruta_critica(G, TF):
    criticas = [a for a, tf in TF.items() if tf == 0]
    subg = G.subgraph(criticas)
    rutas = []
    inicios = [n for n in subg.nodes if subg.in_degree(n) == 0]
    fines   = [n for n in subg.nodes if subg.out_degree(n) == 0]
    for ini in inicios:
        for fin in fines:
            for ruta in nx.all_simple_paths(subg, ini, fin):
                rutas.append(ruta)
    return criticas, rutas

# ──────────────────────────────────────────────
# 5. PERFIL DE RECURSOS
# ──────────────────────────────────────────────

def calcular_perfil_recursos(actividades, ES, EF, T_star, inicio_override=None):
    perfil = defaultdict(int)
    inicio = inicio_override or ES
    for act_id, datos in actividades.items():
        s = inicio[act_id]
        for t in range(s, s + datos["duracion"]):
            perfil[t] += datos["recursos"]
    return [perfil.get(t, 0) for t in range(T_star)]

def nivelar_recursos(actividades, ES, EF, LS, LF, TF, T_star):
    inicio_actual = dict(ES)
    con_holgura = sorted(
        [(a, TF[a]) for a in actividades if TF[a] > 0],
        key=lambda x: -x[1]
    )
    for act_id, holgura in con_holgura:
        mejor_varianza = np.var(
            calcular_perfil_recursos(actividades, ES, EF, T_star, inicio_actual)
        )
        mejor_inicio = inicio_actual[act_id]
        for delta in range(1, holgura + 1):
            candidato = dict(inicio_actual)
            candidato[act_id] = ES[act_id] + delta
            perfil = calcular_perfil_recursos(actividades, ES, EF, T_star, candidato)
            v = np.var(perfil)
            if v < mejor_varianza:
                mejor_varianza = v
                mejor_inicio = ES[act_id] + delta
        inicio_actual[act_id] = mejor_inicio
    return inicio_actual

# ──────────────────────────────────────────────
# 6. INTEGRACIÓN CON IA LOCAL (llamafile)
# ──────────────────────────────────────────────

LLAMAFILE_URL  = "http://localhost:8080/v1/chat/completions"
LLAMAFILE_MODEL = "mistral-7b-instruct-v0.2.Q2_K"  # nombre informativo

def _llamar_ia(prompt: str, max_tokens: int = 800, temperatura: float = 0.7) -> str:
    """
    Envía un prompt al servidor llamafile local y devuelve la respuesta.
    Compatible con la API OpenAI /v1/chat/completions que expone llamafile.
    """
    payload = {
        "model":       LLAMAFILE_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperatura,
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        LLAMAFILE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            resultado = json.loads(resp.read().decode("utf-8"))
            return resultado["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        return (
            f"[⚠ No se pudo conectar con llamafile en {LLAMAFILE_URL}]\n"
            f"  Asegúrate de que el servidor esté corriendo:\n"
            f"  ./mistral-7b-instruct-v0.2.Q2_K.llamafile --server --host 0.0.0.0 --port 8080\n"
            f"  Error técnico: {e}"
        )
    except (KeyError, json.JSONDecodeError) as e:
        return f"[⚠ Error al interpretar la respuesta de llamafile]: {e}"


def construir_contexto_cpm(
    df, rutas_criticas, T_star,
    perfil_original, perfil_nivelado,
    inicio_nivelado, ES, recursos_disp
) -> str:
    """
    Serializa todos los resultados del modelo CPM en texto estructurado
    para que la IA los pueda leer y analizar.
    """
    ruta_str = " → ".join(rutas_criticas[0]) if rutas_criticas else "N/A"

    actividades_info = []
    for _, row in df.iterrows():
        tipo = "CRÍTICA" if row["Crítica"] else f"holgura={row['Holgura (TF)']}d"
        actividades_info.append(
            f"  {row['Actividad']} ({row['Nombre']}): duración={row['Duración']}d, "
            f"recursos={row['Recursos']} trabajadores, ES={row['ES']}, EF={row['EF']}, "
            f"LS={row['LS']}, LF={row['LF']} [{tipo}]"
        )

    perfil_str = []
    for t, (orig, niv) in enumerate(zip(perfil_original, perfil_nivelado)):
        alerta = " ← SOBRECARGA" if orig > recursos_disp else ""
        perfil_str.append(f"  Día {t}: sin nivelar={orig}, nivelado={niv}{alerta}")

    desplazados = [
        f"  {act}: ES={ES[act]} → inicio nivelado={inicio_nivelado[act]} "
        f"(desplazada {inicio_nivelado[act]-ES[act]}d)"
        for act in df["Actividad"]
        if inicio_nivelado.get(act, ES[act]) != ES[act]
    ]

    var_orig = np.var(perfil_original)
    var_niv  = np.var(perfil_nivelado)
    pico_o   = max(perfil_original)
    pico_n   = max(perfil_nivelado)
    sob_o    = sum(1 for v in perfil_original if v > recursos_disp)
    sob_n    = sum(1 for v in perfil_nivelado  if v > recursos_disp)

    contexto = f"""
RESULTADOS DEL MODELO CPM — DISTRIBUCIÓN DE RECURSOS EN REDES
==============================================================

PROYECTO: Construcción de una planta industrial (8 actividades)
Duración mínima del proyecto: {T_star} días
Recursos disponibles por día: {recursos_disp} trabajadores
Ruta crítica: {ruta_str}

ACTIVIDADES:
{chr(10).join(actividades_info)}

PERFIL DE RECURSOS DÍA A DÍA (límite={recursos_disp} trabajadores/día):
{chr(10).join(perfil_str)}

ACTIVIDADES DESPLAZADAS POR NIVELACIÓN:
{chr(10).join(desplazados) if desplazados else "  Ninguna fue desplazada."}

MÉTRICAS GLOBALES:
  Varianza del perfil:   {var_orig:.2f}  →  {var_niv:.2f}  ({((var_niv-var_orig)/var_orig*100):+.1f}%)
  Pico de recursos:      {pico_o}  →  {pico_n} trabajadores
  Días con sobrecarga:   {sob_o}  →  {sob_n}
"""
    return contexto.strip()


def analizar_con_ia(
    df, rutas_criticas, T_star,
    perfil_original, perfil_nivelado,
    inicio_nivelado, ES, recursos_disp
) -> dict:
    """
    Realiza tres consultas a la IA local con distintos enfoques:
      1. Análisis general de la ruta crítica y los tiempos.
      2. Evaluación de la nivelación de recursos.
      3. Recomendaciones de gestión de riesgos y mejoras.
    Devuelve un dict con las tres respuestas.
    """
    contexto = construir_contexto_cpm(
        df, rutas_criticas, T_star,
        perfil_original, perfil_nivelado,
        inicio_nivelado, ES, recursos_disp
    )

    consultas = {
        "analisis_cpm": {
            "titulo": " ANÁLISIS DE RUTA CRÍTICA Y TIEMPOS",
            "prompt": (
                f"{contexto}\n\n"
                "Eres un experto en Investigación de Operaciones y gestión de proyectos. "
                "Basándote ÚNICAMENTE en los datos anteriores, responde en español:\n"
                "1. ¿Cuáles son las actividades más vulnerables del proyecto y por qué?\n"
                "2. ¿Qué implica la holgura de cada actividad no crítica para el equipo de trabajo?\n"
                "3. ¿Existe algún riesgo de retraso en la ruta crítica dada la demanda de recursos?\n"
                "Sé concreto y menciona actividades por nombre."
            ),
        },
        "evaluacion_nivelacion": {
            "titulo": " EVALUACIÓN DE LA NIVELACIÓN DE RECURSOS",
            "prompt": (
                f"{contexto}\n\n"
                "Eres un especialista en planificación de recursos en proyectos de construcción. "
                "Responde en español:\n"
                "1. ¿Fue efectiva la nivelación aplicada? Sustenta con las métricas.\n"
                "2. ¿Siguen existiendo días problemáticos tras la nivelación? ¿Cuáles y por qué?\n"
                "3. ¿Las actividades desplazadas son buenas candidatas para el desplazamiento "
                "o podría haber mejores opciones?\n"
                "Usa los números del perfil de recursos en tu análisis."
            ),
        },
        "recomendaciones": {
            "titulo": " RECOMENDACIONES Y GESTIÓN DE RIESGOS",
            "prompt": (
                f"{contexto}\n\n"
                "Eres un consultor senior de proyectos industriales. "
                "Basándote en los resultados del modelo CPM, responde en español con "
                "recomendaciones prácticas y accionables:\n"
                "1. ¿Qué tres acciones concretas mejorarían la ejecución del proyecto?\n"
                "2. ¿Cómo mitigarías los riesgos asociados a la ruta crítica?\n"
                "3. Si el cliente exigiera reducir el proyecto en 2 días, ¿qué estrategia "
                "aplicarías (crashing/fast-tracking)?\n"
                "Sé directo y específico, menciona actividades y días concretos."
            ),
        },
    }

    print("\n Consultando IA local (llamafile Mistral-7B)...")
    print("   Esto puede tardar 30–120 segundos según el hardware.\n")

    respuestas = {}
    for clave, info in consultas.items():
        print(f"    Enviando: {info['titulo']} ...", end=" ", flush=True)
        respuesta = _llamar_ia(info["prompt"])
        respuestas[clave] = {"titulo": info["titulo"], "respuesta": respuesta}
        estado = "✓" if not respuesta.startswith("[⚠") else "✗"
        print(estado)

    return respuestas


def imprimir_analisis_ia(respuestas: dict):
    """Imprime el análisis de la IA con formato legible en consola."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  ANÁLISIS DE LA IA LOCAL (Mistral-7B-Instruct via llamafile)")
    print(sep)

    for clave, info in respuestas.items():
        print(f"\n{info['titulo']}")
        print("-" * 55)
        # Envolver líneas largas para mejor lectura en consola
        for linea in info["respuesta"].splitlines():
            wrapped = textwrap.fill(linea, width=72, subsequent_indent="   ")
            print(wrapped if wrapped else "")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────
# 7. VISUALIZACIONES
# ──────────────────────────────────────────────

def graficar_red(G, TF, ES, EF, filename="red_cpm.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f8f8f5")
    fig.patch.set_facecolor("#f8f8f5")

    pos = {
        "A": (0, 0), "B": (2, 0), "C": (4, 1),  "D": (4, 0),
        "E": (4, -1), "F": (6, 1), "G": (8, 0), "H": (10, 0),
    }

    criticos = [n for n in G.nodes if TF[n] == 0]
    normales = [n for n in G.nodes if TF[n] > 0]
    arcos_cr = [(u, v) for u, v in G.edges if TF[u] == 0 and TF[v] == 0]
    arcos_no = [(u, v) for u, v in G.edges if (u, v) not in arcos_cr]

    nx.draw_networkx_nodes(G, pos, nodelist=criticos, node_color="#E24B4A",
                           node_size=1600, ax=ax, alpha=0.85)
    nx.draw_networkx_nodes(G, pos, nodelist=normales, node_color="#1D9E75",
                           node_size=1600, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(G, pos, edgelist=arcos_cr, edge_color="#E24B4A",
                           width=2.5, arrows=True, arrowsize=20, ax=ax,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=arcos_no, edge_color="#888780",
                           width=1.2, arrows=True, arrowsize=18, ax=ax,
                           connectionstyle="arc3,rad=0.05")

    etiquetas = {n: f"{n}\nES={ES[n]} EF={EF[n]}" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=etiquetas, font_size=8,
                            font_color="white", font_weight="bold", ax=ax)

    holguras = {e: f"  TF={TF[e[1]]}" for e in arcos_no}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=holguras,
                                 font_size=7, font_color="#444", ax=ax)

    ax.set_title(
        "Red CPM — Distribución de Recursos\n(rojo = ruta crítica, verde = con holgura)",
        fontsize=12, pad=12,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Red guardada → {filename}")


def graficar_gantt(df, inicio_nivelado, T_star, filename="gantt.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f8f8f5")
    fig.patch.set_facecolor("#f8f8f5")

    yticks, ylabels = [], []
    for i, row in df.iterrows():
        y = len(df) - i - 1
        yticks.append(y)
        ylabels.append(f"{row['Actividad']}  {row['Nombre']}")
        color = "#E24B4A" if row["Crítica"] else "#378ADD"

        ax.barh(y, row["Duración"], left=row["ES"], color=color, alpha=0.85,
                height=0.5, edgecolor="white", linewidth=0.5)

        if row["Holgura (TF)"] > 0:
            ax.barh(y, row["Holgura (TF)"], left=row["EF"], color=color,
                    alpha=0.18, height=0.5, edgecolor=color,
                    linewidth=0.8, linestyle="--")

        if row["Actividad"] in inicio_nivelado:
            s_niv = inicio_nivelado[row["Actividad"]]
            if s_niv != row["ES"]:
                ax.plot(s_niv + row["Duración"] / 2, y, "v",
                        color="#EF9F27", markersize=7, zorder=5)

        ax.text(row["ES"] + row["Duración"] / 2, y, f"{row['Duración']}d",
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Días", fontsize=10)
    ax.set_xlim(0, T_star + 1)
    ax.set_xticks(range(T_star + 1))
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_title(
        "Diagrama de Gantt\n(▼ = inicio nivelado | zona punteada = holgura disponible)",
        fontsize=12, pad=12,
    )

    leyenda = [
        mpatches.Patch(color="#E24B4A", label="Actividad crítica"),
        mpatches.Patch(color="#378ADD", label="Actividad con holgura"),
        mpatches.Patch(color="#EF9F27", label="Desplazamiento nivelado"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gantt guardado  → {filename}")


def graficar_perfil_recursos(perfil_original, perfil_nivelado, T_star,
                              recursos_disp, filename="perfil_recursos.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor("#f8f8f5")
    dias = list(range(T_star))

    for ax, perfil, titulo, color in [
        (axes[0], perfil_original, "Sin nivelación", "#E24B4A"),
        (axes[1], perfil_nivelado,  "Con nivelación", "#1D9E75"),
    ]:
        ax.set_facecolor("#f8f8f5")
        bars = ax.bar(dias, perfil, color=color, alpha=0.75,
                      edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, perfil):
            if val > recursos_disp:
                bar.set_color("#E24B4A")
                bar.set_alpha(0.95)

        ax.axhline(recursos_disp, color="#444", linestyle="--",
                   linewidth=1.2, label=f"Límite: {recursos_disp}")
        ax.axhline(np.mean(perfil), color="#EF9F27", linestyle=":",
                   linewidth=1.2, label=f"Promedio: {np.mean(perfil):.1f}")

        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("Día", fontsize=9)
        ax.set_ylabel("Trabajadores", fontsize=9)
        ax.set_xticks(dias)
        ax.set_xticklabels([str(d) for d in dias], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        varianza = np.var(perfil)
        pico     = max(perfil)
        sobrecar = sum(1 for v in perfil if v > recursos_disp)
        ax.text(
            0.02, 0.96,
            f"Varianza: {varianza:.1f}  |  Pico: {pico}  |  Días sobrecargados: {sobrecar}",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.suptitle("Perfil de Recursos — Antes vs Después de la Nivelación",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Perfil guardado → {filename}")


# ──────────────────────────────────────────────
# 8. REPORTE EN CONSOLA
# ──────────────────────────────────────────────

def imprimir_reporte(df, rutas_criticas, T_star, perfil_original, perfil_nivelado,
                     inicio_nivelado, ES, RECURSOS_DISPONIBLES):
    sep = "=" * 65

    print(f"\n{sep}")
    print("  MODELO CPM — DISTRIBUCIÓN DE RECURSOS EN REDES")
    print(sep)

    print("\n📋 TABLA CPM COMPLETA")
    print(df.to_string(index=False))

    print(f"\n⏱  DURACIÓN MÍNIMA DEL PROYECTO: {T_star} días")

    print("\n RUTA(S) CRÍTICA(S):")
    for i, ruta in enumerate(rutas_criticas, 1):
        print(f"   Ruta {i}: {' → '.join(ruta)}")

    print("\n ACTIVIDADES CON HOLGURA:")
    con_holgura = df[df["Holgura (TF)"] > 0][
        ["Actividad", "Nombre", "Holgura (TF)", "ES", "LS"]
    ]
    print(con_holgura.to_string(index=False))

    print("\n PERFIL DE RECURSOS (trabajadores/día):")
    print(f"   {'Día':<6} {'Sin nivelar':>12} {'Nivelado':>10} {'Límite':>8}")
    for t, (orig, niv) in enumerate(zip(perfil_original, perfil_nivelado)):
        alerta = " ⚠" if orig > RECURSOS_DISPONIBLES else ""
        print(f"   {t:<6} {orig:>12} {niv:>10} {RECURSOS_DISPONIBLES:>8}{alerta}")

    print("\n INICIO NIVELADO vs INICIO TEMPRANO:")
    print(f"   {'Act':<6} {'ES (original)':>14} {'Inicio nivelado':>16} {'Desplazamiento':>15}")
    for act in df["Actividad"]:
        es  = ES[act]
        niv = inicio_nivelado.get(act, es)
        dif = niv - es
        marca = "  ← desplazada" if dif > 0 else ""
        print(f"   {act:<6} {es:>14} {niv:>16} {dif:>15}{marca}")

    var_orig = np.var(perfil_original)
    var_niv  = np.var(perfil_nivelado)
    pico_o   = max(perfil_original)
    pico_n   = max(perfil_nivelado)
    sob_o    = sum(1 for v in perfil_original if v > RECURSOS_DISPONIBLES)
    sob_n    = sum(1 for v in perfil_nivelado  if v > RECURSOS_DISPONIBLES)

    print(f"\n RESUMEN DE NIVELACIÓN:")
    print(f"   Varianza del perfil:       {var_orig:6.2f}  →  {var_niv:6.2f}  ({((var_niv-var_orig)/var_orig*100):+.1f}%)")
    print(f"   Pico de recursos:          {pico_o:6d}   →  {pico_n:6d}")
    print(f"   Días con sobrecarga:       {sob_o:6d}   →  {sob_n:6d}")
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────
# 9. FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────

def main():
    print("\n Ejecutando modelo CPM — Distribución de Recursos...\n")

    # Construir red y calcular CPM
    G = construir_red(ACTIVIDADES)
    df, T_star, ES, EF, LS, LF, TF = calcular_cpm(G, ACTIVIDADES)
    criticos, rutas = obtener_ruta_critica(G, TF)

    # Perfiles de recursos
    perfil_original = calcular_perfil_recursos(ACTIVIDADES, ES, EF, T_star)
    inicio_nivelado = nivelar_recursos(ACTIVIDADES, ES, EF, LS, LF, TF, T_star)
    perfil_nivelado = calcular_perfil_recursos(
        ACTIVIDADES, ES, EF, T_star, inicio_nivelado
    )

    # Reporte numérico en consola
    imprimir_reporte(
        df, rutas, T_star, perfil_original, perfil_nivelado,
        inicio_nivelado, ES, RECURSOS_DISPONIBLES,
    )

    # ── CONSULTA A LA IA LOCAL ──────────────────
    respuestas_ia = analizar_con_ia(
        df, rutas, T_star,
        perfil_original, perfil_nivelado,
        inicio_nivelado, ES, RECURSOS_DISPONIBLES,
    )
    imprimir_analisis_ia(respuestas_ia)
    # ────────────────────────────────────────────

    # Gráficas
    print(" Generando gráficas...")
    graficar_red(G, TF, ES, EF, filename="red_cpm.png")
    graficar_gantt(df, inicio_nivelado, T_star, filename="gantt.png")
    graficar_perfil_recursos(
        perfil_original, perfil_nivelado, T_star,
        RECURSOS_DISPONIBLES, filename="perfil_recursos.png",
    )
    print("\n Modelo ejecutado con éxito.\n")

    return (
        df, T_star, ES, EF, LS, LF, TF,
        perfil_original, perfil_nivelado,
        inicio_nivelado, respuestas_ia,
    )


if __name__ == "__main__":
    main()