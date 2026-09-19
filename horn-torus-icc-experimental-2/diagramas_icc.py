#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagramas del horn torus del Icc — figuras para el Cap. 4/5 (tesis RSI–Poincaré).

Se apoya en el motor del repo horn-torus-icc-experimental:
    from horn_torus_experimental import HornTorusICC

Cada figura declara, en la leyenda del artefacto acompañante, el estatuto
(CITA / LECTURA / AXIOMA / PENDIENTE) de lo que dibuja.

Uso:
    python diagramas_icc.py            # genera las figuras en el directorio actual
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from horn_torus_experimental import HornTorusICC

# --- paleta: las cuatro cintas conservan los colores del visualizador React ---
COL = {
    "S":      "#d81b3c",   # crimson  — S (significante)
    "I":      "#2e8b57",   # seagreen — I (imagen del cuerpo)
    "puls":   "#daa520",   # goldenrod— hilo pulsional (Trieb)
    "Sigma":  "#3f51b5",   # royalblue— Σ (síntoma)
    "voz":    "#111111",   # punto de autotangencia / la voz
    "trauma": "#ef6c00",   # núcleo del trauma
    "fant":   "#d81b8c",   # núcleo fantasmático (magenta del visualizador)
    "ding":   "#37474f",   # la superficie de Ding
    "pared":  "#c9d6de",   # espesor Pcs
    "gris":   "#78909c",
}


def estilo():
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 110, "savefig.dpi": 300,
        "font.family": "DejaVu Sans", "axes.unicode_minus": False,
    })


# ----------------------------------------------------------------------
# geometría del toro sólido V encerrado por el horn torus
# ----------------------------------------------------------------------
def punto(a, u, v, rho=None):
    """Punto del toro sólido V: rho = 0 es el círculo-núcleo, rho = a la superficie."""
    rho = a if rho is None else rho
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    r = a + rho * np.cos(v)
    return r * np.cos(u), r * np.sin(u), rho * np.sin(v)


def distancia_angular(u, v, u0, v0):
    """Angustia A(u,v) con distancia ENVUELTA — corrige la euclídea en (u,v).

    La versión del motor usa sqrt((u-u_F)^2 + (v-v_F)^2), que no es una
    distancia sobre el toro: v = 2pi-eps y v = 0 son el mismo punto y sin
    embargo quedan a 2pi de diferencia.
    """
    du = np.angle(np.exp(1j * (u - u0)))
    dv = np.angle(np.exp(1j * (v - v0)))
    return np.hypot(du, dv)


# ----------------------------------------------------------------------
# Figura 1 — el corte axial (§4)
# ----------------------------------------------------------------------
def espesor_pcs(th, e0=0.155, thin=(0.95, 2.05, 4.45), prof=0.62, ancho=0.30):
    """Espesor de la pared: máximo en el ecuador exterior, CERO en la autotangencia."""
    taper = np.abs(np.cos(th / 2)) ** 0.55
    dip = np.ones_like(th)
    for t0 in thin:
        d = np.angle(np.exp(1j * (th - t0)))
        dip = dip - prof * np.exp(-(d / ancho) ** 2)
    return e0 * taper * dip


def fig_corte_axial(path="fig1_corte_axial.png", a=1.0, pared=False):
    """Corte axial del horn torus del Icc.

    Ding es una superficie SIN espesor: el Pcs no ocupa una region de V, esta en
    otro espacio que este modelo no representa (decision del 17/09). pared=True
    reproduce el dibujo anterior, con la banda de espesor Pcs, sólo para cotejo.
    """
    estilo()
    th = np.linspace(0, 2 * np.pi, 2000)
    fig, ax = plt.subplots(figsize=(8.3, 5.0))

    for sg in (+1, -1):
        xi, yi = sg * a + a * np.cos(th), a * np.sin(th)
        if pared:
            e = espesor_pcs(th) * a
            xo, yo = sg * a + (a + e) * np.cos(th), (a + e) * np.sin(th)
            ax.fill(np.r_[xi, xo[::-1]], np.r_[yi, yo[::-1]], color=COL["pared"], lw=0, zorder=1)
        ax.add_patch(Circle((sg * a, 0), 0.999 * a, fc="white", ec="none", zorder=0))
        ax.plot(xi, yi, color=COL["ding"], lw=2.1, zorder=4)

    ax.plot([0, 0], [-1.5 * a, 1.5 * a], ls=(0, (6, 4)), lw=0.8, color="#b0bec5", zorder=0)
    ax.text(0.06 * a, 1.46 * a, "eje de rotación", color=COL["gris"], fontsize=7,
            va="top", rotation=90)

    # --- las marcas, sobre la cara interna del lóbulo derecho (con su espejo) ---
    rin = 0.93 * a
    marcas = [("S", 1.05, COL["S"], "o", 5.6),
              ("hilo pulsional", 0.66, COL["puls"], "o", 4.2),
              ("I", 0.40, COL["I"], "o", 5.6),
              ("Σ", -0.42, COL["Sigma"], "o", 5.6),
              ("núcleo del trauma", -1.02, COL["trauma"], "D", 5.4),
              ("núcleo fantasmático", -1.55, COL["fant"], "s", 5.4)]
    for nom, t0, col, mk, ms in marcas:
        x, y = a + rin * np.cos(t0), rin * np.sin(t0)
        ax.plot(x, y, mk, ms=ms, mfc=col, mec="white", mew=0.7, zorder=7)
        rl = 1.78 * a if nom == "núcleo fantasmático" else 1.55 * a
        ax.annotate(nom, xy=(x, y), xytext=(a + rl * np.cos(t0), rl * np.sin(t0)),
                    color=col, fontsize=8, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=col, alpha=0.55,
                                    shrinkA=1, shrinkB=2))
        t = np.pi - t0
        ax.plot(-a + rin * np.cos(t), rin * np.sin(t), mk, ms=ms * 0.82, mfc=col,
                mec="white", mew=0.6, alpha=0.75, zorder=7)

    # --- las tres vías variables: puntos de la superficie, no adelgazamientos ---
    for nom, t0 in (("palabra", 2.42), ("agieren", 3.14), ("sublimación", 3.86)):
        x, y = -a + a * np.cos(t0), a * np.sin(t0)
        ax.plot(x, y, "o", ms=5.2, mfc="white", mec="#37474f", mew=1.3, zorder=8)
        ax.annotate(nom, xy=(x, y), xytext=(-a + 1.62 * a * np.cos(t0), 1.62 * a * np.sin(t0)),
                    color="#37474f", fontsize=8, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#607d8b", alpha=0.6,
                                    shrinkA=1, shrinkB=3))

    ax.plot(0, 0, "o", ms=7.5, mfc="white", mec=COL["voz"], mew=1.9, zorder=9)
    ax.annotate("punto de autotangencia\nla voz del superyó — único agujero fijo", xy=(0, 0),
                xytext=(0, -1.52 * a), color=COL["voz"], fontsize=8, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.8, color=COL["voz"], shrinkA=0, shrinkB=4))

    ax.annotate("Ding: superficie sin espesor", xy=(-a + a * np.cos(1.35), a * np.sin(1.35)),
                xytext=(-1.55 * a, 1.52 * a), color=COL["ding"], fontsize=8,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=COL["ding"], shrinkB=3))
    ax.text(-a, 0.04 * a, "V = el Icc\n(interior vacío)", ha="center", va="center",
            fontsize=8.5, color="#90a4ae", linespacing=1.4)
    ax.text(a, -0.62 * a, "zeitlos", ha="center", va="center", fontsize=8, color="#b0bec5")
    ax.text(-3.24 * a, -1.46 * a, "afuera: otro espacio — el Pcs y la Cc no están\n"
                                  "en V y este modelo no los representa",
            ha="left", va="bottom", fontsize=8, color="#607d8b", linespacing=1.4)

    ax.set_title("Corte axial del horn torus del Icc: dos círculos iguales, "
                 "tangentes en un solo punto", loc="left")
    ax.set_xlim(-3.30 * a, 3.05 * a); ax.set_ylim(-1.92 * a, 1.66 * a)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    return fig


def cinta_conforme(M, u, rho_frac=0.93, banda=(0.45, 2.70), m=3.0, ancho_I=0.075):
    """Trenza S-I-Σ + hilo pulsional distribuida sobre la cara interna.

    Conserva las fases del motor (v_S, v_I, v_Sigma mod 2pi) y les suma el
    espaciado canónico de una trenza de tres hebras. La banda excluye v = pi
    (el punto de la voz) y v = 0 (el ecuador exterior).
    """
    a = M.effective_a
    rho = rho_frac * a
    v0, A = 0.5 * (banda[0] + banda[1]), 0.5 * (banda[1] - banda[0])
    fases = [M.v_S % (2 * np.pi), M.v_I % (2 * np.pi), M.v_Sigma % (2 * np.pi)]
    out = {}
    for k, (nom, ph) in enumerate(zip(["S", "I", "Sigma"], fases)):
        v = v0 + A * np.sin(m * u + ph + 2 * np.pi * k / 3)
        out[nom] = dict(zip("xyz", punto(a, u, v, rho)), v=v)
    vI = out["I"]["v"]
    out["I"]["borde"] = (vI - ancho_I, vI + ancho_I)
    # hilo pulsional: pegado al borde exterior de I, nunca lo cruza
    d = ancho_I + 0.055 * M.pulsion_attachment_strength
    vP = vI + d
    out["Pulsion"] = dict(zip("xyz", punto(a, u, vP, rho)), v=vP)
    return out


def v_de_curva(M, c):
    """Recupera la coordenada v de una curva del motor: cos v = r/a - 1, sin v = z/a."""
    a = M.effective_a
    r = np.hypot(c["x"], c["y"])
    return np.mod(np.arctan2(np.asarray(c["z"]) / a, r / a - 1.0), 2 * np.pi)


def fig_cinta(path="fig2_cinta_cara_interna.png", M=None, banda=(0.45, 2.70)):
    """(a) la cara interna vista desde abajo; (b) la carta desplegada (u, v)."""
    estilo()
    M = M or HornTorusICC()
    a = M.effective_a
    u = np.linspace(0, 2 * np.pi, 700)
    cin = cinta_conforme(M, u, banda=banda)
    rho = 0.93 * a

    fig = plt.figure(figsize=(9.8, 4.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], left=0.0, right=0.985,
                          top=0.88, bottom=0.115, wspace=0.02)

    # ---------- (a) 3D: mitad inferior del tubo removida, vista desde abajo ----
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 200), np.linspace(0.0, np.pi, 70))
    X, Y, Z = punto(a, U, V)
    ax.plot_surface(X, Y, Z, color="#c3d0d8", alpha=0.62, linewidth=0,
                    antialiased=True, shade=True, rstride=2, cstride=2)
    xb, yb, zb = punto(a, u, 0.0)
    ax.plot(xb, yb, zb, color=COL["ding"], lw=1.0, alpha=0.75)

    VI = np.vstack(cin["I"]["borde"])
    Xi, Yi, Zi = punto(a, np.vstack([u, u]), VI, rho)
    ax.plot_surface(Xi, Yi, Zi, color=COL["I"], alpha=1.0, linewidth=0, shade=False)
    for nom, lw in (("S", 2.1), ("Sigma", 2.1), ("Pulsion", 1.5)):
        c = cin[nom]
        ax.plot(c["x"], c["y"], c["z"], lw=lw, solid_capstyle="round",
                color=COL["puls"] if nom == "Pulsion" else COL[nom])

    ax.scatter([0], [0], [0], s=58, facecolor="white", edgecolor=COL["voz"],
               linewidth=1.7, depthshade=False)
    for (uu, vv), col, mk in (((M.u_F, M.v_F), COL["fant"], "s"), ((1.95, 2.25), COL["trauma"], "D")):
        xp, yp, zp = punto(a, uu, vv, rho)
        ax.scatter([xp], [yp], [zp], s=44, marker=mk, color=col, edgecolor="white",
                   linewidth=0.7, depthshade=False)

    ax.set_xlim(-2.05 * a, 2.05 * a); ax.set_ylim(-2.05 * a, 2.05 * a)
    ax.set_zlim(-0.15 * a, 1.15 * a)
    ax.set_box_aspect((1, 1, 0.42), zoom=1.45)
    ax.view_init(elev=-34, azim=-58)
    ax.set_axis_off()
    ax.text2D(-0.02, 1.0, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.text2D(0.03, 1.0, "la cara interna, vista desde abajo\n(mitad inferior del tubo removida)",
              transform=ax.transAxes, fontsize=8.3, va="top")

    # ---------- (b) carta desplegada (u, v) --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axhspan(banda[0], banda[1], color="#eceff1", zorder=0)

    # campo de angustia con distancia ENVUELTA, y el umbral A_cr
    Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 400), np.linspace(0, 2 * np.pi, 400))
    Ag = distancia_angular(Ug, Vg, M.u_F, M.v_F)
    ax.contourf(Ug, Vg, Ag, levels=[0, M.A_cr], colors=[COL["fant"]], alpha=0.13, zorder=1)
    ax.contour(Ug, Vg, Ag, levels=[M.A_cr], colors=[COL["fant"]], linewidths=0.9,
               linestyles="--", zorder=2)

    # la cinta conforme al §4
    vlo, vhi = cin["I"]["borde"]
    ax.fill_between(u, vlo, vhi, color=COL["I"], lw=0, zorder=4)
    ax.plot(u, cin["Pulsion"]["v"], color=COL["puls"], lw=1.4, zorder=5)
    ax.plot(u, cin["S"]["v"], color=COL["S"], lw=1.8, zorder=5)
    ax.plot(u, cin["Sigma"]["v"], color=COL["Sigma"], lw=1.8, zorder=5)

    # la cinta tal como la calcula hoy get_curves(): pegada a v = pi
    uc = np.linspace(0, 2 * np.pi, 221)
    for nom, c in M.get_curves().items():
        ax.plot(uc, v_de_curva(M, c), lw=0.9, ls=(0, (3, 2)), alpha=0.85, zorder=3,
                color=COL["puls"] if nom == "Pulsion" else COL[nom])

    ax.axhline(np.pi, color=COL["voz"], lw=1.6, zorder=6)
    ax.plot(0, np.pi, "o", ms=7, mfc="white", mec=COL["voz"], mew=1.7,
            clip_on=False, zorder=7)
    ax.plot([M.u_F], [M.v_F], "s", ms=6, color=COL["fant"], mec="white", mew=0.7, zorder=8)
    ax.plot([1.95], [2.25], "D", ms=6, color=COL["trauma"], mec="white", mew=0.7, zorder=8)

    ax.annotate("v = π: todo el segmento es UN solo punto del toro — la voz",
                xy=(3.05, np.pi), xytext=(0.18, 4.30), color=COL["voz"], fontsize=7.8,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["voz"], shrinkB=3))
    ax.annotate("la cinta de get_curves() oscila alrededor de v = π,\n"
                "así que cruza el agujero de la voz en cada vuelta",
                xy=(1.62, 3.60), xytext=(0.18, 5.62), fontsize=7.6, color="#455a64",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#607d8b", shrinkB=3))
    ax.annotate("A ≤ A_cr = π/4\nzona de ruptura", xy=(M.u_F + 0.74, M.v_F + 0.62),
                xytext=(4.66, 5.10), fontsize=7.6, color=COL["fant"], ha="left",
                va="center", arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["fant"],
                                             shrinkB=2))

    tk = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    tl = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(tk); ax.set_xticklabels(tl); ax.set_yticks(tk); ax.set_yticklabels(tl)
    ax.set_xlabel("u  (vuelta alrededor del eje)")
    ax.set_ylabel("v  (vuelta alrededor del tubo)")
    ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(-0.115, 1.045, "b", transform=ax.transAxes, fontweight="bold", fontsize=11,
            ha="left", va="center")
    ax.set_title("la carta desplegada de la cara interna", loc="left", fontsize=8.3, pad=7)

    h = [plt.Line2D([], [], color=COL["S"], lw=2.0, label="S (significante)"),
         plt.Line2D([], [], color=COL["I"], lw=5.0, label="I (imagen del cuerpo), banda con borde"),
         plt.Line2D([], [], color=COL["puls"], lw=1.6, label="hilo pulsional, pegado al borde de I"),
         plt.Line2D([], [], color=COL["Sigma"], lw=2.0, label="Σ (síntoma)"),
         plt.Line2D([], [], color="#607d8b", lw=0.9, ls=(0, (3, 2)), label="las mismas, según get_curves()"),
         plt.Line2D([], [], ls="none", marker="o", mfc="white", mec=COL["voz"], mew=1.5,
                    ms=7, label="autotangencia = la voz"),
         plt.Line2D([], [], ls="none", marker="s", color=COL["fant"], ms=6,
                    label="núcleo fantasmático (u_F, v_F)"),
         plt.Line2D([], [], ls="none", marker="D", color=COL["trauma"], ms=6,
                    label="núcleo del trauma (posición no fijada)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.055), fontsize=7.5, handlelength=1.9,
               columnspacing=1.5)
    fig.text(0.008, 0.985, "La cinta S-I-Σ y el hilo pulsional sobre la cara interna del horn torus",
             fontsize=9.5, va="top")
    fig.savefig(path, bbox_inches="tight")
    return fig


def _dibuja_toro(ax, a, u_ini, u_fin, nu=170, nv=90, alpha=0.30):
    u = np.linspace(u_ini, u_fin, nu)
    v = np.linspace(0, 2 * np.pi, nv)
    U, V = np.meshgrid(u, v)
    X, Y, Z = punto(a, U, V)
    ax.plot_surface(X, Y, Z, color="#b8c7d1", alpha=alpha, linewidth=0,
                    antialiased=True, shade=True, rstride=2, cstride=2)
    for uc in (u_ini, u_fin):                      # caras del corte de cuña
        rr = np.linspace(0, a, 18)
        RR, VV = np.meshgrid(rr, v)
        Xc, Yc, Zc = punto(a, uc, VV, RR)
        ax.plot_surface(Xc, Yc, Zc, color="#dce6ec", alpha=0.55, linewidth=0, shade=False)
        xb, yb, zb = punto(a, uc, v)
        ax.plot(xb, yb, zb, color=COL["ding"], lw=1.0, alpha=0.8)


def _camara(ax, a):
    ax.set_xlim(-2.1 * a, 2.1 * a); ax.set_ylim(-2.1 * a, 2.1 * a)
    ax.set_zlim(-1.25 * a, 1.25 * a)
    ax.set_box_aspect((1, 1, 0.60))
    ax.view_init(elev=24, azim=0)
    ax.set_axis_off()


def fig_cinta_3d(path="fig2_cinta_cara_interna.png", M=None, hueco_deg=52):
    estilo()
    M = M or HornTorusICC()
    a = M.effective_a
    g = np.deg2rad(hueco_deg)
    u_ini, u_fin = g, 2 * np.pi - g
    u = np.linspace(u_ini, u_fin, 700)

    fig = plt.figure(figsize=(9.6, 4.6))

    # ---- (a) tal como está hoy en get_curves(): la cinta colapsa en el cuello
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _dibuja_toro(ax, a, u_ini, u_fin)
    for nom, c in M.get_curves().items():
        col = COL["puls"] if nom == "Pulsion" else COL[nom]
        uu = np.arctan2(c["y"], c["x"]) % (2 * np.pi)
        sel = (uu > u_ini) & (uu < u_fin)
        ax.plot(np.asarray(c["x"])[sel], np.asarray(c["y"])[sel], np.asarray(c["z"])[sel],
                color=col, lw=2.0, solid_capstyle="round")
    ax.scatter([0], [0], [0], s=55, facecolor="white", edgecolor=COL["voz"],
               linewidth=1.6, depthshade=False, zorder=10)
    _camara(ax, a)
    ax.text2D(0.0, 0.97, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.text2D(0.06, 0.96, "tal como la calcula get_curves(): la cinta colapsa en el cuello",
              transform=ax.transAxes, fontsize=8.5, va="top")
    ax.text2D(0.02, 0.10, "las cuatro hebras oscilan alrededor de v = π,\n"
                          "así que ocupan r ≤ 0.45 de un radio exterior 4.25\n"
                          "y atraviesan el punto de la voz (r = 0)",
              transform=ax.transAxes, fontsize=7.5, color="#455a64")

    # ---- (b) conforme al §4: distribuida sobre la cara interna
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    _dibuja_toro(ax, a, u_ini, u_fin)
    cin = cinta_conforme(M, u)
    vlo, vhi = cin["I"]["borde"]                    # I como banda: tiene borde
    for vv, w in ((vlo, 1.0), (vhi, 1.0)):
        pass
    VI = np.vstack([vlo, vhi])
    UU = np.vstack([u, u])
    Xi, Yi, Zi = punto(a, UU, VI, 0.93 * a)
    ax.plot_surface(Xi, Yi, Zi, color=COL["I"], alpha=0.95, linewidth=0, shade=False)
    for nom, lw in (("S", 2.2), ("Sigma", 2.2), ("Pulsion", 1.6)):
        c = cin[nom]
        col = COL["puls"] if nom == "Pulsion" else COL[nom]
        ax.plot(c["x"], c["y"], c["z"], color=col, lw=lw, solid_capstyle="round")

    ax.scatter([0], [0], [0], s=55, facecolor="white", edgecolor=COL["voz"],
               linewidth=1.6, depthshade=False, zorder=10)
    xf, yf, zf = punto(a, M.u_F, M.v_F, 0.93 * a)
    ax.scatter([xf], [yf], [zf], s=42, marker="s", color=COL["fant"],
               edgecolor="white", linewidth=0.7, depthshade=False)
    u_T, v_T = 1.95, 2.25
    xt, yt, zt = punto(a, u_T, v_T, 0.93 * a)
    ax.scatter([xt], [yt], [zt], s=42, marker="D", color=COL["trauma"],
               edgecolor="white", linewidth=0.7, depthshade=False)
    _camara(ax, a)
    ax.text2D(0.0, 0.97, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)
    ax.text2D(0.06, 0.96, "conforme al §4: distribuida sobre la cara interna",
              transform=ax.transAxes, fontsize=8.5, va="top")
    ax.text2D(0.02, 0.10, "misma fase (v_S, v_I, v_Σ) + espaciado de trenza;\n"
                          "la banda excluye v = π, así que ninguna hebra\n"
                          "pasa por el agujero de la voz",
              transform=ax.transAxes, fontsize=7.5, color="#455a64")

    # ---- leyenda única, en la franja inferior
    h = [plt.Line2D([], [], color=COL["S"], lw=2.4, label="S (significante)"),
         plt.Line2D([], [], color=COL["I"], lw=5.0, label="I (imagen del cuerpo) — banda con borde"),
         plt.Line2D([], [], color=COL["puls"], lw=1.8, label="hilo pulsional, pegado al borde de I"),
         plt.Line2D([], [], color=COL["Sigma"], lw=2.4, label="Σ (síntoma)"),
         plt.Line2D([], [], ls="none", marker="o", mfc="white", mec=COL["voz"], mew=1.5,
                    ms=7, label="autotangencia = la voz"),
         plt.Line2D([], [], ls="none", marker="s", color=COL["fant"], ms=6,
                    label="núcleo fantasmático (u_F, v_F)"),
         plt.Line2D([], [], ls="none", marker="D", color=COL["trauma"], ms=6,
                    label="núcleo del trauma (posición no fijada)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.6, handlelength=1.8,
               columnspacing=1.4)
    fig.text(0.01, 0.985, "La cinta S-I-Σ y el hilo pulsional sobre la cara interna del horn torus",
             fontsize=9.5, va="top")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.14, wspace=0.0)
    fig.savefig(path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figura 3 — las cuatro vías de salida (§5) y el tiempo en el cruce (§9)
# ----------------------------------------------------------------------
def fig_vias_salida(path="fig3_vias_de_salida.png", a=1.0):
    """Las cuatro vías de salida, sin pared: la condición de pasaje está en la marca."""
    estilo()
    th = np.linspace(0, 2 * np.pi, 1600)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    ax.add_patch(Circle((a, 0), 0.999 * a, fc="#fbfcfc", ec="none", zorder=0))
    ax.plot(a + a * np.cos(th), a * np.sin(th), color=COL["ding"], lw=2.1, zorder=4)
    ax.plot([0, 0], [-1.42 * a, 1.42 * a], ls=(0, (6, 4)), lw=0.8, color="#cfd8dc", zorder=0)

    vias = [(0.95, "palabra — deformada", COL["S"],
             "sueño, lapsus, acto fallido (Entstellung);\nexige marca ligada a una Wortvorstellung"),
            (2.05, "agieren", COL["Sigma"],
             "cuando la vía representacional está bloqueada:\nse actúa en vez de recordarse"),
            (4.55, "sublimación", COL["I"],
             "el fin se desplaza sin distorsión:\nno hay nada que disfrazar al cruzar")]
    for t0, nom, col, det in vias:
        x, y = a + a * np.cos(t0), a * np.sin(t0)
        nx, ny = np.cos(t0), np.sin(t0)
        ax.plot(x, y, "o", ms=6.0, mfc="white", mec=col, mew=1.6, zorder=8)
        ax.annotate("", xy=(x + 0.52 * a * nx, y + 0.52 * a * ny), xytext=(x - 0.20 * a * nx, y - 0.20 * a * ny),
                    zorder=6, arrowprops=dict(arrowstyle="-|>", lw=1.7, color=col,
                                              connectionstyle="arc3,rad=0.22"))
        ha = "left" if nx > -0.2 else "right"
        ax.text(x + 0.66 * a * nx, y + 0.66 * a * ny, nom, color=col, fontsize=8.6,
                ha=ha, va="center", zorder=9)
        ax.text(x + 0.66 * a * nx, y + 0.66 * a * ny - 0.19 * a, det, color="#546e7a",
                fontsize=7.4, ha=ha, va="top", linespacing=1.35, zorder=9)

    ax.plot(0, 0, "o", ms=8.0, mfc="white", mec=COL["voz"], mew=2.0, zorder=9)
    ax.annotate("", xy=(-0.60 * a, -0.16 * a), xytext=(0.04 * a, 0.0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=COL["voz"],
                                connectionstyle="arc3,rad=0.22"))
    ax.text(-0.70 * a, -0.20 * a, "la voz del superyó — fija", color=COL["voz"], fontsize=8.6,
            ha="right", va="center")
    ax.text(-0.70 * a, -0.39 * a, "el punto singular de la superficie;\n"
                                  "no depende del estado económico", color="#546e7a",
            fontsize=7.4, ha="right", va="top", linespacing=1.35)

    ax.text(a, 0.30 * a, "Icc", ha="center", va="center", fontsize=11, color="#455a64")
    ax.text(a, 0.06 * a, "zeitlos: las marcas no se fechan", ha="center", va="center",
            fontsize=8, color="#90a4ae")
    ax.text(a, -0.26 * a, "V es todo Icc: el Pcs no está acá", ha="center", va="center",
            fontsize=8, color="#90a4ae")

    ax.text(-2.35 * a, 1.66 * a, "cada cruce es un borde de época mínimo:\n"
                                 "ahí se produce la fecha, no adentro",
            ha="left", va="top", fontsize=8, color="#37474f", linespacing=1.4)
    ax.text(-2.35 * a, -1.34 * a,
            "las tres vías variables no son adelgazamientos de una pared:\n"
            "Ding no tiene espesor. La condición de pasaje es una propiedad\n"
            "de la MARCA, no del borde — el borde tiene una sola propiedad\n"
            "estructural, su punto singular.",
            ha="left", va="top", fontsize=8, color="#37474f", linespacing=1.45)

    ax.set_title("Las cuatro vías de salida del Icc: una fija, tres variables", loc="left")
    ax.set_xlim(-2.42 * a, 3.05 * a); ax.set_ylim(-2.26 * a, 1.74 * a)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    return fig


def _horn_par(ax, a=1.0, cx=0.0, cy=0.0, col=None, lw=1.6, pared=True):
    th = np.linspace(0, 2 * np.pi, 800)
    for sg in (+1, -1):
        x, y = cx + sg * a + a * np.cos(th), cy + a * np.sin(th)
        if pared:
            ax.add_patch(Circle((cx + sg * a, cy), a, fc="#f2f5f7", ec="none", zorder=1))
        ax.plot(x, y, color=col or COL["ding"], lw=lw, zorder=3)
    ax.plot(cx, cy, "o", ms=5.5, mfc="white", mec=COL["voz"], mew=1.4, zorder=5)


def _ring_torus(ax, cx, cy, R=1.0, r=0.34, k=0.40, col="#455a64",
                meridiano=None, longitud=None, fc="#eef2f4", lado=+1):
    t = np.linspace(0, 2 * np.pi, 400)
    for rad in (R + r, R - r):
        ax.plot(cx + rad * np.cos(t), cy + rad * k * np.sin(t), color=col, lw=1.4, zorder=3)
    ax.fill(cx + (R + r) * np.cos(t), cy + (R + r) * k * np.sin(t), color=fc, lw=0, zorder=1)
    ax.fill(cx + (R - r) * np.cos(t), cy + (R - r) * k * np.sin(t), color="white", lw=0, zorder=2)
    if meridiano:
        ax.plot(cx + lado * R + r * 0.42 * np.cos(t), cy + r * np.sin(t),
                color=meridiano, lw=1.7, zorder=4)
        ax.text(cx + lado * (R + 0.20), cy + 0.02, "m", color=meridiano, fontsize=9,
                ha="center", va="center", style="italic")
    if longitud:
        ax.plot(cx + R * np.cos(t), cy + R * k * np.sin(t), color=longitud, lw=1.5,
                ls=(0, (4, 2.5)), zorder=4)
        ax.text(cx, cy + R * k + 0.07, "ℓ", color=longitud, fontsize=10,
                ha="center", va="bottom", style="italic")


def fig_heegaard(path="fig4_heegaard_no_anudamiento.png"):
    estilo()
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.2))
    C_M, C_L = "#00838f", "#ad1457"          # meridiano / longitud

    # (1) V: el toro sólido cerrado por Ding
    ax = axs[0, 0]
    _horn_par(ax, a=1.0)
    for nom, (dx, dy), col in (("S", (1.0, 0.42), COL["S"]), ("I", (1.0, -0.02), COL["I"]),
                               ("Σ", (1.0, -0.46), COL["Sigma"])):
        ax.add_patch(Circle((dx, dy), 0.085, fc=col, ec="white", lw=0.8, zorder=4))
        ax.text(dx + 0.16, dy, nom, color=col, fontsize=8.5, va="center")
    ax.text(-1.0, 0.0, "interior\nde V", ha="center", va="center", fontsize=7.6, color="#b0bec5")
    ax.set_title("1 · V: el toro sólido que cierra Ding", loc="left", fontsize=8.6)

    # (2) M = V \ N(S u I u Sigma)
    ax = axs[0, 1]
    _horn_par(ax, a=1.0)
    for (dx, dy), col in (((1.0, 0.42), COL["S"]), ((1.0, -0.02), COL["I"]),
                          ((1.0, -0.46), COL["Sigma"])):
        ax.add_patch(Circle((dx, dy), 0.16, fc="white", ec=col, lw=1.2,
                            ls=(0, (3, 2)), zorder=4))
    ax.text(0.0, -1.30, "∂M = Ding (con su punto singular)\n∪ una o dos componentes toroidales",
            ha="center", va="top", fontsize=7.6, color="#37474f", linespacing=1.4)
    ax.set_title("2 · M = V ∖ N(S ∪ I ∪ Σ)", loc="left", fontsize=8.6)

    # (3) duplicar: Heegaard de género 1, no la identidad
    ax = axs[1, 0]
    _ring_torus(ax, -0.95, 0.30, R=0.62, r=0.21, meridiano=C_M, longitud=C_L, lado=-1)
    _ring_torus(ax, 0.95, 0.30, R=0.62, r=0.21, meridiano=C_L, longitud=C_M)
    ax.annotate("", xy=(0.26, 0.30), xytext=(-0.26, 0.30),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.2, color="#37474f"))
    ax.text(0.0, 0.62, "m ↦ ℓ\nℓ ↦ m", ha="center", va="bottom", fontsize=8,
            color="#37474f", linespacing=1.3)
    ax.text(0.0, -0.18, "pegado de Heegaard de género 1  →  S³", ha="center", va="top",
            fontsize=8.4, color="#1b5e20")
    ax.text(0.0, -0.44, "pegado por la identidad  →  S¹ × S²,  no S³", ha="center", va="top",
            fontsize=8.0, color="#b71c1c")
    ax.text(0.0, -0.66, "(Heegaard 1898; Rolfsen, Knots and Links; Hempel, 3-Manifolds)",
            ha="center", va="top", fontsize=6.9, color=COL["gris"], style="italic")
    ax.set_title("3 · duplicar el toro sólido", loc="left", fontsize=8.6)

    # (4) la hipótesis que el toro exige y la esfera no exigía
    ax = axs[1, 1]
    _ring_torus(ax, -0.82, 0.34, R=0.50, r=0.17)
    ax.text(-0.82, -0.02, "Ding sin anudar", ha="center", va="top", fontsize=8,
            color="#1b5e20")
    ax.text(-0.82, -0.22, "los dos lados son\ntoros sólidos", ha="center", va="top",
            fontsize=7.4, color="#546e7a", linespacing=1.35)
    t = np.linspace(0, 2 * np.pi, 600)
    xt = 0.145 * (np.sin(t) + 2 * np.sin(2 * t)) + 0.86
    yt = 0.145 * (np.cos(t) - 2 * np.cos(2 * t)) + 0.50
    ax.plot(xt, yt, color="#b71c1c", lw=3.4, solid_capstyle="round", alpha=0.85)
    ax.text(0.86, -0.02, "Ding anudado — también tame", ha="center", va="top",
            fontsize=8, color="#b71c1c")
    ax.text(0.86, -0.22, "un lado es el complemento\ndel nudo, no un toro sólido\n"
                         "(hiperbólico — Thurston)", ha="center", va="top",
            fontsize=7.4, color="#546e7a", linespacing=1.35)
    ax.text(0.0, -0.80, "la tameness basta para la esfera (Alexander 1924); para un toro, "
                        "NO:\nque Ding esté sin anudar se axiomatiza aparte",
            ha="center", va="top", fontsize=7.8, color="#37474f", linespacing=1.4)
    ax.set_title("4 · por qué el toro exige una hipótesis más", loc="left", fontsize=8.6)

    for ax in axs.ravel():
        ax.set_aspect("equal"); ax.axis("off")
    axs[0, 0].set_xlim(-2.2, 2.0); axs[0, 0].set_ylim(-1.5, 1.2)
    axs[0, 1].set_xlim(-2.2, 2.0); axs[0, 1].set_ylim(-1.9, 1.2)
    axs[1, 0].set_xlim(-1.85, 1.85); axs[1, 0].set_ylim(-1.05, 1.00)
    axs[1, 1].set_xlim(-1.62, 1.62); axs[1, 1].set_ylim(-1.05, 1.00)
    fig.text(0.008, 0.985, "El §4.7 cerrado: del toro sólido a S³ por pegado de Heegaard",
             fontsize=9.5, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, hspace=0.18, wspace=0.05)
    fig.savefig(path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Versión interactiva (Plotly): rotar, y cortar el tubo para ver la cara interna
# ----------------------------------------------------------------------
def html_interactivo(path="horn_torus_icc_interactivo.html", M=None,
                     banda=(0.45, 2.70), rho_frac=0.93):
    import plotly.graph_objects as go

    M = M or HornTorusICC()
    a = M.effective_a
    rho = rho_frac * a
    u = np.linspace(0, 2 * np.pi, 400)
    cin = cinta_conforme(M, u, rho_frac=rho_frac, banda=banda)

    GRIS = [[0, "#cfd8dc"], [1, "#cfd8dc"]]
    tr, vis = [], {}

    def superficie(v_ini, v_fin, modo):
        U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 150),
                           np.linspace(v_ini, v_fin, 84))
        X, Y, Z = punto(a, U, V)
        if modo == "angustia":
            C, cs, sc = distancia_angular(U, V, M.u_F, M.v_F), "Viridis", True
            cb = dict(title="A(u,v) envuelta", len=0.5, thickness=14, x=0.98)
        else:
            C, cs, sc, cb = np.zeros_like(U), GRIS, False, None
        return go.Surface(x=X, y=Y, z=Z, surfacecolor=C, colorscale=cs, showscale=sc,
                          colorbar=cb, opacity=0.60 if modo == "gris" else 0.90,
                          showlegend=False, hoverinfo="skip")

    for k, (vi, vf) in enumerate([(0.0, 2 * np.pi), (0.0, np.pi)]):
        for modo in ("gris", "angustia"):
            vis[(k, modo)] = len(tr)
            tr.append(superficie(vi, vf, modo))

    etiquetas = {"S": "S (significante)", "I": "I (imagen del cuerpo)",
                 "Pulsion": "hilo pulsional (Trieb)", "Sigma": "\u03a3 (s\u00edntoma)"}
    for nom in ("S", "I", "Pulsion", "Sigma"):
        c = cin[nom]
        tr.append(go.Scatter3d(x=c["x"], y=c["y"], z=c["z"], mode="lines",
                               line=dict(color=COL["puls"] if nom == "Pulsion" else COL[nom],
                                         width=7 if nom != "Pulsion" else 4),
                               name=etiquetas[nom],
                               hovertemplate=etiquetas[nom] + "<extra></extra>"))

    for i, (nom, c) in enumerate(M.get_curves().items()):
        tr.append(go.Scatter3d(x=c["x"], y=c["y"], z=c["z"], mode="lines",
                               line=dict(color=COL["puls"] if nom == "Pulsion" else COL[nom],
                                         width=2, dash="dash"),
                               legendgroup="motor", showlegend=(i == 0),
                               name="las mismas, seg\u00fan get_curves()",
                               hovertemplate="get_curves(): " + nom + "<extra></extra>"))

    tr.append(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                           marker=dict(size=7, color="white",
                                       line=dict(color=COL["voz"], width=3)),
                           name="autotangencia = la voz",
                           hovertemplate="la voz \u2014 \u00fanico agujero fijo<extra></extra>"))
    for (uu, vv), col, nom, mk in (((M.u_F, M.v_F), COL["fant"], "n\u00facleo fantasm\u00e1tico", "square"),
                                   ((1.95, 2.25), COL["trauma"],
                                    "n\u00facleo del trauma (posici\u00f3n no fijada)", "diamond")):
        xp, yp, zp = punto(a, uu, vv, rho)
        tr.append(go.Scatter3d(x=[xp], y=[yp], z=[zp], mode="markers",
                               marker=dict(size=6, color=col, symbol=mk,
                                           line=dict(color="white", width=1)),
                               name=nom,
                               hovertemplate=nom + "<br>(u,v) = (%.3f, %.3f)<extra></extra>" % (uu, vv)))

    n = len(tr)

    def visibles(k, modo):
        m = [True] * n
        for key, idx in vis.items():
            m[idx] = (key == (k, modo))
        return m

    botones = [dict(label=lab, method="update", args=[{"visible": visibles(k, modo)}])
               for lab, k, modo in [("mitad inferior removida", 1, "gris"),
                                    ("abierta \u00b7 coloreada por angustia", 1, "angustia"),
                                    ("superficie completa", 0, "gris"),
                                    ("completa \u00b7 coloreada por angustia", 0, "angustia")]]

    fig = go.Figure(data=tr)
    for key, idx in vis.items():
        fig.data[idx].visible = (key == (1, "gris"))

    sub = ("a = %.4f (a_scale\u00b7GSI) \u00b7 escala visual \u00d7%g \u00b7 A_cr = \u03c0/4"
           " \u00b7 la cinta punteada es get_curves()" % (M.a, M.VISUAL_SCALE))
    fig.update_layout(
        title=dict(text="Horn torus del Icc \u2014 la cinta S-I-\u03a3 sobre la cara interna"
                        "<br><sub>" + sub + "</sub>", x=0.02),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=-1.05))),
        updatemenus=[dict(type="buttons", direction="down", x=0.0, y=0.92,
                          xanchor="left", buttons=botones, showactive=True,
                          bgcolor="#eceff1", bordercolor="#b0bec5", font=dict(size=11))],
        legend=dict(x=0.0, y=0.32, bgcolor="rgba(255,255,255,0.78)", font=dict(size=11)),
        margin=dict(l=0, r=0, t=76, b=0), paper_bgcolor="white")

    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ----------------------------------------------------------------------
# Figura 5 — el horn torus como límite de la familia r -> R (§16 reescrito)
# ----------------------------------------------------------------------
def fig_familia_limite(path="fig5_familia_limite.png", R=1.0):
    estilo()
    fig = plt.figure(figsize=(9.9, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.92], left=0.045, right=0.985,
                          top=0.80, bottom=0.14, wspace=0.30)
    th = np.linspace(0, 2 * np.pi, 800)
    radios = [0.45, 0.70, 0.90, 1.00]
    tonos = ["#cfd8dc", "#90a4ae", "#546e7a", COL["voz"]]

    # ---- (a) la familia en corte axial
    ax = fig.add_subplot(gs[0, 0])
    for r, c in zip(radios, tonos):
        for sg in (+1, -1):
            ax.plot(sg * R + r * np.cos(th), r * np.sin(th),
                    color=c if r < 1 else COL["voz"],
                    lw=2.0 if r == 1.0 else 1.3, zorder=3 if r == 1.0 else 2)
    ax.plot(0, 0, "o", ms=7, mfc="white", mec=COL["voz"], mew=1.7, zorder=6)
    ax.text(0.0, 1.30, "el agujero interior, de radio R − r,\nse cierra sobre un solo punto",
            fontsize=7.6, color="#37474f", ha="center", va="bottom", linespacing=1.35)
    prox = [plt.Line2D([], [], color=c if r < 1 else COL["voz"],
                       lw=2.0 if r == 1 else 1.3,
                       label=f"r/R = {r:.2f}" + ("   (horn torus)" if r == 1 else ""))
            for r, c in zip(radios, tonos)]
    ax.legend(handles=prox, loc="lower left", frameon=False, fontsize=7.2,
              handlelength=1.6, labelspacing=0.32, borderaxespad=0.0,
              bbox_to_anchor=(-0.02, -0.03))
    ax.set_xlim(-2.25, 2.25); ax.set_ylim(-1.55, 1.80)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(-0.02, 1.0, "a", transform=ax.transAxes, fontweight="bold", fontsize=11,
            ha="right", va="bottom")
    ax.set_title("la familia: r/R = 0.45 … 1", loc="left", fontsize=8.6)

    # ---- (b) la carta: qué curva se colapsa
    ax = fig.add_subplot(gs[0, 1])
    ax.add_patch(plt.Rectangle((0, 0), 2 * np.pi, 2 * np.pi, fc="#f4f6f7", ec="#b0bec5", lw=0.8))
    ax.axhline(np.pi, color=COL["voz"], lw=2.0, zorder=4)
    ax.annotate("", xy=(2 * np.pi - 0.35, 5.05), xytext=(0.35, 5.05), zorder=5,
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#ad1457"))
    ax.text(np.pi, 5.25, "λ  longitud", color="#ad1457", fontsize=8, ha="center", va="bottom")
    ax.annotate("", xy=(1.55, 2 * np.pi - 0.35), xytext=(1.55, 0.35), zorder=5,
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#00838f"))
    ax.text(1.72, 4.45, "μ  meridiano", color="#00838f", fontsize=8, ha="left", va="center")
    ax.annotate("λ_int = {v = π}\nse colapsa a un punto",
                xy=(4.55, np.pi), xytext=(4.9, 1.55), fontsize=7.8, color=COL["voz"],
                ha="center", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=COL["voz"], shrinkB=3))
    tk = [0, np.pi, 2 * np.pi]; tl = ["0", "π", "2π"]
    ax.set_xticks(tk); ax.set_xticklabels(tl); ax.set_yticks(tk); ax.set_yticklabels(tl)
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax.set_xlim(-0.15, 2 * np.pi + 0.15); ax.set_ylim(-0.15, 2 * np.pi + 0.15)
    ax.set_aspect("equal")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.text(-0.16, 1.0, "b", transform=ax.transAxes, fontweight="bold", fontsize=11,
            ha="right", va="bottom")
    ax.set_title("la curva que muere es una longitud", loc="left", fontsize=8.6)

    # ---- (c) los invariantes: constantes en la familia, saltan en el límite
    ax = fig.add_subplot(gs[0, 2])
    xs = np.linspace(0.30, 1.0, 200)
    ax.plot(xs[xs < 1], np.full((xs < 1).sum(), 2.0), color="#ad1457", lw=2.0)
    ax.plot(xs[xs < 1], np.zeros((xs < 1).sum()), color="#00838f", lw=2.0)
    ax.plot([1.0], [2.0], "o", ms=6, mfc="white", mec="#ad1457", mew=1.5, clip_on=False)
    ax.plot([1.0], [0.0], "o", ms=6, mfc="white", mec="#00838f", mew=1.5, clip_on=False)
    ax.plot([1.0], [1.0], "o", ms=6.5, color="#ad1457", clip_on=False, zorder=5)
    ax.plot([1.0], [1.0], "o", ms=3.0, color="#00838f", clip_on=False, zorder=6)
    ax.text(0.33, 2.16, "rango de H₁", color="#ad1457", fontsize=8, va="bottom")
    ax.text(0.33, 0.14, "χ", color="#00838f", fontsize=8.5, va="bottom")
    ax.annotate("r = R:  H₁ = Z⟨μ⟩,  χ = 1\nya no es una variedad",
                xy=(1.0, 1.0), xytext=(0.93, 1.62), fontsize=7.6, color="#37474f",
                ha="right", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#607d8b", shrinkB=6))
    ax.text(0.33, 0.62, "r < R: toro liso,\nencajado estándar\nH₁ = Z²⟨μ,λ⟩,  χ = 0",
            fontsize=7.6, color="#546e7a", va="bottom", linespacing=1.35)
    ax.set_xlim(0.30, 1.0); ax.set_ylim(-0.18, 2.45)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 1, 2])
    ax.set_xlabel("r / R")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(-0.22, 1.0, "c", transform=ax.transAxes, fontweight="bold", fontsize=11,
            ha="right", va="bottom")
    ax.set_title("los invariantes saltan sólo en el límite", loc="left", fontsize=8.6)

    fig.text(0.008, 0.975, "El horn torus como límite de la familia r → R: los teoremas "
                           "valen en toda la familia; la voz aparece en el límite",
             fontsize=9.5, va="top")
    fig.savefig(path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figura 6 — los invariantes a lo largo de la familia
# ----------------------------------------------------------------------
def fig_invariantes_familia(path="fig6_invariantes_familia.png", R=2.125):
    """W y las curvaturas sobre la familia. W es adimensional; K usa la escala R."""
    estilo()
    x = np.linspace(0.30, 0.9992, 900)
    W = np.pi ** 2 / (x * np.sqrt(1 - x ** 2))
    Kmin = 1.0 / (R ** 2 * x * (1 - x))
    Kmax = 1.0 / (R ** 2 * x * (1 + x))
    Hm = 1.0 / (2 * x * R)
    xc = 1 / np.sqrt(2)

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axs[0]
    ax.plot(x, W, color="#ad1457", lw=2.0)
    ax.axvline(1.0, color=COL["voz"], lw=1.2, ls=(0, (4, 3)))
    ax.plot([xc], [2 * np.pi ** 2], "o", ms=6, color="#ad1457", zorder=5)
    ax.annotate("mínimo de la familia:  W = 2π² = 19.74\nen r/R = 1/√2  (toro de Clifford)\n"
                "— es la línea de base que usa el motor",
                xy=(xc, 2 * np.pi ** 2), xytext=(0.33, 150), fontsize=7.6, color="#546e7a",
                va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#ad1457", shrinkB=4))
    ax.annotate("r = R:  W → ∞", xy=(1.0, 800), xytext=(0.895, 1400), fontsize=8,
                color=COL["voz"], ha="right", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=COL["voz"], shrinkB=3))
    ax.set_yscale("log"); ax.set_ylim(12, 4000); ax.set_xlim(0.30, 1.035)
    ax.set_xlabel("r / R"); ax.set_ylabel("W = ∫H² dA")
    ax.set_title("la energía de flexión diverge en el límite", loc="left", fontsize=8.8)

    ax = axs[1]
    ax.plot(x, Kmin, color="#b71c1c", lw=2.0)
    ax.plot(x, Kmax, color="#00838f", lw=2.0)
    ax.plot(x, Hm, color="#546e7a", lw=1.6, ls=(0, (5, 2)))
    ax.axvline(1.0, color=COL["voz"], lw=1.2, ls=(0, (4, 3)))
    for y, txt, col, dy in ((Kmin[-1], "|K_min| = 1/(r(R−r))  → ∞", "#b71c1c", 1.0),
                            (Kmax[-1], "K_max = 1/(r(R+r))", "#00838f", 0.72),
                            (Hm[-1], "⟨H⟩ = 1/(2r)", "#546e7a", 1.45)):
        ax.text(0.985, y * dy, txt, color=col, fontsize=7.8, ha="right", va="bottom")
    ax.set_yscale("log"); ax.set_xlim(0.30, 1.035)
    ax.set_xlabel("r / R"); ax.set_ylabel("curvatura  (escala R = %.3f)" % R)
    ax.set_title("sólo la curvatura mínima se va al infinito", loc="left", fontsize=8.8)

    for L, ax in zip("ab", axs):
        ax.text(-0.14, 1.03, L, transform=ax.transAxes, fontweight="bold", fontsize=11,
                ha="left", va="bottom")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.008, 0.985, "Qué le pasa a la geometría cuando r → R: la voz aparece donde "
                           "la flexión y la curvatura mínima divergen", fontsize=9.5, va="top")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.145, wspace=0.30)
    fig.savefig(path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Interactivo con deslizador sobre la familia r/R
# ----------------------------------------------------------------------
def html_familia(path="familia_horn_torus_interactivo.html",
                 pasos=(0.45, 0.60, 0.7071, 0.85, 0.95, 1.0), banda=(0.45, 2.70),
                 scl90r=None):
    import plotly.graph_objects as go
    from familia_horn_torus import HornTorusFamilia

    GRIS = [[0, "#cfd8dc"], [1, "#cfd8dc"]]
    u = np.linspace(0, 2 * np.pi, 360)
    v0, A = 0.5 * (banda[0] + banda[1]), 0.5 * (banda[1] - banda[0])
    etiquetas = {"S": "S (significante)", "I": "I (imagen del cuerpo)",
                 "Pulsion": "hilo pulsional (Trieb)", "Sigma": "\u03a3 (s\u00edntoma)"}
    tr, bloques, etiq_pasos = [], [], []

    for x in pasos:
        Mf = HornTorusFamilia(scl90r, r_over_R=x) if scl90r else HornTorusFamilia(r_over_R=x)
        inv = Mf.invariantes()
        rho = 0.93 * Mf.r
        i0 = len(tr)

        U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 110), np.linspace(0.0, np.pi, 60))
        X, Y, Z = Mf.punto(U, V)
        tr.append(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.zeros_like(U), colorscale=GRIS,
                             showscale=False, opacity=0.60, showlegend=False, hoverinfo="skip"))

        fases = [Mf.v_S % (2 * np.pi), Mf.v_I % (2 * np.pi), Mf.v_Sigma % (2 * np.pi)]
        vs = {}
        for k, (nom, ph) in enumerate(zip(["S", "I", "Sigma"], fases)):
            vs[nom] = v0 + A * np.sin(3.0 * u + ph + 2 * np.pi * k / 3)
        vs["Pulsion"] = vs["I"] + 0.075 + 0.055 * Mf.pulsion_attachment_strength
        for nom in ("S", "I", "Pulsion", "Sigma"):
            px, py, pz = Mf.punto(u, vs[nom], rho)
            tr.append(go.Scatter3d(x=px, y=py, z=pz, mode="lines",
                                   line=dict(color=COL["puls"] if nom == "Pulsion" else COL[nom],
                                             width=7 if nom != "Pulsion" else 4),
                                   name=etiquetas[nom], showlegend=(i0 == 0),
                                   legendgroup=nom, hoverinfo="skip"))

        ex, ey, ez = Mf.punto(u, np.pi)      # la longitud interior lambda_int
        tr.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color=COL["voz"], width=6),
                               name="\u03bb_int  (agujero interior \u2192 la voz)",
                               showlegend=(i0 == 0), legendgroup="lint",
                               hovertemplate="R \u2212 r = %.4f<extra></extra>" % inv["radio_agujero"]))
        for (uu, vv), col, nom, mk in (((Mf.u_F, Mf.v_F), COL["fant"], "n\u00facleo fantasm\u00e1tico", "square"),
                                       ((1.95, 2.25), COL["trauma"], "n\u00facleo del trauma", "diamond")):
            px, py, pz = Mf.punto(uu, vv, rho)
            tr.append(go.Scatter3d(x=[px], y=[py], z=[pz], mode="markers",
                                   marker=dict(size=6, color=col, symbol=mk,
                                               line=dict(color="white", width=1)),
                                   name=nom, showlegend=(i0 == 0), legendgroup=nom,
                                   hoverinfo="skip"))
        bloques.append((i0, len(tr)))
        W = "\u221e" if not np.isfinite(inv["willmore"]) else "%.1f" % inv["willmore"]
        etiq_pasos.append("r/R = %.4f \u00b7 R\u2212r = %.3f \u00b7 \u03c7 = %d \u00b7 rango H\u2081 = %d \u00b7 W = %s"
                          % (x, inv["radio_agujero"], inv["euler_characteristic"],
                             inv["rango_H1"], W))

    n = len(tr)
    pasos_slider = []
    for j, ((i0, i1), et) in enumerate(zip(bloques, etiq_pasos)):
        vis = [False] * n
        for i in range(i0, i1):
            vis[i] = True
        pasos_slider.append(dict(method="update", label="%.2f" % pasos[j],
                                 args=[{"visible": vis},
                                       {"annotations[0].text": et}]))

    fig = go.Figure(data=tr)
    for j, (i0, i1) in enumerate(bloques):
        for i in range(i0, i1):
            fig.data[i].visible = (j == len(bloques) - 1)

    fig.update_layout(
        title=dict(text="La familia r \u2192 R: recorr\u00e9 el deslizador y mir\u00e1 "
                        "qu\u00e9 se rompe s\u00f3lo en el l\u00edmite", x=0.02),
        annotations=[dict(text=etiq_pasos[-1], x=0.02, y=0.90, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=12), align="left")],
        sliders=[dict(active=len(pasos) - 1, currentvalue=dict(prefix="r/R = "),
                      pad=dict(t=30), steps=pasos_slider)],
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=-1.05))),
        legend=dict(x=0.0, y=0.34, bgcolor="rgba(255,255,255,0.78)", font=dict(size=11)),
        margin=dict(l=0, r=0, t=62, b=0), paper_bgcolor="white")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ----------------------------------------------------------------------
# Figura 7 — trauma alcanzable (Emma) y fantasía como agujero (Ein Kind…)
# ----------------------------------------------------------------------
def _carta_base(ax, banda=(0.45, 2.70), salidas=((0.95, "palabra"), (2.05, "agieren")),
                relleno=True):
    """Fondo común: la carta (u,v) de la cara interna, con las vías de salida.

    relleno=False deja ver un campo dibujado debajo (contourf): no pinta el
    rectángulo ni la banda, sólo marca los bordes de la banda.
    """
    if relleno:
        ax.add_patch(plt.Rectangle((0, 0), 2 * np.pi, 2 * np.pi, fc="#fafbfb",
                                   ec="#cfd8dc", lw=0.8))
        ax.axhspan(banda[0], banda[1], color="#eceff1", zorder=1)
    else:
        for y in banda:
            ax.axhline(y, color="#455a64", lw=0.7, ls=(0, (2, 2)), alpha=0.6, zorder=4)
    for v, nom in salidas:
        ax.axhline(v, color="#78909c", lw=1.1, ls=(0, (5, 3)), zorder=2)
        ax.text(2 * np.pi - 0.08, v + 0.07, nom, fontsize=7, color="#607d8b",
                ha="right", va="bottom")
    ax.axhline(np.pi, color=COL["voz"], lw=1.8, zorder=3)
    ax.text(2 * np.pi - 0.08, np.pi + 0.08, "la voz  (v = π)", fontsize=7,
            color=COL["voz"], ha="right", va="bottom")
    ax.text(0.10, banda[1] - 0.10, "cinta S-I-Σ", fontsize=7, va="top", zorder=6,
            color="#90a4ae" if relleno else "#3e2723")
    tk = [0, np.pi, 2 * np.pi]
    ax.set_xticks(tk); ax.set_xticklabels(["0", "π", "2π"])
    ax.set_yticks(tk); ax.set_yticklabels(["0", "π", "2π"])
    ax.set_xlim(-0.1, 2 * np.pi + 0.1); ax.set_ylim(-0.1, 2 * np.pi + 0.1)
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax.set_aspect("equal")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _flecha(ax, p0, p1, col, rad=0.18, lw=1.5, ls="-", alpha=1.0, zorder=6):
    ax.annotate("", xy=p1, xytext=p0, zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=col, alpha=alpha,
                                linestyle=ls, shrinkA=4, shrinkB=3,
                                connectionstyle=f"arc3,rad={rad}"))


def fig_trauma_fantasma(path="fig7_trauma_fantasma.png"):
    estilo()
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.75))

    # ---------------- (a) el trauma: Emma, Nachträglichkeit ----------------
    ax = axs[0]
    _carta_base(ax)
    T = (1.15, 1.95)                      # la escena de los 8 años
    M2 = (3.25, 1.42)                     # la escena de los 12: marca posterior
    Sal = (3.25, 0.95)

    _flecha(ax, T, (1.15, 1.02), "#b0bec5", rad=0.0, lw=1.3, ls=(0, (3, 2)), zorder=5)
    ax.plot([0.92, 1.38], [1.32, 1.62], color="#b71c1c", lw=1.6, zorder=8)
    ax.plot([0.92, 1.38], [1.62, 1.32], color="#b71c1c", lw=1.6, zorder=8)
    ax.text(0.78, 1.47, "época n:\nsin camino", fontsize=7.2, color="#b71c1c",
            ha="right", va="center", linespacing=1.3)

    _flecha(ax, T, M2, COL["trauma"], rad=-0.22)
    _flecha(ax, M2, Sal, COL["trauma"], rad=0.16)
    ax.plot(*T, "D", ms=7, mfc=COL["trauma"], mec="white", mew=0.8, zorder=7)
    ax.plot(*M2, "o", ms=6.5, mfc="#37474f", mec="white", mew=0.8, zorder=7)

    ax.annotate("T — escena de los 8 años\n(no traumática en su época)",
                xy=T, xytext=(0.15, 4.30), fontsize=7.4, color=COL["trauma"],
                ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["trauma"], shrinkB=5))
    ax.annotate("M₂ — escena de los 12\nmarca de una época posterior",
                xy=M2, xytext=(4.15, 4.30), fontsize=7.4, color="#37474f",
                ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#546e7a", shrinkB=5))
    ax.text(2.20, 5.35, "época n+1: el mismo punto,\nun camino que antes no existía",
            fontsize=7.8, color="#37474f", ha="center", va="center", linespacing=1.4)
    ax.text(2.30, 2.12, "elemento compartido\n(Kleider)", fontsize=7, color="#546e7a",
            ha="center", va="bottom", linespacing=1.3)
    ax.set_title("a · el trauma es alcanzable, pero no por sí mismo", loc="left", fontsize=8.8)

    # ---------------- (b) la fantasía: Ein Kind wird geschlagen ------------
    ax = axs[1]
    _carta_base(ax)
    f1, f3, f2 = (0.95, 2.28), (4.85, 1.35), (2.75, 1.62)

    _flecha(ax, f1, (0.95, 2.05), COL["I"], rad=0.0, lw=1.4)
    _flecha(ax, f3, (4.85, 0.95), COL["I"], rad=0.0, lw=1.4)
    ax.plot(*f1, "o", ms=6.5, mfc=COL["I"], mec="white", mew=0.8, zorder=7)
    ax.plot(*f3, "o", ms=6.5, mfc=COL["I"], mec="white", mew=0.8, zorder=7)

    t = np.linspace(0, 2 * np.pi, 300)
    lx, ly = f2[0] + 0.62 * np.cos(t), f2[1] + 0.42 * np.sin(t)
    ax.plot(lx, ly, color=COL["fant"], lw=1.7, zorder=6)
    ax.annotate("", xy=(lx[80], ly[80]), xytext=(lx[70], ly[70]), zorder=7,
                arrowprops=dict(arrowstyle="-|>", lw=1.7, color=COL["fant"]))
    ax.plot(*f2, "o", ms=8, mfc="white", mec=COL["fant"], mew=1.8, zorder=8)
    for dest in ((2.75, 0.95), (2.75, 2.05)):
        _flecha(ax, f2, dest, "#b0bec5", rad=0.0, lw=1.1, ls=(0, (3, 2)), zorder=5)
    for y0 in (1.18, 2.00):
        ax.plot([2.52, 2.98], [y0 - 0.11, y0 + 0.11], color="#b71c1c", lw=1.4, zorder=9)
        ax.plot([2.52, 2.98], [y0 + 0.11, y0 - 0.11], color="#b71c1c", lw=1.4, zorder=9)

    ax.annotate("φ₁ «el padre pega al niño»\nse recuerda", xy=f1, xytext=(0.15, 4.30),
                fontsize=7.4, color=COL["I"], ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["I"], shrinkB=5))
    ax.annotate("φ₃ «pegan a un niño»\nconciente", xy=f3, xytext=(4.55, 4.30),
                fontsize=7.4, color=COL["I"], ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["I"], shrinkB=5))
    ax.annotate("φ₂ «yo soy pegado por el padre»\nnunca conciente — construcción del análisis",
                xy=(f2[0], f2[1] + 0.42), xytext=(1.95, 5.55), fontsize=7.4, color=COL["fant"],
                ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["fant"], shrinkB=4))
    ax.text(4.25, 0.40, "la construcción rodea\nlo que no puede visitar",
            fontsize=7.6, color=COL["fant"], ha="left", va="bottom", linespacing=1.35)
    ax.text(1.95, 0.40, "ningún camino de salida,\nen ninguna época", fontsize=7.2,
            color="#b71c1c", ha="right", va="bottom", linespacing=1.3)
    ax.set_title("b · la fantasía no es un punto: es un agujero", loc="left", fontsize=8.8)

    fig.text(0.008, 0.985, "Trauma y fantasía no son dos puntos con distinto comportamiento: "
                           "uno es una marca, el otro es un defecto de la superficie de inscripción",
             fontsize=9.5, va="top")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.10, wspace=0.22)
    fig.savefig(path, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figura 8 — los dos mecanismos de la angustia: métrico y topológico
# ----------------------------------------------------------------------
def rodeo_longitud(a, b, eps):
    """Camino más corto entre dos puntos colineales con un agujero de radio eps."""
    if eps <= 0:
        return a + b
    return (np.sqrt(a ** 2 - eps ** 2) + np.sqrt(b ** 2 - eps ** 2)
            + eps * (np.pi - np.arccos(eps / a) - np.arccos(eps / b)))


def fig_angustia(path="fig8_angustia_dos_mecanismos.png", A_cr=np.pi / 4):
    estilo()
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.9))

    # ---------------- (a) cerca del trauma: métrico y graduado -------------
    ax = axs[0]
    T = (1.15, 1.95)
    U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 320), np.linspace(0, 2 * np.pi, 320))
    A = distancia_angular(U, V, T[0], T[1])
    im = ax.contourf(U, V, A, levels=12, cmap="YlOrRd_r", zorder=1)
    ax.contour(U, V, A, levels=[A_cr], colors=[COL["voz"]], linewidths=1.4, zorder=5)
    _carta_base(ax, salidas=(), relleno=False)
    ax.axhline(0.95, color="#263238", lw=1.1, ls=(0, (5, 3)), zorder=4)
    ax.text(2 * np.pi - 0.10, 1.04, "palabra", fontsize=7, color="#263238",
            ha="right", va="bottom", zorder=6)
    ax.plot(*T, "D", ms=7.5, mfc=COL["trauma"], mec="white", mew=0.9, zorder=8)
    for u0 in (0.35, 1.95):
        _flecha(ax, (u0, 3.05), (T[0] + 0.12 * np.sign(u0 - T[0]), T[1] + 0.34),
                "#4a148c", rad=0.10, lw=1.4, zorder=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("A_T  (distancia envuelta a T)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    ax.text(0.12, 5.55, "mecanismo MÉTRICO y graduado\nla angustia crece al acercarse;\n"
                        "el umbral A_cr = π/4 tiene sentido acá",
            fontsize=7.8, color="#1a1a1a", va="center", linespacing=1.4, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.72, pad=2.5))
    ax.annotate("A_cr", xy=(T[0] + A_cr * 0.72, T[1] - A_cr * 0.72), xytext=(2.35, 0.55),
                fontsize=7.4, color=COL["voz"], zorder=7,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["voz"], shrinkB=2))
    ax.set_title("a · cerca del trauma", loc="left", fontsize=8.8)

    # ---------------- (b) cerca del agujero: topológico --------------------
    ax = axs[1]
    _carta_base(ax, salidas=((0.95, "palabra"),))
    F = (2.75, 1.62)
    P = (2.75, 2.52)
    eps = 0.13
    _flecha(ax, P, (2.75, 1.05), "#b0bec5", rad=0.0, lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax.plot([2.55, 2.95], [1.98, 2.26], color="#b71c1c", lw=1.5, zorder=9)
    ax.plot([2.55, 2.95], [2.26, 1.98], color="#b71c1c", lw=1.5, zorder=9)

    for lado, col in ((-1, "#00838f"), (+1, "#ad1457")):
        t = np.linspace(0, 1, 200)
        x = P[0] + lado * 0.92 * np.sin(np.pi * t) * (0.55 + 0.45 * np.sin(np.pi * t))
        y = P[1] + (0.95 - P[1]) * t
        ax.plot(x, y, color=col, lw=2.0, zorder=7, solid_capstyle="round")
        ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-6], y[-6]), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=col))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(F[0] + 0.50 * np.cos(t), F[1] + 0.34 * np.sin(t), color=COL["fant"],
            lw=1.5, ls=(0, (4, 2)), zorder=6)
    ax.plot(*F, "o", ms=9, mfc="white", mec=COL["fant"], mew=2.0, zorder=10)
    ax.plot(*P, "o", ms=5, color="#4a148c", zorder=8)

    ax.text(0.12, 5.62, "mecanismo TOPOLÓGICO, no graduado\nel agujero no se puede atravesar,\n"
                        "y los dos modos de rodearlo no son\nel mismo camino",
            fontsize=7.8, color="#37474f", va="center", linespacing=1.4, zorder=6)
    ax.text(0.12, 4.05, "métricamente el rodeo no cuesta casi nada:\n"
                        "el exceso va como ε²  (ε = 0.05 → +0.50 %;\n"
                        "ε = 0.01 → +0.02 %; en el límite, cero)",
            fontsize=7.4, color="#546e7a", va="center", linespacing=1.4, zorder=6)
    ax.annotate("no homotópicos:\nninguno se deforma en el otro",
                xy=(3.62, 1.72), xytext=(4.30, 2.72), fontsize=7.4, color="#37474f",
                ha="left", va="center", linespacing=1.35, zorder=7,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#607d8b", shrinkB=3))
    ax.annotate("la tramitación es el circuito;\nlo que sale de ahí es construcción",
                xy=(F[0] - 0.50, F[1]), xytext=(0.12, 0.42), fontsize=7.4,
                color=COL["fant"], ha="left", va="bottom", linespacing=1.35, zorder=7,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL["fant"], shrinkB=3))
    ax.set_title("b · cerca del agujero", loc="left", fontsize=8.8)

    fig.text(0.008, 0.985, "La angustia no se tramita igual en los dos casos: junto al trauma "
                           "es proximidad; junto al agujero es que hay que elegir un lado",
             fontsize=9.5, va="top")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.855, bottom=0.10, wspace=0.28)
    fig.savefig(path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    M = HornTorusICC()
    fig_corte_axial()
    fig_cinta(M=M)
    fig_vias_salida()
    fig_heegaard()
    fig_familia_limite()
    fig_invariantes_familia()
    fig_trauma_fantasma()
    fig_angustia()
    html_interactivo(M=M)
    html_familia()
    print("[OK] fig1_corte_axial.png, fig2_cinta_cara_interna.png, "
          "fig3_vias_de_salida.png, fig4_heegaard_no_anudamiento.png, "
          "horn_torus_icc_interactivo.html")
