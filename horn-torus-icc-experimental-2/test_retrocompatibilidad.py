#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control de integridad del repo. Es el único control que importa cuando se
modifica el motor —a mano o con un asistente—: con r_over_R = 1.0 la familia
tiene que reproducir HornTorusICC de forma IDÉNTICA.

Uso:
    python test_retrocompatibilidad.py
"""

import numpy as np

from horn_torus_experimental import HornTorusICC
from familia_horn_torus import HornTorusFamilia

FALLAS = []


def check(nombre, cond, extra=""):
    print(("  [ok]   " if cond else "  [FALLA] ") + nombre + (("  " + extra) if extra else ""))
    if not cond:
        FALLAS.append(nombre)


def main():
    base, lim = HornTorusICC(), HornTorusFamilia(r_over_R=1.0)

    print("1. retrocompatibilidad en r = R")
    b = base.horn_torus_surface(40, 40)[:3]
    l = lim.horn_torus_surface(40, 40)[:3]
    check("superficie idéntica", np.allclose(b, l))
    cb, cl = base.get_curves(), lim.get_curves()
    check("las cuatro cintas idénticas",
          set(cb) == set(cl) and all(np.allclose(cb[k][c], cl[k][c]) for k in cb for c in "xyz"))
    check("punto de fantasía idéntico", np.allclose(base.fantasy_point_3d, lim.fantasy_point_3d))

    print("2. invariantes exactos contra integración numérica")
    for x in (0.45, 0.70, 0.90):
        m = HornTorusFamilia(r_over_R=x)
        i = m.invariantes()
        v = np.linspace(0, 2 * np.pi, 20001)
        K = np.cos(v) / (m.r * (m.R + m.r * np.cos(v)))
        dA = m.r * (m.R + m.r * np.cos(v))
        gb = 2 * np.pi * np.trapezoid(K * dA, v)                     # Gauss-Bonnet: 2*pi*chi = 0
        H = (m.R + 2 * m.r * np.cos(v)) / (2 * m.r * (m.R + m.r * np.cos(v)))
        W = 2 * np.pi * np.trapezoid(H ** 2 * dA, v)
        check("r/R = %.2f  Gauss-Bonnet = 0" % x, abs(gb) < 1e-8, "(%.2e)" % gb)
        check("r/R = %.2f  Willmore cerrada = numérica" % x,
              np.isclose(i["willmore"], W, rtol=1e-6), "(%.4f vs %.4f)" % (i["willmore"], W))

    print("3. el límite no es una variedad")
    i = HornTorusFamilia(r_over_R=1.0).invariantes()
    check("chi = 1 y rango de H1 = 1", i["euler_characteristic"] == 1 and i["rango_H1"] == 1)
    check("no es variedad", not i["es_variedad"])
    P = np.stack(HornTorusFamilia(r_over_R=1.0).punto(np.linspace(0, 2 * np.pi, 500), np.pi), -1)
    check("lambda_int colapsa a un punto", float(np.abs(P).max()) < 1e-12)
    check("W y K_min divergen",
          not np.isfinite(i["willmore"]) and not np.isfinite(i["curvatura_gauss_min"]))

    print("4. mínimo de Willmore en el toro de Clifford")
    c = HornTorusFamilia(r_over_R=1 / np.sqrt(2)).invariantes()
    check("W(1/raíz2) = 2*pi^2", np.isclose(c["willmore"], 2 * np.pi ** 2),
          "(%.4f)" % c["willmore"])

    print("\n" + ("TODO OK" if not FALLAS else "FALLAS: " + ", ".join(FALLAS)))
    return 1 if FALLAS else 0


if __name__ == "__main__":
    raise SystemExit(main())
