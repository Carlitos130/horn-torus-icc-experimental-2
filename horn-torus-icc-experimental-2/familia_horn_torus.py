#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
La familia de toros de revolución r -> R, con el horn torus como caso límite.

Extiende `HornTorusICC` sin tocarlo: `HornTorusFamilia(r_over_R=1.0)` reproduce
exactamente el motor actual, y `r_over_R < 1` recorre la familia de toros lisos
en la que valen Heegaard, Alexander y el flujo de Ricci (ver
seccion16_familia_limite.md).

Geometría (R = radio de revolución, r = radio del tubo):

    x = (R + r cos v) cos u ,  y = (R + r cos v) sin u ,  z = r sin v

Con r = R se recupera `a(1 + cos v)`, la parametrización del motor.

Invariantes exactos, todos verificados contra integración numérica:

    A   = 4 pi^2 R r                    área
    V   = 2 pi^2 R r^2                  volumen encerrado
    K   = cos v / (r (R + r cos v))     curvatura de Gauss
    H   = (R + 2 r cos v) / (2 r (R + r cos v))
    <H> = 1 / (2 r)                     curvatura media promediada en área
    W   = pi^2 rho^2 / sqrt(rho^2 - 1)  energía de Willmore, rho = R/r > 1

`K_min = -1/(r(R-r))` y `W` **divergen** cuando r -> R: el límite donde aparece
la voz es un punto de energía de flexión infinita. `K_max = 1/(r(R+r))` y `<H>`
se quedan finitos.

Ojo con la línea de base del motor: `willmore_energy_standard = 2 pi^2` es el
valor del toro de Clifford (r/R = 1/sqrt(2) = 0.7071), que es el mínimo de W
sobre la familia — no el valor del horn torus, que es infinito.
"""

import numpy as np

from horn_torus_experimental import HornTorusICC


class HornTorusFamilia(HornTorusICC):
    """HornTorusICC con el cociente r/R explícito. r_over_R = 1 es el horn torus."""

    def __init__(self, *args, r_over_R=1.0, **kwargs):
        if not (0.0 < r_over_R <= 1.0):
            raise ValueError("r_over_R debe estar en (0, 1]; 1 es el horn torus")
        self.r_over_R = float(r_over_R)
        super().__init__(*args, **kwargs)
        # el punto de fantasía en escala natural, recalculado para la familia
        R0, r0 = self.a, self.r_over_R * self.a
        self.fantasy_point_3d = (
            (R0 + r0 * np.cos(self.v_F)) * np.cos(self.u_F),
            (R0 + r0 * np.cos(self.v_F)) * np.sin(self.u_F),
            r0 * np.sin(self.v_F),
        )

    # ------------------------------------------------------------------
    # geometría
    # ------------------------------------------------------------------
    @property
    def R(self):
        """Radio de revolución (en escala visual)."""
        return self.effective_a

    @property
    def r(self):
        """Radio del tubo (en escala visual)."""
        return self.r_over_R * self.effective_a

    @property
    def es_limite(self):
        return self.r_over_R >= 1.0

    @property
    def radio_agujero(self):
        """R - r: radio del agujero central. Cero en el límite."""
        return self.R - self.r

    def punto(self, u, v, rho=None):
        """Punto del toro sólido; rho = 0 es el círculo-núcleo, rho = r la superficie."""
        rho = self.r if rho is None else rho
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        rad = self.R + rho * np.cos(v)
        return rad * np.cos(u), rad * np.sin(u), rho * np.sin(v)

    def horn_torus_surface(self, u_resolution=None, v_resolution=None, deformed=False):
        res = self.grid_resolution
        u = np.linspace(0, 2 * np.pi, u_resolution or res)
        v = np.linspace(0, 2 * np.pi, v_resolution or res)
        u, v = np.meshgrid(u, v)
        x, y, z = self.punto(u, v)
        if deformed:
            factor, _ = self.compute_scl_deformation(u, v)
            x, y = x * factor, y * factor
            z = z * (1.0 + (factor - 1.0) * 0.85)
        return x, y, z, u, v

    def get_curves(self, u_points=220):
        """Las mismas cuatro cintas del motor, sobre la familia."""
        u_vals = np.linspace(0, 2 * np.pi, u_points + 1)
        R, rt = self.R, self.r
        phi_I = self.v_I % (2 * np.pi)
        phi_S = self.v_S % (2 * np.pi)
        phi_Sigma = self.v_Sigma % (2 * np.pi)

        v_I = np.pi + 0.48 * np.sin(u_vals + phi_I) + 0.12 * np.cos(2 * u_vals)
        r_I = R + rt * np.cos(v_I)

        f = 6
        s = self.pulsion_attachment_strength
        v_P = v_I + 0.14 * np.sin(f * u_vals) * s
        r_P = R + rt * np.cos(v_P) + 0.08 * np.cos(f * u_vals) * rt * s
        z_P = rt * np.sin(v_P) + 0.06 * np.sin(f * u_vals) * rt

        v_S = np.pi + 0.48 * np.cos(u_vals + phi_S) - 0.16 * np.sin(2 * u_vals)
        r_S = R + rt * np.cos(v_S)

        v_Sg = np.pi + 0.58 * np.sin(2 * u_vals + phi_Sigma)
        r_Sg = R + rt * np.cos(v_Sg)

        def _c(rad, v, col, lab, z=None):
            return {"x": rad * np.cos(u_vals), "y": rad * np.sin(u_vals),
                    "z": rt * np.sin(v) if z is None else z, "color": col, "label": lab}

        return {
            "S": _c(r_S, v_S, "crimson", "S (Significante)"),
            "I": _c(r_I, v_I, "seagreen", "I (Imagen del cuerpo)"),
            "Pulsion": _c(r_P, v_P, "goldenrod", "Hilo Pulsional (Trieb)", z=z_P),
            "Sigma": _c(r_Sg, v_Sg, "royalblue", "Sigma (Sintoma)"),
        }

    def find_rupture_points(self, u_resolution=50, v_resolution=50):
        u = np.linspace(0, 2 * np.pi, u_resolution)
        v = np.linspace(0, 2 * np.pi, v_resolution)
        ug, vg = np.meshgrid(u, v)
        ang = self.calculate_angustia(ug, vg)
        m = ang <= self.A_cr
        x, y, z = self.punto(ug, vg)
        return x[m], y[m], z[m], ang[m]

    def compute_differential_tension(self, u, v, deformation_factor=None, effective_a=None):
        """Igual que en el motor, pero con el punto base de la familia."""
        delta = self.deformation_factor if deformation_factor is None else deformation_factor
        escala = max(0.05, delta)

        x0, y0, z0 = self.punto(u, v)
        factor, stress = self.compute_scl_deformation(u, v, delta)
        dx = x0 * factor - x0
        dy = y0 * factor - y0
        dz = z0 * (1.0 + (factor - 1.0) * 0.85) - z0
        desp = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / (self.R * escala + 1e-5)

        eps = 0.04
        gu = (self.compute_scl_deformation(u + eps, v, delta)[0]
              - self.compute_scl_deformation(u - eps, v, delta)[0]) / (2 * eps)
        gv = (self.compute_scl_deformation(u, v + eps, delta)[0]
              - self.compute_scl_deformation(u, v - eps, delta)[0]) / (2 * eps)
        grad = np.sqrt(gu ** 2 + gv ** 2)

        # el cuello: R + r cos v -> 0 sólo en el límite y sólo en v = pi
        cuello = (self.R + self.r * np.cos(v)) / self.R
        strain = (np.abs(gu) * 1.5 + np.abs(gv) * 0.8) / (cuello + 0.18)

        bruto = (0.35 * np.minimum(2.5, desp) + 0.35 * np.minimum(3.0, grad * 2.0)
                 + 0.30 * np.minimum(3.5, strain * 0.45) + 0.20 * stress)
        return {"tension": np.clip(bruto / 1.75, 0.0, 1.0), "displacement_norm": desp,
                "gradient_mag": grad, "cusp_strain": strain}

    # ------------------------------------------------------------------
    # invariantes exactos
    # ------------------------------------------------------------------
    def curvatura_gauss(self, v):
        return np.cos(v) / (self.r * (self.R + self.r * np.cos(v)))

    def curvatura_media(self, v):
        return (self.R + 2 * self.r * np.cos(v)) / (2 * self.r * (self.R + self.r * np.cos(v)))

    def invariantes(self):
        """Invariantes geométricos y topológicos exactos del miembro actual."""
        R, r = self.R, self.r
        limite = self.es_limite
        rho = R / r
        return {
            "r_over_R": self.r_over_R,
            "R": R, "r": r, "radio_agujero": self.radio_agujero,
            "regimen": "límite (toro pinchado)" if limite else "toro liso encajado",
            "es_variedad": not limite,
            "euler_characteristic": 1 if limite else 0,
            "rango_H1": 1 if limite else 2,
            "H1": "Z<mu>" if limite else "Z^2<mu, lambda>",
            "area": 4 * np.pi ** 2 * R * r,
            "volumen": 2 * np.pi ** 2 * R * r ** 2,
            "curvatura_gauss_max": 1.0 / (r * (R + r)),
            "curvatura_gauss_min": -np.inf if limite else -1.0 / (r * (R - r)),
            "curvatura_media_promedio": 1.0 / (2 * r),
            "willmore": np.inf if limite else np.pi ** 2 * rho ** 2 / np.sqrt(rho ** 2 - 1),
            "willmore_minimo_familia": 2 * np.pi ** 2,     # toro de Clifford, r/R = 1/sqrt(2)
        }

    def compute_topological_metrics(self):
        """Las métricas del motor, con las geométricas reemplazadas por las exactas."""
        m = super().compute_topological_metrics()
        inv = self.invariantes()
        delta = self.deformation_factor
        d = self.scl90r_data
        m["surface_area_standard"] = inv["area"]
        m["volume_standard"] = inv["volumen"]
        exp_a = 1.0 + delta * (0.28 * d.get("Somatizacion", 0) + 0.35 * d.get("GSI", 0)
                               + 0.20 * d.get("PST", 0))
        exp_v = 1.0 + delta * (0.42 * d.get("GSI", 0) - 0.18 * d.get("Psicoticismo", 0)
                               + 0.15 * d.get("Obsesion-Compulsion", 0))
        m["surface_area_deformed"] = inv["area"] * exp_a
        m["volume_deformed"] = inv["volumen"] * exp_v
        m["willmore_energy_standard"] = inv["willmore"]
        m["gaussian_curvature_min"] = inv["curvatura_gauss_min"]
        m["gaussian_curvature_max"] = inv["curvatura_gauss_max"]
        m["mean_curvature_avg"] = inv["curvatura_media_promedio"]
        m["euler_characteristic"] = inv["euler_characteristic"]
        m["rango_H1"] = inv["rango_H1"]
        m["regimen"] = inv["regimen"]
        return m

    def resumen_geometrico(self):
        inv = self.invariantes()
        inf = lambda x: "infinito" if not np.isfinite(x) else f"{x:.4f}"
        return f"""{'=' * 74}
  FAMILIA r -> R   ·   r/R = {inv['r_over_R']:.4f}   ·   {inv['regimen']}
{'=' * 74}
  R (radio de revolución)         {inv['R']:.4f}
  r (radio del tubo)              {inv['r']:.4f}
  R - r (radio del agujero)       {inv['radio_agujero']:.4f}
  ¿es una variedad?               {'sí' if inv['es_variedad'] else 'NO — dos hojas en un punto'}
  característica de Euler  chi    {inv['euler_characteristic']}
  primer grupo de homología H1    {inv['H1']}   (rango {inv['rango_H1']})
  área            4 pi^2 R r      {inv['area']:.4f}
  volumen         2 pi^2 R r^2    {inv['volumen']:.4f}
  curvatura de Gauss   max        {inv['curvatura_gauss_max']:.4f}
                       min        {inf(inv['curvatura_gauss_min'])}
  curvatura media  <H> = 1/(2r)   {inv['curvatura_media_promedio']:.4f}
  energía de Willmore  W          {inf(inv['willmore'])}
      (mínimo de la familia: 2 pi^2 = {inv['willmore_minimo_familia']:.4f} en r/R = 1/raíz(2))
{'=' * 74}"""


if __name__ == "__main__":
    for x in (0.45, 1 / np.sqrt(2), 0.90, 1.0):
        print(HornTorusFamilia(r_over_R=x).resumen_geometrico())
