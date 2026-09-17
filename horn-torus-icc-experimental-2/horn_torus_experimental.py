#!/usr/bin/env python3
"""
Modelo 3D del Horn Torus para el Icc (Inconsciente) - motor sofisticado.

Puerto fiel a Python del motor React/TypeScript real del repo
horn-torus-icc-model (src/utils/hornTorusMath.ts): calcula las curvas
interiores entrelazadas S (Significante), I (Imagen del cuerpo), Hilo
Pulsional (Trieb / Vorstellungsrepraesentanz) y Sigma (Sintoma/Sinthome),
el punto de fantasia como foco de angustia maxima, la deformacion del
manifold en funcion del SCL-90-R, la tension topologica diferencial y
las metricas clinicas/topologicas (area, volumen, energia de Willmore,
entropia, Indice ICC, severidad clinica).

Agrega ademas visualizacion interactiva (Plotly), exportacion a HTML
standalone y exportacion de datos a JSON.
"""

import json
from datetime import datetime, timezone

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registra proyeccion 3d)
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


class HornTorusICC:
    """Modela y visualiza el Horn Torus del Icc a partir de un SCL-90-R."""

    # Constante de escala visual usada por el visor React/Three.js del repo
    # (src/utils/hornTorusMath.ts: visualScale = 25.0). Se mantiene aca para
    # que las metricas (area, volumen, energia de Willmore) coincidan
    # numericamente con las que muestra la app real.
    VISUAL_SCALE = 25.0

    # El motor TS usa un valor fijo de 9 escalas primarias (excluye GSI,
    # PST y PSDI), no la cantidad real de claves del dict de entrada.
    NUM_PRIMARY_SCALES = 9

    DEFAULT_SCL90R_DATA = {
        "Somatizacion": 0.8,
        "Obsesion-Compulsion": 0.9,
        "Sensibilidad Interpersonal": 0.7,
        "Depresion": 0.85,
        "Ansiedad": 0.95,
        "Hostilidad": 0.6,
        "Ansiedad Fobica": 0.75,
        "Ideacion Paranoide": 0.8,
        "Psicoticismo": 0.9,
        "GSI": 0.85,
        "PST": 0.7,
        "PSDI": 0.9,
    }

    def __init__(
        self,
        scl90r_data=None,
        a_scale=0.1,
        u_scale=2 * np.pi,
        v_scale=np.pi,
        deformation_factor=0.3,
        grid_resolution=80,
        a_critical=None,
    ):
        self.scl90r_data = scl90r_data if scl90r_data else dict(self.DEFAULT_SCL90R_DATA)
        self.a_scale = a_scale
        self.u_scale = u_scale
        self.v_scale = v_scale
        self.deformation_factor = deformation_factor
        self.grid_resolution = grid_resolution
        self.A_cr = a_critical if a_critical is not None else np.pi / 4

        self._calculate_lacanian_parameters()

    def _d(self, key, default):
        return self.scl90r_data.get(key, default)

    # ------------------------------------------------------------------
    # Parametros lacanianos (equivalente a calculateLacanianParameters en TS)
    # ------------------------------------------------------------------
    def _calculate_lacanian_parameters(self):
        gsi = self._d("GSI", 0.85)
        self.a = self.a_scale * gsi
        self.effective_a = self.a * self.VISUAL_SCALE

        n = self.NUM_PRIMARY_SCALES

        anxiety = self._d("Ansiedad", 0.95)
        obsession = self._d("Obsesion-Compulsion", 0.9)
        self.u_S = (self.u_scale * (anxiety + obsession)) / n

        psdi = self._d("PSDI", 0.9)
        self.v_S = self.v_scale * (1 + psdi)

        somatization = self._d("Somatizacion", 0.8)
        interpersonal = self._d("Sensibilidad Interpersonal", 0.7)
        self.u_I = (self.u_scale * (somatization + interpersonal)) / n

        pst = self._d("PST", 0.7)
        self.v_I = self.v_scale * (1 + pst)

        # Hilo Pulsional: pegado a I (apuntalamiento somatico de la pulsion)
        depression = self._d("Depresion", 0.8)
        self.pulsion_attachment_strength = min(
            1.0, max(0.3, 0.88 + somatization * 0.12 - depression * 0.15)
        )
        self.u_pulsion = self.u_I
        self.v_pulsion = self.v_I

        psychoticism = self._d("Psicoticismo", 0.9)
        hostility = self._d("Hostilidad", 0.6)
        self.u_Sigma = (self.u_scale * (psychoticism + hostility)) / n
        self.v_Sigma = self.v_scale * (1 + psychoticism)

        # Punto de fantasia (angustia maxima): (pi, pi/2)
        self.u_F = np.pi
        self.v_F = np.pi / 2
        self.fantasy_point = (self.u_F, self.v_F)

        # Coordenadas 3D del punto de fantasia en escala "natural" (a, no
        # escalada por VISUAL_SCALE) - asi es como lo reporta el resumen
        # de la app real.
        self.fantasy_point_3d = (
            self.a * (1 + np.cos(self.v_F)) * np.cos(self.u_F),
            self.a * (1 + np.cos(self.v_F)) * np.sin(self.u_F),
            self.a * np.sin(self.v_F),
        )

        # Conteo de puntos de ruptura: muestreo 60x60 en el espacio (u, v)
        # de la distancia al punto de fantasia (igual que el motor TS).
        sample_steps = 60
        idx = np.arange(sample_steps)
        vals = (idx / sample_steps) * 2 * np.pi
        u_grid, v_grid = np.meshgrid(vals, vals, indexing="ij")
        dist = np.sqrt((u_grid - self.u_F) ** 2 + (v_grid - self.v_F) ** 2)
        rupture_mask = dist <= self.A_cr
        self.rupture_count = int(rupture_mask.sum())
        self.rupture_area_percent = self.rupture_count / (sample_steps * sample_steps) * 100

    @property
    def fantasy_point_3d_visual(self):
        """Punto de fantasia escalado igual que las curvas/superficie (para graficar)."""
        return tuple(c * self.VISUAL_SCALE for c in self.fantasy_point_3d)

    def calculate_angustia(self, u, v):
        """Angustia A(u, v) = distancia al punto de fantasia en el espacio (u, v)."""
        return np.sqrt((u - self.u_F) ** 2 + (v - self.v_F) ** 2)

    # ------------------------------------------------------------------
    # Superficie y curvas interiores entrelazadas (equivalente a
    # getLacanianCurves en TS)
    # ------------------------------------------------------------------
    def horn_torus_surface(self, u_resolution=None, v_resolution=None, deformed=False):
        res = self.grid_resolution
        u_resolution = u_resolution or res
        v_resolution = v_resolution or res

        u = np.linspace(0, 2 * np.pi, u_resolution)
        v = np.linspace(0, 2 * np.pi, v_resolution)
        u, v = np.meshgrid(u, v)

        a = self.effective_a
        x = a * (1 + np.cos(v)) * np.cos(u)
        y = a * (1 + np.cos(v)) * np.sin(u)
        z = a * np.sin(v)

        if deformed:
            factor, _stress = self.compute_scl_deformation(u, v)
            x = x * factor
            y = y * factor
            z = z * (1.0 + (factor - 1.0) * 0.85)

        return x, y, z, u, v

    def get_curves(self, u_points=220):
        """Curvas S, I, Hilo Pulsional y Sigma, entrelazadas en el interior del toro."""
        u_vals = np.linspace(0, 2 * np.pi, u_points + 1)
        eff_a = self.effective_a

        phi_I = self.v_I % (2 * np.pi)
        phi_S = self.v_S % (2 * np.pi)
        phi_Sigma = self.v_Sigma % (2 * np.pi)

        # I (Imagen del cuerpo): oscila a traves de la garganta/cuspide
        v_I_curve = np.pi + 0.48 * np.sin(u_vals + phi_I) + 0.12 * np.cos(2 * u_vals)
        r_I = eff_a * (1 + np.cos(v_I_curve))
        x_I = r_I * np.cos(u_vals)
        y_I = r_I * np.sin(u_vals)
        z_I = eff_a * np.sin(v_I_curve)

        # Hilo Pulsional: acoplado helicoidalmente a I
        pulsion_freq = 6
        v_offset = 0.14 * np.sin(pulsion_freq * u_vals) * self.pulsion_attachment_strength
        v_P = v_I_curve + v_offset
        r_offset = 0.08 * np.cos(pulsion_freq * u_vals) * eff_a * self.pulsion_attachment_strength
        r_P = eff_a * (1 + np.cos(v_P)) + r_offset
        x_P = r_P * np.cos(u_vals)
        y_P = r_P * np.sin(u_vals)
        z_P = eff_a * np.sin(v_P) + 0.06 * np.sin(pulsion_freq * u_vals) * eff_a

        # S (Significante): corta transversalmente a I y al Hilo Pulsional
        v_S_curve = np.pi + 0.48 * np.cos(u_vals + phi_S) - 0.16 * np.sin(2 * u_vals)
        r_S = eff_a * (1 + np.cos(v_S_curve))
        x_S = r_S * np.cos(u_vals)
        y_S = r_S * np.sin(u_vals)
        z_S = eff_a * np.sin(v_S_curve)

        # Sigma (Sintoma): cuarto lazo que anuda la estructura
        v_Sigma_curve = np.pi + 0.58 * np.sin(2 * u_vals + phi_Sigma)
        r_Sigma = eff_a * (1 + np.cos(v_Sigma_curve))
        x_Sigma = r_Sigma * np.cos(u_vals)
        y_Sigma = r_Sigma * np.sin(u_vals)
        z_Sigma = eff_a * np.sin(v_Sigma_curve)

        return {
            "S": {"x": x_S, "y": y_S, "z": z_S, "color": "crimson", "label": "S (Significante)"},
            "I": {"x": x_I, "y": y_I, "z": z_I, "color": "seagreen", "label": "I (Imagen del cuerpo)"},
            "Pulsion": {"x": x_P, "y": y_P, "z": z_P, "color": "goldenrod", "label": "Hilo Pulsional (Trieb)"},
            "Sigma": {"x": x_Sigma, "y": y_Sigma, "z": z_Sigma, "color": "royalblue", "label": "Sigma (Sintoma)"},
        }

    def find_rupture_points(self, u_resolution=50, v_resolution=50):
        u = np.linspace(0, 2 * np.pi, u_resolution)
        v = np.linspace(0, 2 * np.pi, v_resolution)
        u_grid, v_grid = np.meshgrid(u, v)

        angustia = self.calculate_angustia(u_grid, v_grid)
        mask = angustia <= self.A_cr

        a = self.effective_a
        x = a * (1 + np.cos(v_grid)) * np.cos(u_grid)
        y = a * (1 + np.cos(v_grid)) * np.sin(u_grid)
        z = a * np.sin(v_grid)

        return x[mask], y[mask], z[mask], angustia[mask]

    # ------------------------------------------------------------------
    # Deformacion sintomatica y tension topologica diferencial
    # (equivalente a computeSclDeformation / computeDifferentialTension)
    # ------------------------------------------------------------------
    def compute_scl_deformation(self, u, v, deformation_factor=None):
        deformation_factor = self.deformation_factor if deformation_factor is None else deformation_factor
        d = self.scl90r_data

        som = d.get("Somatizacion", 0)
        oc = d.get("Obsesion-Compulsion", 0)
        psy = d.get("Psicoticismo", 0)
        gsi = d.get("GSI", 0)
        pst = d.get("PST", 0)
        psdi = d.get("PSDI", 0)
        dep = d.get("Depresion", 0)
        anx = d.get("Ansiedad", 0)

        w_som = som * 0.28 * np.cos(3 * v) * (1 + 0.35 * np.cos(u))
        w_oc = oc * 0.32 * np.sin(4 * u) * np.cos(v)
        cusp_dist = np.abs(np.sin(v * 0.5))
        w_psy = psy * 0.45 * np.power(cusp_dist, 3) * np.sin(2 * u + v)
        w_gsi = gsi * 0.22 * (np.cos(v) + 0.5 * np.sin(u))
        w_pst = pst * 0.15 * np.sin(5 * v + 3 * u)
        w_psdi = psdi * 0.20 * np.cos(2 * v - 2 * u)
        w_dep = dep * 0.18 * np.sin(v)
        w_anx = anx * 0.14 * np.sin(8 * u) * np.cos(2 * v)

        raw_deform = w_som + w_oc + w_psy + w_gsi + w_pst + w_psdi + w_dep + w_anx
        factor = 1.0 + deformation_factor * raw_deform
        stress = np.clip(np.abs(raw_deform) * (1 + psdi * 0.5), 0.0, 1.0)
        return factor, stress

    def compute_differential_tension(self, u, v, deformation_factor=None, effective_a=None):
        deformation_factor = self.deformation_factor if deformation_factor is None else deformation_factor
        effective_a = self.effective_a if effective_a is None else effective_a
        deform_scale = max(0.05, deformation_factor)

        x0 = effective_a * (1 + np.cos(v)) * np.cos(u)
        y0 = effective_a * (1 + np.cos(v)) * np.sin(u)
        z0 = effective_a * np.sin(v)

        factor, stress = self.compute_scl_deformation(u, v, deformation_factor)
        x_def = x0 * factor
        y_def = y0 * factor
        z_def = z0 * (1.0 + (factor - 1.0) * 0.85)

        dx, dy, dz = x_def - x0, y_def - y0, z_def - z0
        displacement = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        displacement_norm = displacement / (effective_a * deform_scale + 1e-5)

        eps = 0.04
        f_u1, _ = self.compute_scl_deformation(u + eps, v, deformation_factor)
        f_u0, _ = self.compute_scl_deformation(u - eps, v, deformation_factor)
        f_v1, _ = self.compute_scl_deformation(u, v + eps, deformation_factor)
        f_v0, _ = self.compute_scl_deformation(u, v - eps, deformation_factor)
        grad_u = (f_u1 - f_u0) / (2 * eps)
        grad_v = (f_v1 - f_v0) / (2 * eps)
        gradient_mag = np.sqrt(grad_u ** 2 + grad_v ** 2)

        throat_radius = 1 + np.cos(v)
        cusp_strain = (np.abs(grad_u) * 1.5 + np.abs(grad_v) * 0.8) / (throat_radius + 0.18)

        raw_tension = (
            0.35 * np.minimum(2.5, displacement_norm)
            + 0.35 * np.minimum(3.0, gradient_mag * 2.0)
            + 0.30 * np.minimum(3.5, cusp_strain * 0.45)
            + 0.20 * stress
        )
        tension = np.clip(raw_tension / 1.75, 0.0, 1.0)

        return {
            "tension": tension,
            "displacement_norm": displacement_norm,
            "gradient_mag": gradient_mag,
            "cusp_strain": cusp_strain,
        }

    # ------------------------------------------------------------------
    # Metricas topologicas y clinicas (equivalente a computeTopologicalMetrics)
    # ------------------------------------------------------------------
    def compute_topological_metrics(self):
        a = self.effective_a
        delta = self.deformation_factor
        d = self.scl90r_data

        gsi = d.get("GSI", 0.85)
        psy = d.get("Psicoticismo", 0.9)
        som = d.get("Somatizacion", 0.8)
        oc = d.get("Obsesion-Compulsion", 0.9)
        psdi = d.get("PSDI", 0.9)
        pst = d.get("PST", 0.7)

        surface_area_standard = 4 * np.pi ** 2 * a ** 2
        volume_standard = 2 * np.pi ** 2 * a ** 3

        area_expansion = 1.0 + delta * (0.28 * som + 0.35 * gsi + 0.20 * pst)
        surface_area_deformed = surface_area_standard * area_expansion
        surface_area_delta_pct = (surface_area_deformed - surface_area_standard) / surface_area_standard * 100

        vol_expansion = 1.0 + delta * (0.42 * gsi - 0.18 * psy + 0.15 * oc)
        volume_deformed = volume_standard * vol_expansion
        volume_delta_pct = (volume_deformed - volume_standard) / volume_standard * 100

        willmore_standard = 2 * np.pi ** 2
        willmore_deformed = willmore_standard * (1.0 + delta * (0.85 * psdi + 0.65 * psy))

        gaussian_min = -12.45 * (1 + psy * 1.5)
        gaussian_max = (1.0 / (a * a)) * (1 + oc * 0.4)
        mean_curv_avg = (1.5 / a) * (1 + delta * 0.25)

        values = [som, oc, psy, gsi, pst, psdi]
        sum_vals = sum(values) or 1
        entropy = 0.0
        for val in values:
            p = val / sum_vals
            if p > 0.001:
                entropy -= p * np.log2(p)

        disharmony = (gsi * 0.3 + psy * 0.3 + oc * 0.2 + som * 0.2) * (1 + delta * 0.5)
        icc_index = max(8.5, min(99.0, (1.0 - disharmony * 0.65) * 100))

        if gsi < 0.35 and psy < 0.4:
            tier = "Normal"
        elif gsi < 0.60:
            tier = "Leve"
        elif gsi < 0.80:
            tier = "Moderado"
        elif gsi < 0.92:
            tier = "Severo"
        else:
            tier = "Critico"

        stability_score = max(0.0, min(100.0, 100 - (delta * 40 + gsi * 35 + psy * 25)))

        # Muestreo 28x28 de la tension topologica diferencial sobre el manifold
        sample_steps = 28
        idx = np.arange(sample_steps)
        vals = (idx / sample_steps) * 2 * np.pi
        v_grid, u_grid = np.meshgrid(vals, vals, indexing="ij")
        result = self.compute_differential_tension(u_grid, v_grid, delta, a)
        tension_grid = result["tension"]

        avg_tension = float(np.mean(tension_grid))
        max_tension = float(np.max(tension_grid))
        high_tension_pct = float(np.mean(tension_grid >= 0.65) * 100)

        return {
            "surface_area_standard": surface_area_standard,
            "surface_area_deformed": surface_area_deformed,
            "surface_area_delta_percent": surface_area_delta_pct,
            "volume_standard": volume_standard,
            "volume_deformed": volume_deformed,
            "volume_delta_percent": volume_delta_pct,
            "willmore_energy_standard": willmore_standard,
            "willmore_energy_deformed": willmore_deformed,
            "gaussian_curvature_min": gaussian_min,
            "gaussian_curvature_max": gaussian_max,
            "mean_curvature_avg": mean_curv_avg,
            "topological_entropy": entropy,
            "icc_index": icc_index,
            "clinical_severity_tier": tier,
            "stability_score": stability_score,
            "max_differential_tension": max_tension,
            "avg_differential_tension": avg_tension,
            "high_tension_area_percent": high_tension_pct,
        }

    # ------------------------------------------------------------------
    # Resumen textual (equivalente a generateModelSummaryText)
    # ------------------------------------------------------------------
    def generate_model_summary_text(self):
        m = self.compute_topological_metrics()
        d = self.scl90r_data
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        xf, yf, zf = self.fantasy_point_3d

        alert = (
            "-> ALERTA: Zona de angustia critica expandida. Ruptura de la fantasia en cercanias de la cuspide."
            if self.rupture_area_percent > 12.0
            else "-> Estructura compensada: Trayectorias S, I y Sigma delimitadas con angustia focalizada."
        )

        return f"""{'=' * 80}
          HORN TORUS ICC MODEL (TOPOLOGICAL INCONSCIENT & SCL-90-R)
{'=' * 80}
Timestamp: {ts} UTC
Manifold: Horn Torus [R = r = a, Cusp Point (0,0,0) at v = pi]
Radio a (a_scale * GSI):           {self.a:.5f}  [a_scale={self.a_scale}, GSI={d.get('GSI', 0.85):.3f}]
Escalas Angulares:                 u_scale={self.u_scale:.4f} rad, v_scale={self.v_scale:.4f} rad
Factor de Deformacion (delta):     {self.deformation_factor:.4f}
Umbral Critico de Angustia (A_cr): {self.A_cr:.4f} rad (pi / 4)

[1] TOPOLOGIA LACANIANA DEL HORN TORUS:
{'-' * 80}
  * Exterior: Cc (Consciente) - Superficie y horizonte visible desde afuera
  * Interior: Icc (Inconsciente) - Cavidad interior, vortice y cuspide de auto-tangencia
  * Cintas Entrecruzadas en el Interior:

  * S (Simbolico / Significante): u_S = {self.u_S:.4f} rad | v_S = {self.v_S:.4f} rad
    -> Funcion: Cadena significante. Ansiedad ({d.get('Ansiedad', 0.95):.2f}) + Obsesion ({d.get('Obsesion-Compulsion', 0.9):.2f})
    -> Color en Visualizador: ROJO (Crimson Ribbon)

  * I (Imaginario / Imagen del Cuerpo): u_I = {self.u_I:.4f} rad | v_I = {self.v_I:.4f} rad
    -> Funcion: Especularidad y cuerpo somatico. Somatizacion ({d.get('Somatizacion', 0.8):.2f}) + Sensibilidad ({d.get('Sensibilidad Interpersonal', 0.7):.2f})
    -> Color en Visualizador: VERDE (Emerald Ribbon)

  * Hilo Pulsional (Trieb / Vorstellungsrepraesentanz): PEGADO A I
    -> Fijacion pulsional: Fuerza de enlace somatico = {self.pulsion_attachment_strength * 100:.1f}%
    -> Funcion: Representante de la representacion pulsional enlazado al cuerpo imaginario.
    -> Color en Visualizador: DORADO / AMBAR (Golden Braid)

  * Sigma (Sintoma / Sinthome): u_Sigma = {self.u_Sigma:.4f} rad | v_Sigma = {self.v_Sigma:.4f} rad
    -> Funcion: Anudamiento y sutura estructural. Psicoticismo ({d.get('Psicoticismo', 0.9):.2f}) + Hostilidad ({d.get('Hostilidad', 0.6):.2f})
    -> Color en Visualizador: AZUL COBALTO (Cobalt Ribbon)

  * Punto de Fantasia [La Fantasia es Angustia]: ($ <> a) en (u_F, v_F) = (pi, pi/2)
    -> Coordenadas 3D (x,y,z): ({xf:.4f}, {yf:.4f}, {zf:.4f})
    -> Vortice de Angustia y limite de ruptura en el umbral A_cr = pi/4
    -> Puntos de Ruptura (A <= A_cr): {self.rupture_count} nodos ({self.rupture_area_percent:.2f}% del Manifold)

[2] VECTOR PSICOMETRICO SCL-90-R (DEROGATIS):
{'-' * 80}
  * Somatizacion (SOM):                {d.get('Somatizacion', 0.8):.3f}
  * Obsesion-Compulsion (O-C):          {d.get('Obsesion-Compulsion', 0.9):.3f}
  * Sensibilidad Interpersonal (I-S):   {d.get('Sensibilidad Interpersonal', 0.7):.3f}
  * Depresion (DEP):                    {d.get('Depresion', 0.85):.3f}
  * Ansiedad (ANX):                     {d.get('Ansiedad', 0.95):.3f}
  * Hostilidad (HOS):                   {d.get('Hostilidad', 0.6):.3f}
  * Ansiedad Fobica (PHOB):             {d.get('Ansiedad Fobica', 0.75):.3f}
  * Ideacion Paranoide (PAR):           {d.get('Ideacion Paranoide', 0.8):.3f}
  * Psicoticismo (PSY):                 {d.get('Psicoticismo', 0.9):.3f}
  * Global Severity Index (GSI):        {d.get('GSI', 0.85):.3f}
  * Positive Symptom Total (PST):       {d.get('PST', 0.7):.3f}
  * Positive Symptom Distress (PSDI):   {d.get('PSDI', 0.9):.3f}

[3] INVARIANTES TOPOLOGICOS Y ENERGETICOS:
{'-' * 80}
  * Caracteristica de Euler (chi):      0 (Toro Manifold Genero 1)
  * Area Superficial Estandar:          {m['surface_area_standard']:.4f} u^2
  * Area Superficial Deformada:         {m['surface_area_deformed']:.4f} u^2 ({'+' if m['surface_area_delta_percent'] >= 0 else ''}{m['surface_area_delta_percent']:.2f}%)
  * Volumen Encerrado Estandar:         {m['volume_standard']:.4f} u^3
  * Volumen Encerrado Deformado:        {m['volume_deformed']:.4f} u^3 ({'+' if m['volume_delta_percent'] >= 0 else ''}{m['volume_delta_percent']:.2f}%)
  * Energia de Willmore W = Int(H^2 dA): {m['willmore_energy_deformed']:.4f} (Base: {m['willmore_energy_standard']:.4f})
  * Tension Topologica Diferencial dE:  Promedio: {m['avg_differential_tension'] * 100:.1f}% | Maxima: {m['max_differential_tension'] * 100:.1f}%
  * Area de Alta Tension (tau >= 0.65): {m['high_tension_area_percent']:.1f}% del Manifold
  * Indice ICC (Coherencia Icc):        {m['icc_index']:.2f} %
  * Diagnostico Clinico Estructural:    [ {m['clinical_severity_tier'].upper()} ]
    {alert}
{'=' * 80}"""

    def print_summary(self):
        print(self.generate_model_summary_text())

    # ------------------------------------------------------------------
    # Visualizacion estatica (matplotlib)
    # ------------------------------------------------------------------
    def plot_3d(self, save_path=None, show=True, deformed=False, color_mode=None):
        """Visualiza el modelo 3D (matplotlib) con curvas S, I, Hilo Pulsional, Sigma."""
        color_mode = color_mode or ("differential_stress" if deformed else "angustia")

        x, y, z, u, v = self.horn_torus_surface(deformed=deformed)

        if color_mode == "angustia":
            c = self.calculate_angustia(u, v)
            cmap_name, label = "viridis", "Angustia A(u, v)"
        elif color_mode == "differential_stress":
            c = self.compute_differential_tension(u, v)["tension"]
            cmap_name, label = "inferno", "Tension Topologica Diferencial"
        elif color_mode == "stress":
            _factor, c = self.compute_scl_deformation(u, v)
            cmap_name, label = "plasma", "Stress SCL-90-R"
        else:
            raise ValueError(f"color_mode desconocido: {color_mode}")

        cmap = matplotlib.colormaps[cmap_name]
        norm = Normalize(vmin=float(c.min()), vmax=float(c.max()))

        fig = plt.figure(figsize=(12, 9), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, facecolors=cmap(norm(c)), alpha=0.72, edgecolor="none", antialiased=True)

        for curve in self.get_curves().values():
            ax.plot(curve["x"], curve["y"], curve["z"], color=curve["color"], linewidth=2.5, label=curve["label"])

        xf, yf, zf = self.fantasy_point_3d_visual
        ax.scatter([xf], [yf], [zf], color="magenta", s=130, edgecolors="black", label="Fantasia ($ <> a)")

        title = "Horn Torus del Icc - Deformado por Sintoma" if deformed else "Horn Torus del Icc: S, I, Hilo Pulsional, Sigma"
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, aspect=10, label=label)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"[OK] Guardado en '{save_path}'")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_deformed(self, save_path=None, show=True):
        """Atajo: visualiza el manifold deformado, coloreado por tension topologica."""
        self.plot_3d(save_path=save_path, show=show, deformed=True, color_mode="differential_stress")

    # ------------------------------------------------------------------
    # Visualizacion interactiva (Plotly) + exportaciones
    # ------------------------------------------------------------------
    def _build_plotly_figure(self, deformed=False, color_mode=None):
        if go is None:
            raise ImportError("Falta plotly. Instala con: pip install plotly")

        color_mode = color_mode or ("differential_stress" if deformed else "angustia")
        x, y, z, u, v = self.horn_torus_surface(u_resolution=80, v_resolution=80, deformed=deformed)

        if color_mode == "angustia":
            c = self.calculate_angustia(u, v)
            colorscale, label = "Viridis", "Angustia A(u, v)"
        elif color_mode == "differential_stress":
            c = self.compute_differential_tension(u, v)["tension"]
            colorscale, label = "Inferno", "Tension Topologica Diferencial"
        else:
            _factor, c = self.compute_scl_deformation(u, v)
            colorscale, label = "Plasma", "Stress SCL-90-R"

        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=c,
            colorscale=colorscale,
            opacity=0.78,
            colorbar=dict(title=label),
            name="Horn Torus",
            showscale=True,
        ))

        for curve in self.get_curves().values():
            fig.add_trace(go.Scatter3d(
                x=curve["x"], y=curve["y"], z=curve["z"],
                mode="lines",
                line=dict(color=curve["color"], width=6),
                name=curve["label"],
            ))

        xf, yf, zf = self.fantasy_point_3d_visual
        fig.add_trace(go.Scatter3d(
            x=[xf], y=[yf], z=[zf],
            mode="markers",
            marker=dict(color="magenta", size=6, line=dict(color="black", width=1)),
            name="Fantasia ($ <> a)",
        ))

        subtitle = "Modelo Deformado por Sintoma" if deformed else "Modelo Interactivo"
        fig.update_layout(
            title=f"Horn Torus del Icc (SCL-90-R) - {subtitle}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    def plot_interactive(self, deformed=False, color_mode=None):
        """Muestra el modelo 3D interactivo (Plotly): rotar/zoom con el mouse."""
        fig = self._build_plotly_figure(deformed=deformed, color_mode=color_mode)
        fig.show()
        return fig

    def save_to_html(self, path="mi_visualizacion.html", deformed=False, color_mode=None):
        """Exporta la visualizacion interactiva a un archivo HTML standalone."""
        fig = self._build_plotly_figure(deformed=deformed, color_mode=color_mode)
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"[OK] Visualizacion interactiva guardada en '{path}'")
        return path

    def export_to_json(self, path="mis_datos.json"):
        """Exporta datos SCL-90-R, coordenadas lacanianas y metricas topologicas a JSON."""
        metrics = self.compute_topological_metrics()

        payload = {
            "scl90r_data": self.scl90r_data,
            "params": {
                "a_scale": self.a_scale,
                "u_scale": self.u_scale,
                "v_scale": self.v_scale,
                "deformation_factor": self.deformation_factor,
                "a_critical": self.A_cr,
                "a": self.a,
                "effective_a": self.effective_a,
            },
            "lacanian_coordinates": {
                "u_S": self.u_S, "v_S": self.v_S,
                "u_I": self.u_I, "v_I": self.v_I,
                "u_pulsion": self.u_pulsion, "v_pulsion": self.v_pulsion,
                "pulsion_attachment_strength": self.pulsion_attachment_strength,
                "u_Sigma": self.u_Sigma, "v_Sigma": self.v_Sigma,
            },
            "fantasy_point": {
                "u_v": list(self.fantasy_point),
                "xyz": list(self.fantasy_point_3d),
            },
            "rupture": {
                "count": self.rupture_count,
                "area_percent": self.rupture_area_percent,
            },
            "topological_metrics": metrics,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[OK] Datos exportados a '{path}'")
        return path


if __name__ == "__main__":
    model = HornTorusICC(HornTorusICC.DEFAULT_SCL90R_DATA)
    model.print_summary()
    model.plot_3d(save_path="mi_modelo.png", show=False)
    model.plot_deformed(save_path="mi_modelo_deformado.png", show=False)
    model.save_to_html("mi_visualizacion.html")
    model.export_to_json("mis_datos.json")
