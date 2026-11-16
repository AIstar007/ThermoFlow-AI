"""Simple 1-D Pennes bioheat forward simulator for sensitivity education.

This is an idealized homogeneous slab model. It is NOT an inverse perfusion
solver and must not be fitted to a surface-only thermal trace as though that
uniquely identifies blood perfusion. It is provided to explore how assumed
parameters influence a modeled depth/time temperature field.
"""
from __future__ import annotations

import numpy as np


def simulate_pennes_slab(
    *,
    duration_s: float = 120.0,
    output_interval_s: float = 1.0,
    depth_mm: float = 10.0,
    n_depth_nodes: int = 41,
    initial_temperature_c: float = 32.0,
    deep_temperature_c: float = 36.5,
    arterial_temperature_c: float = 37.0,
    ambient_temperature_c: float = 22.0,
    challenge_ambient_c: float = 12.0,
    challenge_start_s: float = 20.0,
    challenge_end_s: float = 40.0,
    perfusion_s_inv: float = 0.0005,
    tissue_density_kg_m3: float = 1050.0,
    tissue_specific_heat_j_kg_k: float = 3600.0,
    tissue_conductivity_w_m_k: float = 0.45,
    blood_density_kg_m3: float = 1060.0,
    blood_specific_heat_j_kg_k: float = 3770.0,
    metabolic_heat_w_m3: float = 500.0,
    surface_heat_transfer_w_m2_k: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return time, depth, temperature[time, depth], and numerical metadata."""
    if duration_s <= 0 or output_interval_s <= 0 or depth_mm <= 0:
        raise ValueError("Duration, output interval and depth must be positive")
    if n_depth_nodes < 5:
        raise ValueError("Use at least five depth nodes")
    if perfusion_s_inv < 0 or tissue_density_kg_m3 <= 0 or tissue_specific_heat_j_kg_k <= 0:
        raise ValueError("Perfusion must be non-negative and tissue density/specific heat positive")
    if tissue_conductivity_w_m_k <= 0 or surface_heat_transfer_w_m2_k < 0:
        raise ValueError("Conductivity must be positive and surface heat-transfer coefficient non-negative")
    if not (0 <= challenge_start_s <= challenge_end_s <= duration_s):
        raise ValueError("Challenge interval must be inside the simulated duration")

    depth_m = float(depth_mm) / 1000.0
    x = np.linspace(0.0, depth_m, int(n_depth_nodes))
    dx = float(x[1] - x[0])
    rho_c = float(tissue_density_kg_m3) * float(tissue_specific_heat_j_kg_k)
    alpha = float(tissue_conductivity_w_m_k) / rho_c
    beta = float(perfusion_s_inv) * float(blood_density_kg_m3) * float(blood_specific_heat_j_kg_k) / rho_c
    reaction = beta
    # Conservative explicit-step stability bounds for diffusion, reaction and Robin surface boundary.
    dt_diff = 0.40 * dx * dx / (2.0 * alpha)
    dt_react = 0.20 / reaction if reaction > 0 else float("inf")
    dt_surface = (0.20 * rho_c * dx / (2.0 * surface_heat_transfer_w_m2_k)
                  if surface_heat_transfer_w_m2_k > 0 else float("inf"))
    dt = min(float(output_interval_s), dt_diff, dt_react, dt_surface)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Could not determine a stable model timestep")

    output_times = np.arange(0.0, duration_s + output_interval_s * 0.25, output_interval_s)
    temperature = np.full(int(n_depth_nodes), float(initial_temperature_c), dtype=float)
    temperature[-1] = float(deep_temperature_c)
    snapshots = np.empty((len(output_times), len(x)), dtype=np.float32)
    snapshots[0] = temperature.astype(np.float32)
    current_t = 0.0
    for out_idx in range(1, len(output_times)):
        target_t = float(output_times[out_idx])
        while current_t < target_t - 1e-10:
            step = min(dt, target_t - current_t)
            ambient = (float(challenge_ambient_c)
                       if challenge_start_s <= current_t < challenge_end_s
                       else float(ambient_temperature_c))
            old = temperature
            new = old.copy()
            lap = (old[2:] - 2.0 * old[1:-1] + old[:-2]) / (dx * dx)
            new[1:-1] = old[1:-1] + step * (
                alpha * lap + beta * (float(arterial_temperature_c) - old[1:-1])
                + float(metabolic_heat_w_m3) / rho_c
            )
            # Robin convective boundary at the skin surface, with a ghost-node approximation.
            lap_surface = (2.0 * (old[1] - old[0]) / (dx * dx)
                           - 2.0 * surface_heat_transfer_w_m2_k
                           * (old[0] - ambient) / (float(tissue_conductivity_w_m_k) * dx))
            new[0] = old[0] + step * (
                alpha * lap_surface + beta * (float(arterial_temperature_c) - old[0])
                + float(metabolic_heat_w_m3) / rho_c
            )
            new[-1] = float(deep_temperature_c)
            temperature = new
            current_t += step
        snapshots[out_idx] = temperature.astype(np.float32)
    metadata = {
        "model": "1-D homogeneous Pennes bioheat forward model",
        "perfusion_parameter_s_inv": float(perfusion_s_inv),
        "thermal_diffusivity_m2_s": float(alpha),
        "internal_dt_s": float(dt),
        "depth_mm": float(depth_mm),
        "n_depth_nodes": int(n_depth_nodes),
        "challenge_interval_s": [float(challenge_start_s), float(challenge_end_s)],
        "boundary_condition": "convective Robin at skin surface; fixed deep temperature",
        "warning": "Idealized forward simulation only. It is not fitted to participant data and does not estimate actual perfusion from surface thermography.",
    }
    return output_times, x * 1000.0, snapshots, metadata
