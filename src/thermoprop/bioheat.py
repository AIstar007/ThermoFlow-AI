"""Optional paper-derived heat-balance index; never a direct perfusion measurement."""
from __future__ import annotations

import numpy as np


def wu_style_model_index(
    skin_temperature_c: np.ndarray,
    ambient_temperature_c: float,
    arterial_temperature_c: float = 39.0,
    emissivity: float = 0.98,
    exchange_ratio: float = 0.8,
    blood_specific_heat: float = 3780.0,
    skin_conductivity: float = 0.5,
    air_conductivity: float = 0.024,
    metabolic_heat_flux: float = 4.186,
    prandtl: float = 0.72,
    air_kinematic_viscosity: float = 1.56e-5,
    air_expansion: float = 3.354e-3,
    gravity: float = 9.8,
    geometry_a: float = 0.27,
    geometry_m: float = 0.25,
    characteristic_length_m: float = 0.170,
    core_to_skin_distance_m: float = 0.085,
) -> np.ndarray:
    """Implement Eq. (10) in Aristizabal-Tique et al. (2023) literally.

    Output has the paper's model units kg/(m^2 s). The model assumes a controlled
    environment, negligible evaporation, fixed tissue/air parameters, and a fixed
    arterial/core temperature. Pixels with invalid denominator or skin <= ambient
    are masked; no clipping is applied to the resulting values.
    """
    ts_c = np.asarray(skin_temperature_c, dtype=np.float64)
    te_c = float(ambient_temperature_c)
    ts = ts_c + 273.15
    te = te_c + 273.15
    ta = float(arterial_temperature_c) + 273.15
    sigma = 5.67e-8

    # The paper's natural-convection term uses a fractional power (M + 1 = 1.25).
    # Restrict the calculation to physically compatible skin-above-air pixels.
    delta_air = ts - te
    conv = air_conductivity * geometry_a * (characteristic_length_m ** (3 * geometry_m - 1))
    conv *= ((prandtl * gravity * air_expansion) / (air_kinematic_viscosity ** 2)) ** geometry_m
    conv *= np.maximum(delta_air, 0.0) ** (geometry_m + 1)

    numerator = (
        emissivity * sigma * (ts ** 4 - te ** 4)
        + conv
        + skin_conductivity * (ta - ts) / core_to_skin_distance_m
        - metabolic_heat_flux
    )
    denominator = exchange_ratio * blood_specific_heat * (ta - ts)
    valid = np.isfinite(ts) & (ts > te) & (np.abs(denominator) > 1e-6)
    result = np.full(ts.shape, np.nan, dtype=np.float64)
    result[valid] = numerator[valid] / denominator[valid]
    return result.astype(np.float32)
