import numpy as np


def binder_from_uv(u, v):
    """Map unconstrained positives u,v to binder fractions that sum to 1."""
    cement = u / (u + v + 1)
    glass = v / (u + v + 1)
    cha = 1.0 / (u + v + 1)
    return cement, glass, cha


def predict_void_ratio(aggregate_size_mm, cement_frac, wbr, aggregate_content=0.78):
    """Simple empirical void-ratio estimator (dimensionless).
    Keeps outputs in realistic range ~0.12-0.40
    """
    size_factor = np.clip(aggregate_size_mm / 20.0, 0.1, 1.0)
    paste = 1.0 - aggregate_content
    void = 0.35 - 0.12 * size_factor - 0.08 * cement_frac + 0.5 * (wbr - 0.06)
    return float(np.clip(void, 0.08, 0.45))


def predict_strength(cement_frac, glass_frac, cha_frac, void_ratio, curing_factor=1.0):
    """Predict compressive strength in MPa (simple linear surrogate).
    Coefficients are illustrative and tuned for plausible ranges.
    """
    # Coefficient meanings (units: MPa per unit fraction or MPa per unit void ratio):
    #  - cement_eff: strength contribution of cement per binder fraction (cement is highly reactive)
    #  - glass_eff: strength contribution of glass (pozzolanic, lower early reactivity than cement)
    #  - cha_eff: strength contribution of CHA (lowest contribution here)
    #  - void_penalty: dominant reduction in strength per unit increase in void ratio
    # --- Fitted substitution → strength relation (from Results.csv) ---
    # Linear fit performed on experimental Results.csv:
    #   strength_mpa ≈ s_intercept + s_slope * substitution_pct
    # where substitution_pct = 100*(glass_frac + cha_frac)
    # Coefficients extracted by fitting: (s_slope, s_intercept)
    s_slope = -0.36950210
    s_intercept = 38.9819385

    # Compute substitution in percent
    substitution_pct = 100.0 * (glass_frac + cha_frac)

    # Base strength from the regression (reproduces experimental mapping)
    strength_base = s_intercept + s_slope * substitution_pct

    # Add a simple void-ratio correction (physically: more voids → lower strength).
    # This term was NOT in the substitution-only regression; we add it as an
    # interpretable correction: strength = strength_base - c_void * (void_ratio - mean_void)
    # where mean_void was computed from the dataset (≈0.18255). c_void chosen so
    # that a one-sigma increase in void_ratio (~0.0226) changes strength by ~1 MPa.
    mean_void = 0.18255
    c_void = 44.0
    strength = (strength_base - c_void * (void_ratio - mean_void)) * curing_factor

    return float(strength)


def predict_permeability(void_ratio, aggregate_size_mm, paste_content=0.22):
    """Predict permeability in mm/s.
    Rough surrogate: increases with void ratio and aggregate size, decreases with paste content.
    """
    # We'll use a simple, interpretable model that begins with a substitution-based
    # regression (fit to Results.csv) and then adds a small geometric dependence.

    # Linear fit from Results.csv: k ≈ p_intercept + p_slope * substitution_pct
    p_slope = 0.00324357
    p_intercept = 2.76825951

    # conversion: aggregate-size geometric factor (20 mm reference)
    size_factor = aggregate_size_mm / 20.0

    # Without changing the regression mapping, include a modest geometric modifier
    # tied to aggregate size and paste content (physically: larger aggregates and
    # lower paste content increase connectivity). This modifier is multiplicative
    # on the regression baseline and keeps the model simple/interpretable.
    geom_modifier = size_factor * (1.0 - paste_content)

    # substitution_pct will be supplied by the caller via evaluate_mix_from_vars
    # (we keep this function signature unchanged). For stand-alone calls, users
    # should compute substitution_pct externally as 100*(glass+cha).
    # Here we return the regression baseline scaled by geom_modifier when used.

    # We'll return a baseline value (caller multiplies by connectivity boost if needed).
    k_base = p_intercept + p_slope * 0.0  # placeholder if no substitution provided
    # To preserve the simple signature, this function will be used with a base
    # value (see evaluate_mix_from_vars where substitution connectivity is applied).
    # We include the geometric modifier as a small additive contribution proportional
    # to void ratio to preserve the physical trend (more voids → higher k).
    k = k_base + 8.0 * void_ratio * geom_modifier
    return float(k)


def estimate_co2(cement_frac, glass_frac, cha_frac, binder_mass_per_m3=350.0):
    """Estimate CO2 kg-eq per m3 for binder portion only (illustrative).
    Typical emissions: cement ~0.9 kgCO2/kg, glass low, CHA low.
    """
    cement_mass = cement_frac * binder_mass_per_m3
    glass_mass = glass_frac * binder_mass_per_m3
    cha_mass = cha_frac * binder_mass_per_m3
    co2 = 0.9 * cement_mass + 0.05 * glass_mass + 0.02 * cha_mass
    return float(max(0.0, co2))


def evaluate_mix_from_vars(x):
    """Given decision vector x = [u, v, aggregate_size_mm, wbr]
    return dict with cement/glass/cha fractions and predicted objectives.
    """
    u, v, aggregate_size_mm, wbr = x
    cement, glass, cha = binder_from_uv(u, v)
    substitution = float(glass + cha)

    void = predict_void_ratio(aggregate_size_mm, cement, wbr)

    # Strength uses explicit substitution sensitivity internally
    strength = predict_strength(cement, glass, cha, void)

    # Permeability: use substitution → permeability regression extracted from Results.csv
    # Linear fit: k = p_intercept + p_slope * substitution_pct
    p_slope = 0.00324357
    p_intercept = 2.76825951

    substitution_pct = 100.0 * substitution

    # Small global scale factor applied so that mid-range substitution (≈42.5%)
    # reproduces the lab mid-target (~2.2 mm/s). This scale was computed as:
    #   scale = target_mid / (p_intercept + p_slope * 42.5)
    scale = 0.7570254386518729

    # Regression baseline (from data) scaled to align with lab mid-range
    k_reg = p_intercept + p_slope * substitution_pct
    k_scaled = scale * k_reg

    # Geometric/void correction: larger aggregates and higher void ratio increase k
    size_factor = aggregate_size_mm / 20.0
    paste_content = 0.22
    geom_term = 3.0 * void * size_factor * (1.0 - paste_content)

    permeability = float(k_scaled + geom_term)

    co2 = estimate_co2(cement, glass, cha)
    return {
        "cement_frac": cement,
        "glass_frac": glass,
        "cha_frac": cha,
        "substitution": substitution,
        "aggregate_size_mm": aggregate_size_mm,
        "wbr": wbr,
        "void_ratio": void,
        "strength_mpa": strength,
        "permeability_mms": permeability,
        "co2_kg_per_m3": co2,
    }
