import numpy as np


def binder_from_uv(u, v):
    """Map unconstrained positives u,v to binder fractions that sum to 1."""
    cement = u / (u + v + 1)
    glass = v / (u + v + 1)
    cha = 1.0 / (u + v + 1)
    return cement, glass, cha


def _zero_small_negative(x, tol=1e-8):
    """If x is a small negative number (>-tol and <0), return 0.0, else return x."""
    try:
        if x < 0 and abs(x) < tol:
            return 0.0
    except Exception:
        pass
    return x


def _zero_small(x, tol=1e-6):
    """If x is very close to zero (abs(x) < tol) return 0.0 else return x."""
    try:
        if abs(x) < tol:
            return 0.0
    except Exception:
        pass
    return x


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


def evaluate_mix_from_vars(x, aggregate_size_mm=None, glass_ratio_norm=None, cha_ratio_norm=None):
    """Evaluate a mix given decision variables.

    Supports two interfaces for backward compatibility:
    1) x = [u, v, aggregate_size_mm, wbr] (original): keeps previous behaviour.
    2) x = [S, wbr] with fixed parameters provided:
         - aggregate_size_mm (required)
         - glass_ratio_norm (required)
         - cha_ratio_norm (required)

    In mode (2) binder fractions are computed as:
        cement_frac = 1.0 - S
        glass_frac  = S * glass_ratio_norm
        cha_frac    = S * cha_ratio_norm
    ensuring cement+glass+cha = 1.0.
    """
    # Mode detection
    x = list(x)
    if len(x) == 4 and aggregate_size_mm is None:
        # legacy mode: x = [u, v, aggregate_size_mm, wbr]
        u, v, aggregate_size_mm, wbr = x
        cement, glass, cha = binder_from_uv(u, v)
    elif len(x) == 2 and (aggregate_size_mm is not None and glass_ratio_norm is not None and cha_ratio_norm is not None):
        S, wbr = x
        # normalize S into [0,1]
        S = float(min(max(S, 0.0), 1.0))
        cement = 1.0 - S
        glass = S * float(glass_ratio_norm)
        cha = S * float(cha_ratio_norm)
    else:
        raise ValueError("evaluate_mix_from_vars: unexpected inputs. Provide either legacy x (len=4) or x=[S,wbr] with fixed params.")

    substitution = float(glass + cha)

    # ensure aggregate_size_mm and wbr are available in both modes
    if len(x) == 4 and 'wbr' in locals():
        pass
    elif len(x) == 2:
        # wbr variable defined earlier
        pass

    # retrieve wbr (in legacy branch, it was assigned; in new mode it's x[1])
    if len(x) == 4:
        wbr = float(wbr)
    else:
        wbr = float(x[1])

    void = predict_void_ratio(aggregate_size_mm, cement, wbr)

    # Strength uses explicit substitution sensitivity internally
    strength = predict_strength(cement, glass, cha, void)

    # Permeability: same regression approach as before
    p_slope = 0.00324357
    p_intercept = 2.76825951

    substitution_pct = 100.0 * substitution

    # Small global scale factor preserved
    scale = 0.7570254386518729

    k_reg = p_intercept + p_slope * substitution_pct
    k_scaled = scale * k_reg

    size_factor = aggregate_size_mm / 20.0
    paste_content = 0.22
    geom_term = 3.0 * void * size_factor * (1.0 - paste_content)

    permeability = float(k_scaled + geom_term)

    co2 = estimate_co2(cement, glass, cha)
    # avoid tiny negative artefacts from floating-point ops
    cement = float(_zero_small_negative(cement))
    glass = float(_zero_small_negative(glass))
    cha = float(_zero_small_negative(cha))
    substitution = float(_zero_small_negative(substitution))
    void = float(_zero_small_negative(void))
    strength = float(_zero_small_negative(strength))
    permeability = float(_zero_small_negative(permeability))
    co2 = float(_zero_small_negative(co2))

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


def compute_cost_from_mix(mix, *,
                          aggregate_mass_per_unit=3.0,
                          unit_prices=None):
    """Compute material cost breakdown per GLONUT unit using GLONUT assumptions.

    Methodology (GLONUT integration):
    - One GLONUT is the base unit (not 1 m^3). Binder mass per GLONUT = 1.0 kg.
    - Base GLONUT masses (for reference):
        Water: 0.2 kg
        Cement: 0.6 kg
        Gravel: 3.0 kg (fixed)
        CHA: 0.2 kg
        Glass: 0.2 kg
    - When using optimized binder fractions (cement_frac, glass_frac, cha_frac),
      scale cement/glass/CHA so their masses sum to the binder mass per GLONUT (1.0 kg).
    - Water mass per GLONUT = wbr * binder_mass_per_glonut.
    - Cost per m² = cost_per_glonut * 14.8.

    Returns (rows, totals) where rows is a list of dicts with keys:
      'Material', 'Volume (kg per GLONUT)', 'Unit price (Rp/kg)', 'Subtotal (Rp per GLONUT)'
    """
    if unit_prices is None:
        unit_prices = {
            "water": 4000.0,
            "cement": 1500.0,
            "gravel": 2000.0,
            "cha": 500.0,
            "glass": 500.0,
        }

    # mix is expected to contain keys: cement_frac, glass_frac, cha_frac, wbr
    cement_frac = float(mix.get("cement_frac", 0.0))
    glass_frac = float(mix.get("glass_frac", 0.0))
    cha_frac = float(mix.get("cha_frac", 0.0))
    wbr = float(mix.get("wbr", 0.0))

    binder_mass_per_glonut = 1.0  # kg per GLONUT (by definition)
    cement_mass = cement_frac * binder_mass_per_glonut
    glass_mass = glass_frac * binder_mass_per_glonut
    cha_mass = cha_frac * binder_mass_per_glonut
    aggregate_mass = float(aggregate_mass_per_unit)  # fixed at 3.0 kg per GLONUT
    water_mass = wbr * binder_mass_per_glonut

    rows = []

    # clamp tiny negative masses and very small positives
    cement_mass = float(_zero_small(_zero_small_negative(cement_mass)))
    glass_mass = float(_zero_small(_zero_small_negative(glass_mass)))
    cha_mass = float(_zero_small(_zero_small_negative(cha_mass)))
    water_mass = float(_zero_small(_zero_small_negative(water_mass)))

    def add_row(name, mass, price_key, vol_label="Volume (kg per GLONUT)", sub_label="Subtotal (Rp per GLONUT)"):
        mass = float(_zero_small(_zero_small_negative(mass)))
        price = unit_prices.get(price_key, 0.0)
        subtotal = mass * price
        subtotal = float(_zero_small(_zero_small_negative(subtotal)))
        rows.append({
            "Material": name,
            vol_label: round(float(mass), 4),
            "Unit price (Rp/kg)": int(price),
            sub_label: int(round(subtotal)),
        })

    add_row("Cement", cement_mass, "cement")
    add_row("Crushed glass", glass_mass, "glass")
    add_row("Coconut shell ash (CHA)", cha_mass, "cha")
    add_row("Gravel (aggregate)", aggregate_mass, "gravel")
    add_row("Water", water_mass, "water")

    total_per_glonut = sum(r[next(k for k in r.keys() if k.startswith("Subtotal"))] for r in rows)

    cost_per_m2 = total_per_glonut * 14.8

    totals = {
        "total_per_glonut": int(round(total_per_glonut)),
        "total_per_m2": int(round(cost_per_m2)),
        "binder_mass_per_glonut": binder_mass_per_glonut,
    }

    return rows, totals
