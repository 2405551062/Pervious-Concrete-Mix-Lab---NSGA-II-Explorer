import streamlit as st
import pandas as pd
import plotly.express as px

from optimizer import run_nsga2
from models import compute_cost_from_mix


st.set_page_config(page_title="Pervious Concrete Mix Lab", layout="wide")

st.title("GLONUT Pervious Concrete Mix Lab — Simple Explorer")

st.markdown("Use the Run button to generate recommended mixes. Expand 'Advanced settings' for expert options.")

with st.sidebar:
    st.header("Run NSGA-II")
    # sensible defaults; advanced settings hidden by default
    pop_size = 80
    n_gen = 60
    seed = 0
    st.subheader("Mix options")
    agg_choice = st.selectbox("Aggregate size", ["Small (10 mm)", "Medium (14 mm)", "Large (20 mm)"], index=1)
    agg_map = {"Small (10 mm)": 10.0, "Medium (14 mm)": 14.0, "Large (20 mm)": 20.0}
    aggregate_size_mm_sel = agg_map.get(agg_choice, 14.0)

    st.markdown("**Glass : CHA ratio**")
    glass_ratio_input = st.number_input("Glass ratio (relative)", min_value=10, max_value=100, value=70, step=1)
    cha_ratio_input = st.number_input("CHA ratio (relative)", min_value=10, max_value=100, value=30, step=1)

    with st.expander("Advanced settings"):
        pop_size = st.slider("Population size", 20, 200, 80, step=10)
        n_gen = st.slider("Generations", 5, 200, 60, step=5)
        seed = st.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
    run_button = st.button("Run optimizer")

if run_button:
    with st.spinner("Running NSGA-II — please wait..."):
        seed_arg = None if seed == 0 else int(seed)
        # normalize glass/cha ratio to fractions
        gr = float(glass_ratio_input)
        cr = float(cha_ratio_input)
        glass_norm = gr / (gr + cr)
        cha_norm = cr / (gr + cr)
        res = run_nsga2(pop_size=pop_size, n_gen=n_gen, seed=seed_arg,
                        aggregate_size_mm=aggregate_size_mm_sel,
                        glass_ratio_norm=glass_norm,
                        cha_ratio_norm=cha_norm)
    st.session_state["pareto_df"] = pd.DataFrame(res["details"])
    # store the parameters used to generate these results
    st.session_state["pareto_params"] = {
        "aggregate_size_mm": aggregate_size_mm_sel,
        "glass_norm": glass_norm,
        "cha_norm": cha_norm,
        "pop_size": pop_size,
        "n_gen": n_gen,
    }
    # reset selected index to top
    st.session_state["selected_mix_idx"] = 0

if "pareto_df" in st.session_state:
    df = st.session_state["pareto_df"]
else:
    df = None


if df is None:
    st.info("Click 'Run optimizer' in the sidebar to generate Pareto-optimal mixes.")
else:
    # human-friendly column names
    rename_map = {
        "aggregate_size_mm": "Stone size (mm)",
        "wbr": "Mix wetness",
        "cement_frac": "Cement fraction",
        "glass_frac": "Glass fraction",
        "cha_frac": "CHA fraction",
        "void_ratio": "Void ratio",
        "strength_mpa": "Strength (MPa)",
        "permeability_mms": "Permeability (mm/s)",
        "co2_kg_per_m3": "CO2 (kg/m3)",
    }

    st.subheader("Pareto front — strength vs permeability")
    # show which parameters were used to compute the displayed solutions
    pareto_params = st.session_state.get("pareto_params")
    if pareto_params is not None:
        used_agg = pareto_params.get("aggregate_size_mm")
        used_gn = pareto_params.get("glass_norm")
        used_cn = pareto_params.get("cha_norm")
        if (used_agg != aggregate_size_mm_sel) or (abs(used_gn - (glass_ratio_input/(glass_ratio_input+cha_ratio_input))) > 1e-6):
            st.warning(f"Displayed solutions were computed with aggregate size {used_agg} mm and waste split {int(round(100*used_gn))}% glass / {int(round(100*used_cn))}% CHA. Change inputs and click Run to regenerate results.")
        else:
            st.info(f"Solutions computed with aggregate size {used_agg} mm and waste split {int(round(100*used_gn))}% glass / {int(round(100*used_cn))}% CHA.")
    # sanitize dataframe for display: remove tiny floats and round
    def _sanitize_df_for_display(df):
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                # Replace very small values near zero with 0.0
                def _clean(v):
                    v = float(v)
                    if abs(v) < 1e-9:
                        return 0.0
                    # if small negative, bump to zero
                    if v < 0 and abs(v) < 1e-3:
                        return 0.0
                    return v

                d[c] = d[c].apply(_clean)
                # rounding rules
                if any(k in c for k in ["cement_frac", "glass_frac", "cha_frac", "substitution", "wbr", "void_ratio"]):
                    d[c] = d[c].clip(lower=0.0).round(4)
                elif any(k in c for k in ["strength_mpa", "permeability_mms", "co2_kg_per_m3"]) or any(k in c for k in ["strength", "permeability", "co2"]):
                    # clip strength/permeability/CO2 to non-negative for display and round
                    d[c] = d[c].apply(lambda v: 0.0 if v < 0 and abs(v) < 1e-3 else v)
                    # ensure strength isn't negative for display
                    if "strength" in c:
                        d[c] = d[c].clip(lower=0.0)
                    d[c] = d[c].round(3)
                elif "aggregate_size" in c:
                    d[c] = d[c].round(1)
                else:
                    d[c] = d[c].round(4)
        return d

    df_display_full = _sanitize_df_for_display(df)
    # show user selections
    gr = float(glass_ratio_input)
    cr = float(cha_ratio_input)
    glass_norm = gr / (gr + cr)
    cha_norm = cr / (gr + cr)
    st.markdown(f"**Selected aggregate size:** {agg_choice} — {aggregate_size_mm_sel} mm")
    st.markdown(f"**Waste split:** {int(round(100*glass_norm))}% glass, {int(round(100*cha_norm))}% CHA")
    fig = px.scatter(
        df,
        x="strength_mpa",
        y="permeability_mms",
        color="co2_kg_per_m3",
        hover_data=["cement_frac", "glass_frac", "cha_frac", "wbr", "void_ratio"],
        color_continuous_scale="Viridis",
        title="Pareto Set — strength vs permeability (color = CO2)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Solutions (table)")
    # sort the dataframe for display and use the same sorted frame for selection
    df_sorted = df_display_full.sort_values(by=["strength_mpa"], ascending=False).reset_index(drop=True)
    # drop stone size column from the displayed table (it's constant for this run)
    display_df = df_sorted.drop(columns=["aggregate_size_mm"]).rename(columns=rename_map).copy()
    st.dataframe(display_df)

    # Selection: exactly one mix — use the same sorted dataframe so selection indexes match the table
    df_display = df_sorted
    options_idx = list(range(len(df_display)))
    default_idx_raw = st.session_state.get("selected_mix_idx", 0)
    try:
        default_idx = int(default_idx_raw)
    except Exception:
        default_idx = 0
    def fmt(i):
        row = df_display.loc[i]
        return f"{i}: Strength {row['strength_mpa']:.1f} MPa — Permeability {row['permeability_mms']:.2f} mm/s"

    sel_index = st.selectbox(
        "Select one recommended mix to inspect and estimate cost",
        options_idx,
        index=default_idx if 0 <= default_idx < len(options_idx) else 0,
        format_func=fmt,
        key="selected_mix_idx",
    )
    selected_mix = df_display.loc[sel_index].to_dict()

    st.markdown("**Selected mix details**")
    # show selected mix properties in readable form with rounded numbers
    formatted = {}
    for k, v in selected_mix.items():
        label = rename_map.get(k, k)
        if isinstance(v, float):
            if abs(v) < 1e-6:
                v2 = 0.0
            elif k == "strength_mpa":
                v2 = 0.0 if v < 0 else round(v, 3)
            elif k in ("cement_frac", "glass_frac", "cha_frac", "substitution", "wbr", "void_ratio"):
                v2 = round(v, 4)
            elif k in ("permeability_mms", "co2_kg_per_m3"):
                v2 = round(v, 3)
            else:
                v2 = round(v, 4)
        else:
            v2 = v
        formatted[label] = v2
    selected_display = pd.DataFrame([formatted])
    st.table(selected_display)

    # Compute cost breakdown
    rows, totals = compute_cost_from_mix(selected_mix)
    cost_df = pd.DataFrame(rows)

    st.subheader("Estimated Cost Breakdown (per GLONUT)")
    # tidy cost table: round volumes and replace tiny values with 0
    cost_df_display = cost_df.copy()
    vol_col = next((c for c in cost_df_display.columns if c.lower().startswith("volume")), None)
    if vol_col is not None:
        cost_df_display[vol_col] = cost_df_display[vol_col].apply(lambda x: 0.0 if abs(float(x)) < 1e-6 else round(float(x), 4))
    st.table(cost_df_display)

    st.markdown(f"**Total cost per GLONUT:** Rp {totals.get('total_per_glonut', 0):,}")
    st.markdown(f"**Total estimated cost per m² (×14.8):** Rp {totals.get('total_per_m2', 0):,}")
    st.caption("Results are intended for comparison and educational use.")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="pareto_solutions.csv", mime="text/csv")
