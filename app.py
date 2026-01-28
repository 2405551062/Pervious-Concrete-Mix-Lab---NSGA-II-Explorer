import streamlit as st
import pandas as pd
import plotly.express as px

from optimizer import run_nsga2


st.set_page_config(page_title="Pervious Concrete Mix Lab", layout="wide")

st.title("Pervious Concrete Mix Lab — NSGA-II Explorer")

with st.sidebar:
    st.header("Run NSGA-II")
    pop_size = st.slider("Population size", 20, 200, 80, step=10)
    n_gen = st.slider("Generations", 5, 200, 60, step=5)
    seed = st.number_input("Random seed (0 = random)", min_value=0, value=0, step=1)
    run_button = st.button("Run optimizer")

st.markdown("Use the controls to run the optimizer and explore trade-offs between strength, permeability and CO2.")

if run_button:
    with st.spinner("Running NSGA-II — this may take a little while..."):
        seed_arg = None if seed == 0 else int(seed)
        res = run_nsga2(pop_size=pop_size, n_gen=n_gen, seed=seed_arg)

    df = pd.DataFrame(res["details"])
    # show Pareto scatter
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
    st.dataframe(df.sort_values(by=["strength_mpa"], ascending=False).reset_index(drop=True))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="pareto_solutions.csv", mime="text/csv")

else:
    st.info("Set run parameters in the sidebar and click 'Run optimizer' to generate Pareto-optimal mixes.")
