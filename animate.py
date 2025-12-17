import pandas as pd

df = pd.read_csv("tic_tac_toe_scoreboard.csv")

df["A_score"] = (df["winner"] == "A").cumsum()
df["B_score"] = (df["winner"] == "B").cumsum()

score_df = df[["game", "A_score", "B_score"]].melt(
    id_vars="game",
    value_vars=["A_score", "B_score"],
    var_name="Player",
    value_name="Score"
)


import plotly.express as px
import streamlit as st

fig = px.line(
    score_df,
    x="game",
    y="Score",
    color="Player",
    animation_frame="game",
    markers=True,
    range_y=[0, score_df["Score"].max() + 1],
    title="LLM vs LLM — Score Progression by Game"
)

fig.update_layout(
    transition={"duration": 500},
    xaxis_title="Game #",
    yaxis_title="Cumulative Wins"
)

st.plotly_chart(fig, use_container_width=True)
