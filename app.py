"""Streamlit presentation layer. All financial logic lives in src/."""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline import run_for_client
from src.product_library import load_risk_library
from src.payoff import payoff_profile
from src.term_sheet import product_terms,build_markdown_term_sheet

st.set_page_config(page_title="ML-Assisted Exotic Structuring & Product Design",layout="wide",page_icon="📈")
st.title("ML-Assisted Exotic Structuring & Product Design")
st.caption(
    "Transforms client mandates into tailored equity-derivatives structures through "
    "ML-assisted product recommendation, structured-product design, pricing, "
    "risk analysis and risk-adjusted selection.")
clients=pd.read_csv("data/raw/client_requests.csv")
client_id=st.sidebar.selectbox("Client",clients["client_id"].tolist())
paths=st.sidebar.select_slider("Monte Carlo paths",options=[50000,100000],value=50000)
with st.spinner("Running structuring pipeline..."):
    mandate,market,recs,candidates,top3,avg,ranking_meta=run_for_client(client_id,use_yfinance=True,mc_paths=paths)

st.subheader("1. Client mandate")
c1,c2=st.columns([1.2,1])
with c1: st.write(mandate.request_raw)
with c2:
    st.json({
        "Underlying": mandate.underlyings[0] if mandate.underlyings else "Not specified",
        "Theme": mandate.theme or "Not specified",
        "Directional view": mandate.directional_view,
        "Target return": None if mandate.target_return is None else f"{100*mandate.target_return:.2f}%",
        "Return type": mandate.return_type,
        "Maturity": f"{mandate.maturity_years:.2f}Y" if mandate.maturity_years is not None else "Not specified",
        "Notional": f"{mandate.currency} {mandate.notional:,.0f}" if mandate.notional is not None else "Not specified",
        "Currency": mandate.currency,
        "Downside tolerance": None if mandate.downside_tolerance is None else f"{100*mandate.downside_tolerance:.1f}%",
        "Capital protection": None if mandate.capital_protection is None else f"{100*mandate.capital_protection:.1f}%",
        "Income preference": mandate.income_preference,
        "Autocall acceptance": mandate.autocall_acceptance,
        "Risk appetite": mandate.risk_appetite,
        "Complexity tolerance": mandate.complexity_tolerance,
        "Explicit product": mandate.requested_product_family or "Not specified",
        "Memory requested": mandate.memory_requested,
    })

st.subheader("2. Random Forest product recommendation — Top 5")
st.dataframe(pd.DataFrame([{"Product":r.product_family,"Recommendation Score":f"{100*r.score:.1f}%"} for r in recs]),use_container_width=True,hide_index=True)
st.caption("Recommendation Score = RandomForest predict_proba for each supported single-asset product class. No extra semantic-score weighting.")

st.subheader("3. Final ranking")
st.caption("Final Score = equal-weight average of four min-max normalized criteria: Recommendation Score, Sharpe Ratio, inverted ES95, and inverted Probability of Loss.")
rows=[]
for i,c in enumerate(top3):
    meta=ranking_meta.get(c.recommendation.product_family,{})
    rows.append({"Rank":i+1,"Product":c.recommendation.product_family,"Final Score":f"{100*meta.get('combined_score',0):.1f}%","Recommendation Score":f"{100*c.recommendation.score:.1f}%","Sharpe":f"{c.outcome.sharpe_ratio:.3f}","Expected Shortfall 95%":f"{100*c.outcome.expected_shortfall_95:.2f}%","Probability of Loss":f"{100*c.outcome.probability_of_loss:.2f}%","Expected Return":f"{100*c.outcome.expected_return:.2f}%","Fair Value":f"{c.pricing.fair_value:.2f}","Solved Term":c.pricing.solved_term_name or "—","Solved Value":f"{100*c.pricing.solved_term_value:.2f}%" if c.pricing.solved_term_value is not None else "—"})
st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

if top3:
    selected=st.selectbox("Inspect final product",[c.recommendation.product_family for c in top3])
    c=next(x for x in top3 if x.recommendation.product_family==selected); meta=ranking_meta.get(selected,{})
    title_col,button_col=st.columns([5,1])
    with title_col: st.subheader(f"4. {selected} — Quantitative & business view")
    with button_col:
        st.write(""); st.write(""); show_terms=st.button("View Product Terms",use_container_width=True)
    if show_terms:
        with st.container(border=True):
            st.markdown("### Product Terms"); st.dataframe(pd.DataFrame(product_terms(c.spec,c.pricing),columns=["Term","Value"]),use_container_width=True,hide_index=True)
    q1,q2,q3,q4,q5=st.columns(5)
    q1.metric("Final Score",f"{100*meta.get('combined_score',0):.1f}%")
    q2.metric("Recommendation",f"{100*c.recommendation.score:.1f}%")
    q3.metric("Sharpe",f"{c.outcome.sharpe_ratio:.3f}")
    q4.metric("ES 95%",f"{100*c.outcome.expected_shortfall_95:.2f}%",delta_color="inverse")
    q5.metric("Probability of Loss",f"{100*c.outcome.probability_of_loss:.2f}%",delta_color="inverse")
    left,right=st.columns(2)
    with left:
        st.markdown("#### Current local sensitivities")
        st.write({"Current Delta":round(c.risk.delta,4),"Current Gamma":round(c.risk.gamma,4),"Current Vega":round(c.risk.vega,4),"Greek Method":c.risk.method,"Fair Value":round(c.pricing.fair_value,2),"Pricing Method":c.pricing.pricing_method})
    with right:
        st.markdown("### Business interpretation"); risks=load_risk_library(); rr=risks[risks["product_family"]==c.spec.product_family]
        for _,r in rr.iterrows(): st.markdown(f"**{r.risk_name}** — {r.description}")
        if rr.empty: st.info("No static business-risk description mapped yet for this family.")

    st.subheader("5. Payoff & risk profiles")
    payoff_rows,barriers=payoff_profile(c.spec,c.pricing); payoff_df=pd.DataFrame(payoff_rows); prof=pd.DataFrame(c.risk.spot_profile)
    def add_barriers(fig):
        for b in barriers: fig.add_vline(x=b["x"],line_dash="dash",annotation_text=b["label"],annotation_position="top")
        return fig
    tab0,tab1,tab2,tab3,tab4=st.tabs(["Payoff","Price vs Spot","Delta vs Spot","Gamma vs Spot","Vega vs Spot"])
    with tab0: st.plotly_chart(add_barriers(px.line(payoff_df,x="spot_pct",y="payoff_pct",labels={"spot_pct":"Terminal Spot (% of inception spot)","payoff_pct":"Payoff (% of notional)"})),use_container_width=True)
    with tab1: st.plotly_chart(add_barriers(px.line(prof,x="spot_pct",y="price",markers=True)),use_container_width=True)
    with tab2: st.plotly_chart(add_barriers(px.line(prof,x="spot_pct",y="delta",markers=True)),use_container_width=True)
    with tab3: st.plotly_chart(add_barriers(px.line(prof,x="spot_pct",y="gamma",markers=True)),use_container_width=True)
    with tab4: st.plotly_chart(add_barriers(px.line(prof,x="spot_pct",y="vega",markers=True)),use_container_width=True)

    st.subheader("6. Preset stress scenarios")
    st.dataframe(pd.DataFrame([{"Scenario":k,"Price":round(v["price"],2),"P&L (price points)":round(v["pnl"],2)} for k,v in c.risk.stresses.items()]),use_container_width=True,hide_index=True)
    st.divider(); st.subheader("7. Indicative Product Proposal")
    text=build_markdown_term_sheet(mandate,c,avg,meta.get("combined_score"))
    st.download_button("Download Indicative Term Sheet",text.encode("utf-8"),file_name=f"{mandate.client_id}_{selected.replace(' ','_')}_indicative_term_sheet.md",mime="text/markdown",use_container_width=True)
