import streamlit as st

st.set_page_config(
    page_title="Investment App",

)

st.write("# Investment Helper 💵")

st.sidebar.success("Select a tool above.")

st.markdown(
    """
    Several free tools to aid you in portfolio construction or investing. 
    
    **👈 Select a tool from the sidebar** to get started!
    ### Features 
    - Portfolio Visualizer: Presents key visuals and statistics for any given portfolio
    - Algorithmic Trading: Build Equal-Weight S&P 500 Index Fund, Quantitative Momentum Investing Strategy, or Quantitative Value Investing Strategy
    - *Future Addition: Bear Market Recovery Predictor*
    
"""
)
