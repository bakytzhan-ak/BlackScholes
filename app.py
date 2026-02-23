import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns


# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)


class BlackScholes:
    def __init__(
            self,
            time_to_maturity: float,
            strike: float,
            current_price: float,
            volatility: float,
            interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):

        self.d1 = (
                     log(self.current_price / self.strike) +
                     (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
             ) / (
                     self.volatility * sqrt(self.time_to_maturity)
             )
        self.d2 = self.d1 - self.volatility * sqrt(self.time_to_maturity)

        self.call_price = self.current_price * norm.cdf(self.d1) - (
                self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(self.d2)
        )
        self.put_price = (
                            self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(-self.d2)
                    ) - self.current_price * norm.cdf(-self.d1)

        # Delta
        self.call_delta = norm.cdf(self.d1)
        self.put_delta = 1 - norm.cdf(self.d1)

        # Gamma
        self.call_gamma = norm.pdf(self.d1) / (
                self.strike * self.volatility * sqrt(self.time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return {'call_price': self.call_price, 'put_price': self.put_price,
                'call_delta': self.call_delta, 'put_delta': self.put_delta,
                'call_gamma': self.call_gamma, 'put_gamma': self.put_gamma}


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    github_url = "https://github.com/bakytzhan-ak"
    st.markdown(
        f'<a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;">`Bakytzhan`</a>',
        unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=50.0, value=volatility * 0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=100.0, value=volatility * 1.5, step=0.01)

    spot_min = st.number_input('Min Spot Price (Fixed Strike)', min_value=0.01, value=current_price * 0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price (Fixed Strike)', min_value=0.01, value=current_price * 1.2, step=0.01)

    strike_min = st.number_input('Min Strike Price (Fixed Spot)', min_value=0.01, value=strike * 0.8, step=0.01)
    strike_max = st.number_input('Max Strike Price (Fixed Spot)', min_value=0.01, value=strike * 1.2, step=0.01)

    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)
    strike_range = np.linspace(strike_min, strike_max, 10)

def plot_heatmap_spot(bs_model, spot_range, vol_range, strike):

    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
                bs_temp_spot = BlackScholes(
                    time_to_maturity=bs_model.time_to_maturity,
                    strike=strike,
                    current_price=spot,
                    volatility=vol,
                    interest_rate=bs_model.interest_rate
                )
                bs_temp_spot.calculate_prices()
                call_prices[i, j] = bs_temp_spot.call_price
                put_prices[i, j] = bs_temp_spot.put_price

    min_price = min(call_prices.min(), put_prices.min())
    max_price = max(call_prices.max(), put_prices.max())

    fig_heatmap, ax = plt.subplots(ncols=2, figsize=(19, 8))
    ax_call, ax_put = ax

    # Plotting Call Price Heatmap
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                fmt=".2f", cmap="coolwarm", ax=ax_call, vmin=min_price, vmax=max_price)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')

    # Plotting Put Price Heatmap
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                fmt=".2f", cmap="coolwarm", ax=ax_put, vmin=min_price, vmax=max_price)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')

    ax_call.set_yticklabels(ax_call.get_yticklabels(), rotation=0, ha='right')
    ax_put.set_yticklabels(ax_put.get_yticklabels(), rotation=0, ha='right')

    plt.subplots_adjust(wspace=0.03)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=['CALL', 'PUT']
    )

    # Add the first surface trace
    fig.add_trace(
        go.Surface(z=call_prices, x=spot_range, y=vol_range, colorscale='RdBu_r', showscale=True),
        row=1, col=1
    )

    # Add the second surface trace
    fig.add_trace(
        go.Surface(z=put_prices, x=spot_range, y=vol_range, colorscale='RdBu_r', showscale=True),
        row=1, col=2
    )

    fig.data[0].update(colorbar=dict(x=0.45))
    fig.data[1].update(colorbar=dict(x=1.00))
    fig.update_traces(cmin=min_price, cmax=max_price,
                      contours_z=dict(
                          show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
                      )
                      )


    fig.update_layout(
        scene1=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Call Price'
        ),
        scene2=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Put Price'
        ),
        height=320,
        width=320,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig_heatmap, fig

def plot_heatmap_strike(bs_model, spot, vol_range, strike_range):

    call_prices = np.zeros((len(vol_range), len(strike_range)))
    put_prices = np.zeros((len(vol_range), len(strike_range)))

    for i, vol in enumerate(vol_range):
        for j, strike in enumerate(strike_range):
                bs_temp_strike = BlackScholes(
                    time_to_maturity=bs_model.time_to_maturity,
                    strike=strike,
                    current_price=spot,
                    volatility=vol,
                    interest_rate=bs_model.interest_rate
                )
                bs_temp_strike.calculate_prices()
                call_prices[i, j] = bs_temp_strike.call_price
                put_prices[i, j] = bs_temp_strike.put_price

    min_price = min(call_prices.min(), put_prices.min())
    max_price = max(call_prices.max(), put_prices.max())

    fig_heatmap, ax = plt.subplots(ncols=2, figsize=(19, 8))
    ax_call, ax_put = ax

    # Plotting Call Price Heatmap
    sns.heatmap(call_prices, xticklabels=np.round(strike_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                fmt=".2f", cmap="coolwarm", ax=ax_call, vmin=min_price, vmax=max_price)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Strike Price')
    ax_call.set_ylabel('Volatility')

    # Plotting Put Price Heatmap
    sns.heatmap(put_prices, xticklabels=np.round(strike_range, 2), yticklabels=np.round(vol_range, 2), annot=True,
                fmt=".2f", cmap="coolwarm", ax=ax_put, vmin=min_price, vmax=max_price)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Strike Price')
    ax_put.set_ylabel('Volatility')

    ax_call.set_yticklabels(ax_call.get_yticklabels(), rotation=0, ha='right')
    ax_put.set_yticklabels(ax_put.get_yticklabels(), rotation=0, ha='right')

    plt.subplots_adjust(wspace=0.03)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=['CALL', 'PUT']
    )

    # Add the first surface trace
    fig.add_trace(
        go.Surface(z=call_prices, x=strike_range, y=vol_range, colorscale='RdBu_r', showscale=True),
        row=1, col=1
    )

    # Add the second surface trace
    fig.add_trace(
        go.Surface(z=put_prices, x=strike_range, y=vol_range, colorscale='RdBu_r', showscale=True),
        row=1, col=2
    )

    fig.data[0].update(colorbar=dict(x=0.45))
    fig.data[1].update(colorbar=dict(x=1.00))
    fig.update_traces(cmin=min_price, cmax=max_price,
                      contours_z=dict(
                          show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
                      )
                      )


    fig.update_layout(
        scene1=dict(
            xaxis_title='Strike Price',
            yaxis_title='Volatility',
            zaxis_title='Call Price'
        ),
        scene2=dict(
            xaxis_title='Strike Price',
            yaxis_title='Volatility',
            zaxis_title='Put Price'
        ),
        height=320,
        width=320,
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig_heatmap, fig


# Main Page for Output Display
st.markdown("<h1 style='text-align: center;'>Black-Scholes Pricing Model</h1>", unsafe_allow_html=True)

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values: prices, deltas
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
bs_calculation = bs_model.calculate_prices()
call_price, put_price = bs_calculation['call_price'], bs_calculation['put_price']
call_delta, put_delta = bs_calculation['call_delta'], bs_calculation['put_delta']

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
                <div class="metric-label">CALL Delta</div>
                <div class="metric-value">${call_delta:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
                <div class="metric-label">PUT Delta</div>
                <div class="metric-value">${put_delta:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("<h1 style='text-align: center;'>Options Price - Interactive Map</h1>", unsafe_allow_html=True)

# Interactive Sliders and Heatmaps for Call and Put Options
st.markdown("<h2 style='text-align: center; color: brown;'>Spot Price-Volatility Heatmap</h2>", unsafe_allow_html=True)
heatmap_fig_spot, surface_fig_spot = plot_heatmap_spot(bs_model, spot_range, vol_range, strike)
st.pyplot(heatmap_fig_spot)
st.markdown("<h2 style='text-align: center; color: brown;'>Spot Price-Volatility Surfacemap</h2>", unsafe_allow_html=True)
st.plotly_chart(surface_fig_spot)

st.markdown("")

st.markdown("<h2 style='text-align: center; color: brown;'>Strike Price-Volatility Heatmap</h2>", unsafe_allow_html=True)
heatmap_fig_strike, surface_fig_strike = plot_heatmap_strike(bs_model, current_price, vol_range, strike_range)
st.pyplot(heatmap_fig_strike)
st.markdown("<h2 style='text-align: center; color: brown;'>Strike Price-Volatility Surfacemap</h2>", unsafe_allow_html=True)
st.plotly_chart(surface_fig_strike)
