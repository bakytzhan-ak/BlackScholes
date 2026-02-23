# 📈 Black-Scholes Pricing Model

An interactive dashboard for visualizing option prices using the **Black-Scholes Pricing Model**.  
This tool helps users explore how changes in parameters such as spot price, volatility, strike price, and time to maturity influence the value of call and put options.

---

## 🚀 Features

- **Options Pricing Visualization**
  - Interactive heatmaps for both **Call** and **Put** option prices.
  - Dynamic updates as you adjust parameters like **Spot Price**, **Volatility**, and **Time to Maturity**.

- **Interactive Dashboard**
  - Real-time updates to Black-Scholes model parameters.
  - Input custom values for:
    - Spot Price  
    - Volatility  
    - Strike Price  
    - Time to Maturity  
    - Risk-Free Interest Rate  
  - Instant calculation and comparison of Call and Put option prices.

- **Customizable Parameters**
  - Define custom ranges for Spot Price, Strike Price and Volatility.
  - Generate a comprehensive view of option prices under varying market conditions.

---

## 🔧 Dependencies

This project requires the following Python libraries:

| Dependency    | Purpose              |
|---------------|----------------------|
| **streamlit** | Interactive app tool |
| **pandas**    | Data process         |
| **scipy**     | Math Stats operation |
| **numpy**     | Numerical operations |
| **matplotlib**| Map visualization    |
| **plotly**    | Surface visualization|
| **seaborn**   | Heatmap visualization|

Install them with:

```bash```
uv pip sync requirements.txt

Integrating with pyproject.toml
```bash```
uv add -r requirements.txt

Installation & Usage
- Clone the repository

- Run the dashboard:
streamlit run app.py

- Open the interactive interface in your browser and start exploring option pricing dynamics.


Model Overview
The Black-Scholes model calculates option prices using the formula:
C=S_0\cdot N(d_1)-K\cdot e^{-rT}\cdot N(d_2)
Where:
- C = Call option price
- S_0 = Current spot price
- K = Strike price
- T = Time to maturity
- r = Risk-free interest rate
- N(\cdot ) = Cumulative distribution function of the standard normal distribution
Put option prices are derived similarly.

Roadmap
- [ ] Add support for dividend-adjusted Black-Scholes model
- [ ] Integrate historical volatility analysis
