import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Linear Regression Interactive Explainer",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Interactive Linear Regression Explainer: House Prices")
st.write(
    "Explore how linear regression works using a real-world example of predicting house prices based on size."
)

# 2. Hardcoded Sample Data
X = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 35])
Y = np.array([145, 170, 215, 250, 290, 310, 365, 400, 430, 500])

# 3. Calculate Linear Regression Line (y = mx + c)
m, c = np.polyfit(X, Y, 1)

# Layout: Split into two columns
col1, col2 = st.columns([1, 1.2])

with col1:
  st.sidebar.header("Interactive Controls")
  user_size = st.sidebar.slider(
      "Select House Size (in 100 sq ft)",
      min_value=5,
      max_value=50,
      value=20,
      step=1,
  )

  predicted_price = m * user_size + c

  st.subheader("📊 Live Prediction")
  st.success(
      f"For a house size of **{user_size * 100} sq ft**, the estimated price"
      f" is **${predicted_price:.1f} thousand** (~${predicted_price * 1000:,.0f})."
  )

  st.markdown("---")

  # Educational Section: Understanding Linear Regression
  st.subheader("💡 What is Linear Regression?")
  st.write(
      "Linear regression is a supervised machine learning algorithm used to"
      " find a **linear relationship** between an independent variable ($X$)"
      " and a dependent variable ($Y$). It models the relationship using a"
      " straight line equation:"
  )
  st.latex(r"Y = mX + c")
  st.markdown(
      f"* **$m$ (Slope):** Here, $m = {m:.2f}$, meaning for every 100 sq ft"
      " increase, the price increases by roughly $"
      f"{m * 1000:,.0f}.\n* **$c$ (Y-Intercept):** Here, $c = {c:.2f}$, representing"
      " the baseline price value when size is zero."
  )

with col2:
  # Educational Section: Understanding the Graph
  st.subheader("📈 How to Understand This Graph")
  st.markdown("""
        * **Blue Dots (Actual Data):** Each dot represents a real historical house, plotting its actual size against its actual selling price.
        * **Red Line (Regression Line):** This is the "line of best fit." It is mathematically calculated to pass as close as possible to all blue dots simultaneously, minimizing overall error.
        * **Green Dot (Your Prediction):** This marker shows where your selected house size falls directly onto the model's trend line.
        """)

  # Plotting the Graph using Matplotlib
  fig, ax = plt.subplots(figsize=(7, 4.5))

  ax.scatter(X, Y, color="blue", s=60, label="Actual Dataset Houses")

  x_line = np.linspace(5, 50, 100)
  y_line = m * x_line + c
  ax.plot(
      x_line,
      y_line,
      color="red",
      linewidth=2,
      label=f"Regression Line (y = {m:.2f}x + {c:.2f})",
  )

  ax.scatter(
      [user_size],
      [predicted_price],
      color="green",
      s=140,
      zorder=5,
      label=f"Your House (${predicted_price:.1f}k)",
  )

  ax.set_xlabel("House Size (in 100 sq ft)")
  ax.set_ylabel("Price (in $1,000s)")
  ax.set_title("House Price vs. Size Trend")
  ax.legend(fontsize=9)
  ax.grid(True)

  st.pyplot(fig)