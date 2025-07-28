# PFE (potential future exposure) Calculation - Monto Carlo
# Created by Jiejie Zhang, last modified on 2025/07/26.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Inputs =====
trade_date = pd.Timestamp('2026-01-01')
delivery_date = pd.Timestamp('2026-06-15')
payment_date = pd.Timestamp('2026-06-30')  # Payment 15 days post-delivery
fixed_price = 12  # $/MMBtu
cargo_size = 1000000  # MMBtu (assume 1 LNG cargo = 1,000,000 MMBtu.)
n_simulations = 10000  # Number of Monte Carlo simulations
confidence_level = 0.95  # 95% PFE

# Price model parameters (assume LNG price follows Geometric Brownian Motion with annual drift \mu and annualized volatility \signma)
initial_price = 12  # $/MMBtu (equal to fixed price at inception)
annual_volatility = 0.3  # 30% volatility (to be adjusted based on market)
annual_drift = 0.02  # 2% annual price drift
days_to_delivery = (delivery_date - trade_date).days

# ===== Simulate Daily Prices =====
np.random.seed(1234)  # For reproducibility
## starts from standard normal distribution, generates a NumPy array of random daily returns (shape given)
## daily_returns = P_t/P_{t-1} = exp{(\mu_daily - 0.5*(\sigma_daily)^2) + \sigma_daily * standard BM}
daily_returns = np.exp(
    (annual_drift - 0.5 * annual_volatility**2) * (1/252) +
    annual_volatility * np.sqrt(1/252) * np.random.randn(n_simulations, days_to_delivery)
)

# Generate price paths
price_paths = np.zeros_like(daily_returns)
price_paths[:, 0] = initial_price
for t in range(1, days_to_delivery):
    price_paths[:, t] = price_paths[:, t-1] * daily_returns[:, t]

# ===== Calculate Daily Exposure =====  -- to verify
# Pre-delivery exposure: max(0, Market Price - Fixed Price) * Cargo Size 
exposure_pre_delivery = np.maximum(price_paths - fixed_price, 0) * cargo_size

# Post-delivery exposure: Full contract value until payment
exposure_post_delivery = np.full((n_simulations, (payment_date - delivery_date).days), fixed_price * cargo_size)

# Combine into full exposure matrix
exposure = np.hstack([exposure_pre_delivery, exposure_post_delivery]) ## horizontal stack / row combine

# ===== Compute PFE (95th percentile) =====
pfe = np.percentile(exposure, confidence_level * 100, axis=0)

# ===== Generate Dates for Plotting =====
dates = pd.date_range(start=trade_date, end=payment_date)
pfe_curve = pd.Series(pfe, index=dates[:len(pfe)])

# ===== Plot PFE Curve =====
plt.figure(figsize=(10, 6))
plt.plot(pfe_curve, label='95% PFE', color='blue')
plt.axvline(delivery_date, color='black', linestyle='--', label='Delivery Date')
plt.axvline(payment_date, color='green', linestyle='--', label='Payment Date')
plt.title(f'PFE Evolution for LNG Trade (Fixed Price: ${fixed_price}/MMBtu)')
plt.xlabel('Date')
plt.ylabel('Exposure ($)')
plt.legend()
plt.grid(True)
plt.show()

# ===== Output Peak PFE =====
print(f"Peak Pre-Delivery PFE (95%): ${np.max(pfe_curve[:days_to_delivery]):,.0f}")
print(f"Post-Delivery PFE: ${fixed_price * cargo_size:,.0f} (Credit Risk)")