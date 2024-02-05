import streamlit as st
from scipy.stats import norm
import math
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas_datareader.data as web
import pandas as pd
import random
import string


# Fonction pour télécharger les données en temps réel
def fetch_real_time_price(stock_ticker1):
    stock_data = yf.download(stock_ticker1, period="1d")
    return stock_data['Close'].iloc[-1]

# Fonction pour calculer le prix d'une option européenne
def calculate_european_option(S, K, T, r, sigma, option_type):
    # Calculer d1 et d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Calculer le prix en fonction du type d'option
    if option_type == "Call":
        option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:  # "Put"
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price


st.set_page_config(page_title='Hajar CHAAIRI-Application Web de Simulation')
# Création des onglets
tab1, tab2, tab3= st.tabs([ "Simulation Mouvement Brownien géométrique du prix d'un stock(AAPL)", "Simulation Monte Carlo des prix futurs d'Apple Inc. (AAPL)",  "Simulation d'options Européenes(Put & Call)"])
# Initialisation de st.session_state si ce n'est pas déjà fait
if 'simulation_results' not in st.session_state:
    st.session_state['simulation_results'] = None
def simulate_aapl_brownian_motion(r, sigma, T, N):
    # Obtenir les données en temps réel pour AAPL
    aapl = yf.Ticker("AAPL")
    hist = aapl.history(period="1d")
    S0 = hist['Close'][-1]  # Utiliser le dernier prix de clôture comme prix initial

    # Exécution de la simulation
    dt = T/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W)*np.sqrt(dt)
    X = (r-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X)
    return t, S
    # Define a function to fetch real-time data
def fetch_stock_data(ticker_symbol):
    end_date = datetime.now()
    start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    return df['Close'][-1]
with tab1:
    st.header("Simulation Mouvement Brownien géométrique du prix d'un stock(AAPL)")
    r = st.number_input('Taux sans risque (r)', min_value=0.0, value=0.01, step=0.01)
    sigma = st.number_input('Volatilité (sigma)', min_value=0.0, value=0.2, step=0.01)
    T = st.number_input('Durée jusqu\'à l\'expiration (T en années)', min_value=0.0, value=1.0, step=0.1)
    N = st.number_input('Nombre de points dans le temps (N)', min_value=0, value=252, step=1)
    if st.button('Run Geometric Brownian Motion'):
        # Simulation du mouvement brownien géométrique
        t_aapl, S_aapl = simulate_aapl_brownian_motion(r, sigma, T, N)

       # Sauvegarde des résultats dans st.session_state pour les utiliser plus tard
        st.session_state['simulation_results'] = (t_aapl, S_aapl)

# Visualisation de la simulation
        fig, ax = plt.subplots()
        ax.plot(t_aapl, S_aapl, label='Simulation Mouvement Brownien')
        ax.set_title('Simulation du mouvement brownien géométrique pour AAPL')
        ax.set_xlabel('Temps (années)')
        ax.set_ylabel('Prix de l\'actif sous-jacent')
        ax.legend()
        st.pyplot(fig)
with tab2:
    st.header('Simulation Monte Carlo des prix futurs Apple Inc. (AAPL)')

    # Widgets for Monte Carlo simulation
    start_date = st.date_input('Start Date', datetime(2017, 3, 3), key='mc_start_date')
    end_date = st.date_input('End Date', datetime(2017, 10, 20), key='mc_end_date')
    num_simulations = st.number_input('Number of Simulations', 100, 10000, 1000, key='mc_num_simulations')
    num_days = st.number_input('Days to Forecast', 10, 365, 252, key='mc_num_days')
    if st.button('Run Monte Carlo Simulation', key='mc_run_simulations'):
        # Fetching stock data
        prices = web.DataReader('AAPL', 'av-daily', start_date, end_date, api_key='YOUR_API_KEY')['close']
        last_price = prices[-1]
        returns = prices.pct_change()
        daily_vol = returns.std()

        simulation_df = pd.DataFrame()

        for x in range(num_simulations):
            count = 0
            price_series = [last_price]

            for y in range(num_days):
                if count == 251:
                    break
                price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
                count += 1

            simulation_df[x] = price_series

        # Plotting
        plt.figure(figsize=(10,5))
        plt.plot(simulation_df)
        plt.axhline(y = last_price, color = 'r', linestyle = '-')
        plt.title('Monte Carlo Simulation: AAPL')
        plt.xlabel('Day')
        plt.ylabel('Price')
        st.pyplot(plt)

with tab3:
    st.header("Simulation d'options Européenes(Put & Call)")
    # Input fields for the European options simulation
    stock_ticker = st.text_input("Enter Stock Ticker", "AAPL", key='option_ticker').upper()
    S = fetch_real_time_price(stock_ticker)
    K = st.number_input("Strike Price (K)", value=100.0, key='option_strike')
    T = st.number_input("Time to Expiration (T) in Years", value=1.0, key='option_time')
    r = st.number_input("Risk-Free Rate (r)", value=0.01, key='option_rate')
    sigma = st.number_input("Volatility (σ)", value=0.2, key='option_volatility')
    option_type = st.selectbox("Type of Option", ["Call", "Put"], key='option_type')

    # Calculate option price button
    if st.button(f"Calculate {option_type} Option Price", key='calculate_option'):
        # European option price calculation
        option_price = calculate_european_option(S, K, T, r, sigma, option_type)
        st.success(f"The price of the {option_type} option is: {option_price:.2f}")
