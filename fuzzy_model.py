import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


def get_stock_data(symbol, start_date, end_date, api_key):
    ts = TimeSeries(key=api_key, output_format="pandas")
    df, _ = ts.get_daily(symbol=symbol, outputsize="full")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df[::-1]
    df = df.loc[start_date:end_date]

    # Technical Indicators
    df["SMA"] = df["Close"].rolling(window=14).mean()
    df["EMA"] = df["Close"].ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(window=14).mean() /
                               df["Close"].pct_change().rolling(window=14).std()))
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Stochastic"] = (df["Close"] - df["Low"].rolling(14).min()) / (df["High"].rolling(14).max() - df["Low"].rolling(14).min()) * 100
    df["ADX"] = abs(df["High"] - df["Low"]).rolling(14).mean()
    df.dropna(inplace=True)
    return df

def fuzzy_predict(df):
    sma = ctrl.Antecedent(np.arange(df["SMA"].min(), df["SMA"].max(), 0.1), "SMA")
    rsi = ctrl.Antecedent(np.arange(0, 100, 1), "RSI")
    macd = ctrl.Antecedent(np.arange(df["MACD"].min(), df["MACD"].max(), 0.1), "MACD")
    stochastic = ctrl.Antecedent(np.arange(0, 100, 1), "Stochastic")
    adx = ctrl.Antecedent(np.arange(df["ADX"].min(), df["ADX"].max(), 0.1), "ADX")
    predicted_price = ctrl.Consequent(np.arange(df["Close"].min()*0.9, df["Close"].max()*1.1, 0.1), "Predicted_Price")

    sma.automf(3)
    rsi.automf(3)
    macd.automf(3)
    stochastic.automf(3)
    adx.automf(3)

    predicted_price["low"] = fuzz.trimf(predicted_price.universe, [df["Close"].min()*0.9, df["Close"].min(), df["Close"].mean()])
    predicted_price["stable"] = fuzz.trimf(predicted_price.universe, [df["Close"].mean()*0.9, df["Close"].mean(), df["Close"].mean()*1.1])
    predicted_price["high"] = fuzz.trimf(predicted_price.universe, [df["Close"].mean(), df["Close"].max(), df["Close"].max()*1.1])

    def save_membership_plot(var, name):
        
        plt.title(f"{name} Membership Function")
        plt.savefig(f"static/membership_{name}.png")
        plt.close()

    save_membership_plot(sma, "")
    save_membership_plot(rsi, "RSI")
    save_membership_plot(macd, "MACD")
    save_membership_plot(stochastic, "Stochastic")
    save_membership_plot(adx, "ADX")
    save_membership_plot(predicted_price, "Predicted_Price")

    rules = [
        ctrl.Rule(sma["poor"] & rsi["poor"], predicted_price["low"]),
        ctrl.Rule(sma["good"] & rsi["good"], predicted_price["high"]),
        ctrl.Rule(macd["poor"] & stochastic["poor"], predicted_price["low"]),
        ctrl.Rule(macd["good"] & stochastic["good"], predicted_price["high"]),
        ctrl.Rule(adx["average"], predicted_price["stable"]),
        ctrl.Rule(macd["average"] & rsi["average"], predicted_price["stable"]),
        
        ctrl.Rule(sma["average"] & rsi["good"] & macd["good"], predicted_price["high"]),
        ctrl.Rule(sma["poor"] & macd["poor"] & adx["poor"], predicted_price["low"]),
        ctrl.Rule(rsi["good"] & stochastic["good"] & adx["good"], predicted_price["high"]),
        ctrl.Rule(sma["average"] & rsi["average"] & adx["average"], predicted_price["stable"]),
        ctrl.Rule(macd["good"] & stochastic["average"] & adx["good"], predicted_price["high"]),
        ctrl.Rule(macd["poor"] & stochastic["average"] & adx["poor"], predicted_price["low"]),
    ]

    stock_ctrl = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(stock_ctrl)

    fuzzy_preds = []
    for _, row in df.iterrows():
        try:
            sim.input["SMA"] = row["SMA"]
            sim.input["RSI"] = row["RSI"]
            sim.input["MACD"] = row["MACD"]
            sim.input["Stochastic"] = row["Stochastic"]
            sim.input["ADX"] = row["ADX"]
            sim.compute()
            fuzzy_preds.append(sim.output["Predicted_Price"])
        except:
            fuzzy_preds.append(row["Close"])
    return fuzzy_preds, sim



def xgboost_predict(df):
    features = ["SMA", "RSI", "MACD", "Stochastic", "ADX"]
    X = df[features]
    y = df["Close"]
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    xgb_preds = model.predict(X)
    return xgb_preds, model

def get_fundamentals(symbol, api_key):
    fd = FundamentalData(key=api_key, output_format="pandas")
    data, _ = fd.get_company_overview(symbol=symbol)
    fields = [
        "MarketCapitalization", "PERatio", "PEGRatio", "EPS", "ProfitMargin",
        "QuarterlyRevenueGrowthYOY", "QuarterlyEarningsGrowthYOY"
    ]
    return {k: data[k][0] for k in fields if k in data.columns}

def run_model(symbol, start_date, end_date, api_key):
    df = get_stock_data(symbol, start_date, end_date, api_key)
    fuzzy_preds, fuzzy_sim = fuzzy_predict(df)
    xgb_preds, xgb_model = xgboost_predict(df)
    combined_preds = (np.array(fuzzy_preds) + np.array(xgb_preds)) / 2

    df["Fuzzy_Pred"] = fuzzy_preds
    df["XGB_Pred"] = xgb_preds
    df["Combined"] = combined_preds

    # Add Buy/Hold/Sell Rating
    def classify_rating(predicted, actual):
        if predicted > actual * 1.02:
            return "BUY 📈"
        elif predicted < actual * 0.98:
            return "SELL 📉"
        else:
            return "HOLD 🤝"

    df["Rating"] = df.apply(lambda row: classify_rating(row["Combined"], row["Close"]), axis=1)

    last_10 = df[["Close", "Combined", "Rating"]].tail(10)
    last_10["Date"] = last_10.index.strftime('%Y-%m-%d')
    rating_table = last_10[["Date", "Close", "Combined", "Rating"]].values.tolist()

    # Predict Next Day
    next_input = df.iloc[-1][["SMA", "RSI", "MACD", "Stochastic", "ADX"]]
    fuzzy_sim.input["SMA"] = next_input["SMA"]
    fuzzy_sim.input["RSI"] = next_input["RSI"]
    fuzzy_sim.input["MACD"] = next_input["MACD"]
    fuzzy_sim.input["Stochastic"] = next_input["Stochastic"]
    fuzzy_sim.input["ADX"] = next_input["ADX"]
    fuzzy_sim.compute()
    next_fuzzy = fuzzy_sim.output["Predicted_Price"]
    next_xgb = xgb_model.predict([next_input])[0]
    next_combined = (next_fuzzy + next_xgb) / 2

    fundamentals = get_fundamentals(symbol, api_key)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    actual = df["Close"]
    predicted = df["Combined"]

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print("\n📊 Model Performance Metrics:",flush=True)
    print(f"Mean Absolute Error (MAE): {mae:.4f}",flush=True)
    print(f"Mean Squared Error (MSE): {mse:.4f}",flush=True)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}",flush=True)
    print(f"R² Score: {r2:.4f}",flush=True)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%",flush=True)
    
    
    def plot_model_comparison(df):
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df["Close"], label="📈 Actual Price", color="black")
        plt.plot(df.index, df["Fuzzy_Pred"], label="🧠 Fuzzy Logic", linestyle="--", color="blue")
        plt.plot(df.index, df["XGB_Pred"], label="🤖 XGBoost", linestyle="--", color="orange")
        plt.plot(df.index, df["Combined"], label="🔮 Combined Prediction", color="green", linewidth=2)
        plt.title("Stock Price Prediction: Actual vs Fuzzy vs XGBoost vs Combined")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("static/model_comparison.png")
        
    return {
        "df": df,
        "next_fuzzy": next_fuzzy,
        "next_xgb": next_xgb,
        "next_combined": next_combined,
        "fundamentals": fundamentals,
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "actual_prices": df["Close"].tolist(),
        "predicted_prices": df["Combined"].tolist(),
        "ratings_table": rating_table
        
    }
    
    