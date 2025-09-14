# Stock Price Predictor (Fuzzy Logic + XGBoost)

This project predicts stock prices using a hybrid approach of Fuzzy Logic and XGBoost regression. It provides Buy/Hold/Sell ratings and displays key company fundamentals in a web interface.

## Features

- Predicts next-day stock price using Fuzzy Logic and XGBoost.
- Visualizes actual vs predicted prices.
- Shows Buy/Hold/Sell recommendations for the last 10 days.
- Displays company fundamentals (Market Cap, PE Ratio, EPS, etc.).
- Interactive web interface built with Flask.

## Project Structure

- `app.py`: Flask web server and main entry point.
- `fuzzy_model.py`: Implements data fetching, feature engineering, fuzzy logic, XGBoost, and prediction logic.
- `templates/index.html`: Web UI template.
- `static/`: Contains generated charts and membership function plots.

## Requirements

- Python 3.8+
- Alpha Vantage API key (free from https://www.alphavantage.co/support/#api-key)
- Packages: `flask`, `numpy`, `pandas`, `matplotlib`, `skfuzzy`, `xgboost`, `scikit-learn`, `alpha_vantage`

Install dependencies:
```sh
pip install flask numpy pandas matplotlib scikit-learn xgboost alpha_vantage scikit-fuzzy
```

## Usage

1. Set your Alpha Vantage API key in `app.py` (`API_KEY` variable).
2. Run the app:
    ```sh
    python app.py
    ```
3. Open [http://localhost:5000](http://localhost:5000) in your browser.
4. Enter a stock symbol (e.g., `AAPL`), start date, and end date, then click "Predict".

## Output

- Interactive chart of actual vs predicted prices.
- Buy/Hold/Sell table for last 10 days.
- Company fundamentals.
- Membership function and model comparison plots saved in `static/`.

## Notes

- Predictions are for educational purposes only.
- Ensure your Alpha Vantage API key has sufficient quota.

## License

MIT License