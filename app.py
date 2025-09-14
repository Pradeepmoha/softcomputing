from flask import Flask, render_template, request
from fuzzy_model import run_model

app = Flask(__name__)

API_KEY = "E1CMI0A0KKX3I8Y2"

@app.route("/", methods=["GET", "POST"])
def index():
    context = {}
    if request.method == "POST":
        symbol = request.form["symbol"].upper()
        start_date = request.form["start"]
        end_date = request.form["end"]

        try:
            results = run_model(symbol, start_date, end_date, API_KEY)
            context = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "next_fuzzy": round(results["next_fuzzy"], 2),
                "next_xgb": round(results["next_xgb"], 2),
                "next_combined": round(results["next_combined"], 2),
                "fundamentals": results["fundamentals"],
                "dates": results["dates"],
                "actual": results["actual_prices"],
                "predicted": results["predicted_prices"],
                "ratings_table": results["ratings_table"]
            }
        except Exception as e:
            context["error"] = f"Something went wrong: {str(e)}"

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(debug=True)
