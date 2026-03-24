#!/usr/bin/env python3
"""Mock API server for the agent arena.

Provides simple endpoints that agents can interact with during episodes.
Run with: python /opt/mock_api_server.py
"""

from flask import Flask, jsonify, request

app = Flask(__name__)

# --- Mock data -----------------------------------------------------------

WEATHER_DATA = {
    "new york": {"temp_f": 72, "condition": "partly cloudy", "humidity": 55},
    "london": {"temp_f": 59, "condition": "rainy", "humidity": 80},
    "tokyo": {"temp_f": 78, "condition": "sunny", "humidity": 60},
    "sydney": {"temp_f": 68, "condition": "clear", "humidity": 45},
}

SEARCH_DATA = [
    {"title": "Python documentation", "url": "https://docs.python.org", "snippet": "Official Python docs"},
    {"title": "Flask quickstart", "url": "https://flask.palletsprojects.com", "snippet": "Flask web framework"},
    {"title": "Pandas guide", "url": "https://pandas.pydata.org", "snippet": "Data analysis library"},
]

ORDERS_DATA = {
    "ORD-001": {"item": "Widget A", "quantity": 3, "status": "shipped", "total": 29.97},
    "ORD-002": {"item": "Gadget B", "quantity": 1, "status": "processing", "total": 49.99},
    "ORD-003": {"item": "Widget C", "quantity": 5, "status": "delivered", "total": 74.95},
}

# --- Endpoints ------------------------------------------------------------


@app.route("/weather", methods=["GET"])
def weather():
    city = request.args.get("city", "").lower().strip()
    if not city:
        return jsonify({"error": "Missing 'city' query parameter"}), 400
    data = WEATHER_DATA.get(city)
    if data is None:
        return jsonify({"error": f"No weather data for '{city}'"}), 404
    return jsonify({"city": city, **data})


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").lower().strip()
    if not query:
        return jsonify({"error": "Missing 'q' query parameter"}), 400
    results = [r for r in SEARCH_DATA if query in r["title"].lower() or query in r["snippet"].lower()]
    return jsonify({"query": query, "results": results})


@app.route("/orders", methods=["GET"])
def list_orders():
    status_filter = request.args.get("status", "").lower().strip()
    if status_filter:
        filtered = {k: v for k, v in ORDERS_DATA.items() if v["status"] == status_filter}
    else:
        filtered = ORDERS_DATA
    return jsonify({"orders": filtered})


@app.route("/orders/<order_id>", methods=["GET"])
def get_order(order_id: str):
    order = ORDERS_DATA.get(order_id.upper())
    if order is None:
        return jsonify({"error": f"Order '{order_id}' not found"}), 404
    return jsonify({"order_id": order_id.upper(), **order})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
