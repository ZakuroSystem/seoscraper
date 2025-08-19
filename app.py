from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup

from analysis import analyze

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    data = request.get_json(force=True)
    content = data.get("text", "")
    url = data.get("url")
    if url:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text(" ")
        except requests.RequestException:
            return jsonify({"error": "Failed to retrieve URL"}), 400
    result = analyze(content)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
