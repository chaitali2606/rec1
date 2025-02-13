from flask import Flask, request, jsonify
import pandas as pd
from recommender import get_recommendations, df, tfidf, cosine_sim

app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    user_query = request.args.get("query")
    
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    recommendations = get_recommendations(user_query, df, tfidf, cosine_sim)

    if recommendations.empty:
        return jsonify({"message": "No matching recommendations found"}), 404

    return recommendations.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
