from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return """
    <h1>Welcome to Volunteer Hours Clustering API</h1>
    <p>Send a POST request to /cluster with JSON data containing 'hours'</p>
    <p>Example using curl:</p>
    <code>
    curl -X POST -H "Content-Type: application/json" -d '{"hours": [10, 25, 8, 30]}' http://localhost:5000/cluster
    </code>
    """


@app.route('/cluster', methods=['GET', 'POST'])
def cluster_data():
    if request.method == 'GET':
        return jsonify({
            "message": "Send a POST request with JSON data containing 'hours'",
            "example_request": {
                "method": "POST",
                "url": "/cluster",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "hours": [10, 25, 8, 30]
                }
            }
        })

    try:
        # Get data from JSON request
        if request.is_json:
            json_data = request.get_json()

            # Check if 'hours' key exists
            if 'hours' not in json_data:
                return jsonify({"error": "'hours' key not found in JSON data"}), 400

            # Convert to DataFrame
            data = pd.DataFrame({'hours': json_data['hours']})

        # For backward compatibility with CSV uploads
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            data = pd.read_csv(file)
        else:
            return jsonify({
                "error": "Unsupported content type",
                "solution": "Send either JSON with 'hours' array or a CSV file"
            }), 400

        # Drop rows with missing values
        data = data.dropna()

        if data.empty:
            return jsonify({"error": "All rows were dropped because of missing values"}), 400

        # Prepare data for clustering
        X = data[['hours']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X_scaled)
        data['cluster'] = kmeans.labels_

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)

        # Create plot (optional - can be removed if not needed)
        plt.figure(figsize=(10, 3))
        plt.scatter(X_scaled[:, 0], np.zeros(len(X_scaled)), c=kmeans.labels_, cmap='viridis', s=100)
        plt.title('K-Means Clustering of Volunteer Hours')
        plt.xlabel('Volunteer Hours (scaled)')
        plt.yticks([])
        plt.colorbar(label='Cluster')

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Prepare response
        response = {
            "status": "success",
            "silhouette_score": silhouette_avg,
            "clustered_data": data.to_dict(orient='records'),
            "cluster_summary": {
                "total_samples": len(data),
                "cluster_counts": data['cluster'].value_counts().to_dict()
            },
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "solution": "Please ensure you're sending valid data (JSON with 'hours' array or CSV file)"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)