from flask import Flask, request, jsonify
import os
from your_image_search_module import search_image  # Import your search function here

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    image_url = data.get('image_url')
    threshold = data.get('threshold', 0.8)  # Default threshold to 0.8
    
    # Call your search function
    results = search_image(image_url, threshold)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
