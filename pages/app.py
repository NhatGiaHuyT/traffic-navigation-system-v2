from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from llama_cpp import Llama
from algo import optimize_traffic, detect_cars

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 "..", "models",
                                 "nomic-ai-gpt4all-falcon-Q2_K.gguf"))
)
llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=20)

# Load and summarize CSV data once at startup
def load_and_summarize_data():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "district5_complex_routes_with_datetime.csv"))
    try:
        df = pd.read_csv(csv_path)
        # Extract minimal key unique values and statistics for very concise data context
        unique_origins = df['Origin'].dropna().unique()[:3]
        unique_destinations = df['Destination'].dropna().unique()[:3]
        unique_routes = df['Route'].dropna().unique()[:3]
        weather_conditions = df['Weather Conditions'].dropna().unique()[:2]
        total_accidents = df['Accident Reports'].sum()
        avg_traffic_intensity = df['Traffic Intensity'].mean()

        summary_lines = [
            "Traffic and route data summary:",
            f"- Origins: {', '.join(unique_origins)}",
            f"- Destinations: {', '.join(unique_destinations)}",
            f"- Routes: {', '.join(unique_routes)}",
            f"- Weather conditions: {', '.join(weather_conditions)}",
            f"- Total accident reports: {total_accidents}",
            f"- Average traffic intensity: {avg_traffic_intensity:.2f}",
        ]

        # Remove example Q&A pairs to reduce prompt length
        summary = "\n".join(summary_lines)
    except Exception as e:
        summary = f"Could not load data summary due to error: {e}"
    return summary

data_summary = load_and_summarize_data()

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'reply': "Please send a non-empty message."}), 400

    try:
        system_prompt = "You are a helpful virtual assistant specialized in traffic and route optimization."
        prompt = f"{system_prompt}\n{data_summary}\nUser: {user_message}\nAssistant:"
        response = llm(prompt=prompt, max_tokens=150, stop=["User:", "Assistant:"])
        reply_text = response.get('choices', [{}])[0].get('text', '').strip()
        if not reply_text:
            reply_text = "Sorry, I couldn't generate a response."
        return jsonify({'reply': reply_text})
    except Exception as e:
        return jsonify({'reply': f"Error processing your message: {e}"}), 500


# ... your /upload route and app.run() as before ...


@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('videos')
    if len(files) != 4:
        return jsonify({'error': 'Please upload exactly 4 videos'}), 400

    os.makedirs('uploads', exist_ok=True)
    video_paths = []
    for i, file in enumerate(files):
        path = os.path.join('uploads', f'video_{i}.mp4')
        file.save(path)
        video_paths.append(path)

    num_cars_list = [detect_cars(p) for p in video_paths]
    result = optimize_traffic(num_cars_list)
    return jsonify(result)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
