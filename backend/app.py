from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from llama_cpp import Llama
from algo import optimize_traffic, detect_cars

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 "..", "models",
                                 "ggml-gpt4all-j-v1.3-groovy.bin"))
)
llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_gpu_layers=0)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'reply': "Please send a non-empty message."}), 400

    try:
        res = llm.create_completion(
            prompt=user_message,
            max_tokens=150,
            temperature=0.7
        )
        reply = res.choices[0].text.strip()
    except Exception as e:
        reply = f"Error generating reply: {e}"

    return jsonify({'reply': reply})

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
