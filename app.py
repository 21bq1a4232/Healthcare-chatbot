import os
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for sessions

client = OpenAI(
  base_url="https://api-inference.huggingface.co/v1",
  api_key = os.getenv('API_KEY')  # Replace with your OpenAI API key
)
model_links = {
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Nous-Hermes-2-Mixtral-8x7B-DPO": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Yi-1.5-34B-Chat": "01-ai/Yi-1.5-34B-Chat",
    "Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma-7B":"google/gemma-1.1-7b-it",
    "Zephyr-7B-Beta": "HuggingFaceH4/zephyr-7b-beta",
    "Zephyr-7B-Alpha": "HuggingFaceH4/zephyr-7b-alpha",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
}

@app.route('/')
def index():
    models = list(model_links.keys())
    return render_template('index.html', models=models)

@app.route('/chat', methods=['GET'])
def chat():
    message = request.args.get('message')
    selected_model = request.args.get('model')
    temperature = float(request.args.get('temperature', 0.5))

    if not message or not selected_model:
        return jsonify({'error': 'Missing message or model'}), 400

    if selected_model not in model_links:
        return jsonify({'error': 'Invalid model selection'}), 400

    repo_id = model_links[selected_model]
    
    if 'conversation' not in session:
        session['conversation'] = []
    session['conversation'].append({"role": "user", "content": message})

    messages = [
        {"role": "system", "content": "You are a healthcare assistant. you will answer the questions related to healthcare. And you think wisely before answering. And dont answer like you are not a human."},
        {"role": "user", "content": message}
    ]

    def generate():
        for chunk in client.chat.completions.create(
            model=repo_id,
            messages=messages,
            temperature=temperature,
            max_tokens=3000,
            stream=True,
        ):
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content)
                yield f"data: {chunk.choices[0].delta.content}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)