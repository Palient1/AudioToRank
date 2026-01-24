"""
LLM Agent Server - holds system prompt and handles analysis requests via HTTP
Runs on localhost:5001 (locally) or 0.0.0.0:5001 (Docker)
"""
import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

SYSTEM_PROMPT = """Ты - профессиональный анализатор диалогов колл-центра поликлиники.

Твоя задача:
1. Проанализировать диалог между работником и клиентом.
2. Оценить вежливость работника по шкале от 1 до 10, где:
   - 10: Идеально вежлив, профессионален, внимателен
   - 7-9: Вежлив и профессионален
   - 4-6: Приемлемо, но есть моменты невежливости
   - 1-3: Невежлив, грубый, непрофессионален
3. Определить, была ли решена проблема клиента (Да/Нет)
4. Определить, была ли создана новая запись в поликлинику (Да/Нет)
5. Объяснить оценку конкретными примерами из диалога

Ответь в формате:
Оценка вежливости: [число]/10
Проблема решена: [Да/Нет]
Новая запись создана: [Да/Нет]
Объяснение: [краткое объяснение с примерами]"""

OLLAMA_API = os.getenv("OLLAMA_API", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze dialog and return assessment"""
    data = request.json
    dialog_text = data.get("dialog", "")
    
    if not dialog_text:
        return jsonify({"error": "No dialog provided"}), 400
    
    full_prompt = f"{SYSTEM_PROMPT}\n\nДиалог:\n{dialog_text}"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }
    
    print(f"[analyze] Sending request to Ollama at {OLLAMA_API}")
    
    try:
        print(f"[analyze] Calling Ollama with timeout=600...")
        response = requests.post(OLLAMA_API, json=payload, timeout=600)
        print(f"[analyze] Got response with status {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            analysis = result.get("response", "").strip()
            print(f"[analyze] Success, response length: {len(analysis)}")
            return jsonify({"analysis": analysis, "status": "success"})
        else:
            print(f"[analyze] Ollama error: {response.status_code}")
            return jsonify({"error": f"Ollama error: {response.status_code}", "status": "error"}), 500
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Cannot connect to Ollama on {OLLAMA_API}", "status": "error"}), 500
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({"status": "ok", "model": MODEL_NAME})


if __name__ == "__main__":
    PORT = 5001
    # Detect if running in Docker or locally
    IN_DOCKER = os.path.exists("/.dockerenv")
    HOST = "0.0.0.0" if IN_DOCKER else "127.0.0.1"
    
    print(f"LLM Agent Server starting on http://{HOST}:{PORT}")
    print(f"Running in: {'Docker' if IN_DOCKER else 'Local'}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Ollama API: {OLLAMA_API}")
    print(f"System prompt loaded.\n")
    print("Endpoints:")
    print(f"  POST /analyze - Send dialog_text, get analysis")
    print(f"  GET /health - Health check\n")
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except OSError as e:
        print(f"Error: {e}")
        print(f"Port {PORT} might be already in use. Try: lsof -i :{PORT}")
