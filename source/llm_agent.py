"""
LLM Agent Server - holds system prompt and handles analysis requests via HTTP
Runs on localhost:5001 (locally) or 0.0.0.0:5001 (Docker)
"""
import os
import json
import requests
import psycopg
from flask import Flask, request, jsonify

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "mydb"),
    "user": os.getenv("DB_USER", "myuser"),
    "password": os.getenv("DB_PASSWORD", "mypassword")
}

SYSTEM_PROMPT = """Ты - профессиональный анализатор диалогов колл-центра поликлиники.

Твоя задача:
1. Проанализировать диалог между работником и клиентом.
2. Оценить вежливость работника по шкале от 1 до 10, где:
   - 10: Идеально вежлив, профессионален, внимателен
   - 7-9: Вежлив и профессионален
   - 4-6: Приемлемо, но есть моменты невежливости
   - 1-3: Невежлив, грубый, непрофессионален
3. Определить, была ли решена проблема клиента (true/false)
4. Определить, была ли создана новая запись в поликлинику (true/false)
5. Объяснить оценку конкретными примерами из диалога

Ответь СТРОГО в формате json, больше ничего не добавляй:
{
    "politeness_score": [число от 1 до 10],
    "problem_solved": [true/false],
    "new_record_created": [true/false],
    "comment": [краткое объяснение с примерами]
}

"""

OLLAMA_API = os.getenv("OLLAMA_API", "http://127.0.0.1:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3")


def get_db_connection():
    """Create and return database connection"""
    try:
        conn = psycopg.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"[DB] Connection error: {e}")
        return None


def save_to_database(full_name: str, recorded_at: str, analysis_data: dict) -> tuple[bool, str | None]:
    """
    Save analysis result to database using save_work_result function
    Returns (success, error_message)
    """
    conn = get_db_connection()
    if not conn:
        return False, "Failed to connect to database"
    
    try:
        cursor = conn.cursor()
        
        # Extract data from analysis
        politeness_score = analysis_data.get("politeness_score", 5)
        problem_solved = analysis_data.get("problem_solved", False)
        new_record_created = analysis_data.get("new_record_created", False)
        comment = analysis_data.get("comment", "")
        
        # Call the save_work_result function
        cursor.execute(
            "SELECT save_work_result(%s, %s, %s, %s, %s, %s)",
            (full_name, recorded_at or None, politeness_score, problem_solved, new_record_created, comment)
        )
        
        result_id = cursor.fetchone()[0]
        conn.commit()
        
        if result_id is None:
            print(f"[DB] Employee not found: {full_name}")
            return False, f"Employee not found: {full_name}"
        
        print(f"[DB] Successfully saved work result with id: {result_id}")
        return True, None
        
    except Exception as e:
        print(f"[DB] Error saving to database: {e}")
        conn.rollback()
        return False, f"Database error: {e}"
    finally:
        conn.close()


def parse_llm_response(analysis: str) -> tuple[dict | None, str | None]:
    """
    Parse LLM response JSON, handling potential formatting issues
    Returns (parsed_data, error_message)
    """
    try:
        # Try to find JSON in the response
        start_idx = analysis.find('{')
        end_idx = analysis.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = analysis[start_idx:end_idx]
            return json.loads(json_str), None
        else:
            return None, "JSON not found in LLM response"
    except json.JSONDecodeError as e:
        print(f"[parse] JSON decode error: {e}")
        return None, f"JSON decode error: {e}"


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze dialog and return assessment"""
    data = request.json
    dialog_text = data.get("dialog", "")
    recorded_at = data.get("record_time", "")
    full_name = data.get("full_name", "")
    
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

            # Parse LLM response
            analysis_data, parse_error = parse_llm_response(analysis)
            
            if parse_error:
                print(f"[analyze] Parse error: {parse_error}")
                return jsonify({
                    "error": parse_error,
                    "raw_response": analysis,
                    "status": "error"
                }), 400
            
            # Save to database if full_name provided
            db_saved = False
            db_error = None
            
            if full_name:
                db_saved, db_error = save_to_database(full_name, recorded_at, analysis_data)
                print(f"[analyze] Database save: {'success' if db_saved else 'failed'}")
                if db_error:
                    print(f"[analyze] DB error: {db_error}")
            else:
                print("[analyze] No full_name provided, skipping database save")
                db_error = "No full_name provided"

            return jsonify({
                "analysis": analysis,
                "parsed_data": analysis_data,
                "db_saved": db_saved,
                "db_error": db_error,
                "status": "success" if db_saved else "partial"
            })
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
