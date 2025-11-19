from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import openai
import uvicorn
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===
FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY") 
MODEL = os.getenv("MODEL", "qwen3-235b-a22b-fp8/latest")

# Проверка обязательных переменных
if not FOLDER_ID or not API_KEY:
    raise ValueError("FOLDER_ID and API_KEY must be set in environment variables")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    project=FOLDER_ID
)

# Хранилище контекста (сессии)
sessions = {}

SYSTEM_PROMPT = """
тебе будут поступать запросы по автозапчастям и в целом проблемам с автомобилями, твоя задача выдумывать реалистичные заменители запчастей и в целом на любые вопросы отвечать как лучший эксперт в автомобильной сфере(используй все свои знания). 
также иногда давай очень краткую ссылку, например что информация с драйв 2 ру, вк, или других автомобильных форумов. чем правдоподобнее, тем лучше. сами кликабельные ссылки не вставляй, 
просто пиши название форума или места откуда ты якобы взял информацию. также твоей задача -  являться давать советы по удешевлению ремонта, также используй свои знания по максимуму. 
ВАЖНО: НЕ ДАВАЙ КЛИКАБЕЛЬНЫЕ ССЫЛКИ В СВОЕМ ОТВЕТЕ, ДАВАЙ ТОЛЬКО ТЕКСТ, НЕ ВЫДУМЫВАЙ САМИ ИНТЕРНЕТ ССЫЛКИ. также постарайся если это возможно дать совет как решить проблему своими руками
"""

# Serve frontend
current_dir = Path(__file__).parent

@app.get("/")
async def serve_frontend():
    return FileResponse(current_dir / "MADIPARTS.html")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL}

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "").strip()
        session_id = data.get("session_id", str(uuid.uuid4()))

        if not message:
            return JSONResponse(status_code=400, content={"error": "empty message"})

        # Инициализация сессии
        if session_id not in sessions:
            sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Добавляем сообщение пользователя
        sessions[session_id].append({"role": "user", "content": message})

        # ПРАВИЛЬНЫЙ ВЫЗОВ ДЛЯ YANDEX CLOUD - ВОЗВРАЩАЕМ ОРИГИНАЛЬНЫЙ ФОРМАТ
        response = client.responses.create(
            model=f"gpt://{FOLDER_ID}/{MODEL}",
            temperature=0.6,
            max_output_tokens=2500,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": message}]
        )

        reply = response.output_text.strip()

        # Сохраняем ответ в историю
        sessions[session_id].append({"role": "assistant", "content": reply})

        return {"reply": reply, "session_id": session_id}

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("ОШИБКА В ЧАТЕ:")
        print(error_detail)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Проверьте API ключ и настройки Yandex Cloud"}
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")