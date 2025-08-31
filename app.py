import os
import urllib.request, urllib.parse
from flask import Flask, request
import psycopg2
import psycopg2.extras

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"
DATABASE_URL = os.environ.get("DATABASE_URL")  # строка подключения Neon

app = Flask(__name__)

def db():
    # Одно соединение на процесс — просто и достаточно для Render
    if not hasattr(app, "_db"):
        app._db = psycopg2.connect(DATABASE_URL, sslmode="require")
    return app._db

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()

@app.get("/health")
def health():
    return {"ok": True, "service": "memory-bot"}

@app.post(WEBHOOK_PATH)
def webhook():
    upd = request.get_json(force=True, silent=True) or {}
    msg = upd.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    user = msg.get("from") or {}
    user_id = user.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}

    if text == "/start":
        send_message(chat_id,
            "Привет! Я бот-память.\n"
            "Команды:\n"
            "/help — справка\n"
            "/use <Project> — выбрать активный проект (например, /use Innertrade)\n"
        )
        return {"ok": True}

    if text == "/help":
        send_message(chat_id,
            "Справка:\n"
            "/use <Project> — активировать проект для этого чата.\n"
            "Дальше добавим /find, /pin и интеграцию с курсом."
        )
        return {"ok": True}

    if text.startswith("/use"):
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            send_message(chat_id, "Укажи проект: например, /use Innertrade")
            return {"ok": True}
        project = parts[1].strip()

        # upsert в chat_context
        conn = db()
        with conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                INSERT INTO chat_context (chat_id, user_id, project)
                VALUES (%s, %s, %s)
                ON CONFLICT (chat_id, user_id)
                DO UPDATE SET project = EXCLUDED.project, created_at = now()
            """, (chat_id, user_id, project))
        send_message(chat_id, f"Проект активирован: {project}")
        return {"ok": True}

    # Фоллбэк: пока просто отвечаем, дальше добавим /find и RAG
    send_message(chat_id, "Команда не распознана. Введи /help")
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
