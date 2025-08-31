import os
import urllib.request
import urllib.parse
from flask import Flask, request
import psycopg  # psycopg v3

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"
DATABASE_URL = os.environ["DATABASE_URL"]  # Neon URL

app = Flask(__name__)

# --- DB helpers (psycopg v3) -------------------------------------------------
def get_conn():
    # Простое ленивое соединение, reuse между запросами
    if not hasattr(app, "_db_conn"):
        app._db_conn = psycopg.connect(DATABASE_URL, sslmode="require", autocommit=True)
        init_db(app._db_conn)
    return app._db_conn

def init_db(conn):
    """Создаём минимально нужные сущности, если их ещё нет."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_context (
              id SERIAL PRIMARY KEY,
              chat_id BIGINT NOT NULL,
              user_id BIGINT NOT NULL,
              project TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT now()
            );
        """)
        # уникальность пары (chat_id, user_id) для upsert
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ix_chat_context_chat_user
              ON chat_context (chat_id, user_id);
        """)

# --- Telegram helpers ---------------------------------------------------------
def send_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()

# --- Health -------------------------------------------------------------------
@app.get("/health")
def health():
    # Пинг БД по-быстрому
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        db_ok = True
    except Exception:
        db_ok = False
    return {"ok": True, "service": "innertrade-memory-bot", "db": db_ok}

# --- Webhook ------------------------------------------------------------------
@app.post(WEBHOOK_PATH)
def webhook():
    upd = request.get_json(force=True, silent=True) or {}
    msg = upd.get("message") or upd.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    user = msg.get("from") or {}
    user_id = user.get("id")
    text = (msg.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}

    # /start
    if text == "/start":
        send_message(
            chat_id,
            "Привет! Я бот-память.\n"
            "Команды:\n"
            "/help — справка\n"
            "/use <Project> — выбрать активный проект (например, /use Innertrade)\n"
        )
        return {"ok": True}

    # /help
    if text == "/help":
        send_message(
            chat_id,
            "Справка:\n"
            "/use <Project> — активировать проект для этого чата.\n"
            "Дальше добавим /find, /pin и поиск по материалам."
        )
        return {"ok": True}

    # /use <Project>
    if text.startswith("/use"):
        parts = text.split(maxsplit=1)
        if len(parts) == 1 or not parts[1].strip():
            send_message(chat_id, "Укажи проект: например, /use Innertrade")
            return {"ok": True}
        project = parts[1].strip()

        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_context (chat_id, user_id, project)
                VALUES (%s, %s, %s)
                ON CONFLICT (chat_id, user_id)
                DO UPDATE SET project = EXCLUDED.project, created_at = now();
                """,
                (chat_id, user_id, project),
            )
        send_message(chat_id, f"Проект активирован: {project}")
        return {"ok": True}

    # Фоллбэк
    send_message(chat_id, "Команда не распознана. Введи /help")
    return {"ok": True}

# --- Entrypoint ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
