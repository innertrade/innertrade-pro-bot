import os
import urllib.request
import urllib.parse
from flask import Flask, request
import psycopg
from psycopg import OperationalError

# --- ENV / Const --------------------------------------------------------------
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
DATABASE_URL   = os.environ["DATABASE_URL"]
WEBHOOK_PATH   = f"/webhook/{TELEGRAM_TOKEN}"

# --- Flask app ----------------------------------------------------------------
app = Flask(__name__)

# --- DB helpers (psycopg v3, автопереподключение + keepalive) -----------------
def _new_conn():
    return psycopg.connect(
        DATABASE_URL,
        sslmode="require",
        autocommit=True,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )

def init_db(conn):
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
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ix_chat_context_chat_user
              ON chat_context (chat_id, user_id);
        """)

def get_conn():
    if not hasattr(app, "_db_conn") or app._db_conn.closed:
        app._db_conn = _new_conn()
        init_db(app._db_conn)
        return app._db_conn
    try:
        with app._db_conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        return app._db_conn
    except OperationalError:
        app._db_conn = _new_conn()
        init_db(app._db_conn)
        return app._db_conn

def get_active_project(conn, chat_id, user_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT project
            FROM chat_context
            WHERE chat_id=%s AND user_id=%s
            ORDER BY id DESC
            LIMIT 1;
        """, (chat_id, user_id))
        row = cur.fetchone()
    return row[0] if row else None

# --- Telegram helper ----------------------------------------------------------
def send_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()

# --- Health -------------------------------------------------------------------
@app.get("/health")
def health():
    info = {"ok": True, "service": "innertrade-memory-bot", "db": False}
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT now();")
            cur.fetchone()
        info["db"] = True
    except Exception as e:
        info["error"] = str(e)[:200]
    return info

# --- Webhook ------------------------------------------------------------------
@app.post(WEBHOOK_PATH)
def webhook():
    chat_id = None  # чтобы except не падал на NameError
    try:
        upd = request.get_json(force=True, silent=True) or {}
        msg = upd.get("message") or upd.get("edited_message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        user = msg.get("from") or {}
        user_id = user.get("id")
        text_raw = (msg.get("text") or "").strip()

        if not chat_id or not text_raw:
            return {"ok": True}

        parts = text_raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        # /start
        if cmd == "/start":
            send_message(chat_id,
                "Привет! Я бот-память.\n"
                "Команды:\n"
                "/help — справка\n"
                "/use <Project> — выбрать активный проект (например, /use Innertrade)\n"
                "/find <запрос> — найти документы в активном проекте"
            )
            return {"ok": True}

        # /help
        if cmd == "/help":
            send_message(chat_id,
                "Справка:\n"
                "/use <Project> — активировать проект для этого чата\n"
                "/find <запрос> — поиск по названию и содержимому актуальных версий\n"
                "Скоро: /pin и ответы по контексту (RAG)."
            )
            return {"ok": True}

        # /use <Project>
        if cmd == "/use":
            if not arg:
                send_message(chat_id, "Укажи проект: например, /use Innertrade")
                return {"ok": True}
            project = arg
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_context (chat_id, user_id, project)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (chat_id, user_id)
                    DO UPDATE SET project = EXCLUDED.project, created_at = now();
                """, (chat_id, user_id, project))
            send_message(chat_id, f"Проект активирован: {project}")
            return {"ok": True}

        # /find <query>
        if cmd == "/find":
            if not arg:
                send_message(chat_id, "Формат: /find ключевые слова")
                return {"ok": True}
            query = arg
            conn = get_conn()
            project = get_active_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use Innertrade")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT title, type, version, doc_version_id,
                           LEFT(content_md, 160) AS preview
                    FROM vw_latest_versions
                    WHERE project = %s
                      AND (title ILIKE '%%'||%s||'%%'
                       OR  content_md ILIKE '%%'||%s||'%%')
                    ORDER BY created_at DESC
                    LIMIT 5;
                """, (project, query, query))
                rows = cur.fetchall()

            if not rows:
                send_message(chat_id, "Ничего не нашёл. Попробуй другие слова.")
                return {"ok": True}

            lines = []
            for i, (title, doc_type, version, ver_id, preview) in enumerate(rows, 1):
                preview = (preview or "").replace("\n", " ")
                lines.append(f"{i}) {title} [{doc_type} • {version}] (id:{ver_id})\n↳ {preview}")
            reply = "Нашёл:\n" + "\n\n".join(lines)
            if len(reply) > 3800:
                reply = reply[:3800] + "…"
            send_message(chat_id, reply)
            return {"ok": True}

        # нераспознанная команда/текст
        if text_raw.startswith("/"):
            send_message(chat_id, "Команда не распознана. /help")
        else:
            send_message(chat_id, "Это не команда. Используй /help")
        return {"ok": True}

    except Exception as e:
        try:
            if chat_id is not None:
                send_message(chat_id, f"Ошибка: {e}")
        except Exception:
            pass
        return {"ok": True}

# --- Entrypoint ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
