import os
import urllib.request
import urllib.parse
from flask import Flask, request
import psycopg
from psycopg import OperationalError
from openai import OpenAI

# --- ENV / Const --------------------------------------------------------------
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
DATABASE_URL     = os.environ["DATABASE_URL"]
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
DEFAULT_PROJECT  = os.environ.get("DEFAULT_PROJECT", "").strip()  # опционально
WEBHOOK_PATH     = f"/webhook/{TELEGRAM_TOKEN}"

LLM_MODEL = "gpt-4o-mini"
ASSISTANT_SYSTEM = """Ты — Ассистент с внешней памятью (RAG).
Отвечай кратко, по делу, опираясь ТОЛЬКО на предоставленный контекст.
Если в контексте нет ответа — прямо скажи, чего не хватает, и предложи уточнить запрос или добавить материалы.
НЕЛЬЗЯ давать финансовые/медицинские рекомендации; это не инвестиционный совет.
Язык ответа: русский."""

# --- Flask app ----------------------------------------------------------------
app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pins (
              id SERIAL PRIMARY KEY,
              user_id BIGINT NOT NULL,
              chat_id BIGINT NOT NULL,
              project TEXT NOT NULL,
              doc_version_id INT NOT NULL,
              note TEXT,
              created_at TIMESTAMP DEFAULT now()
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS ix_pins_chat_user_proj
              ON pins (chat_id, user_id, project);
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

def resolve_project(conn, chat_id, user_id):
    """Возвращает активный проект; если нет — подставляет DEFAULT_PROJECT (если задан) и возвращает его."""
    proj = get_active_project(conn, chat_id, user_id)
    if proj:
        return proj
    if DEFAULT_PROJECT:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_context (chat_id, user_id, project)
                VALUES (%s,%s,%s)
                ON CONFLICT (chat_id, user_id)
                DO UPDATE SET project = EXCLUDED.project, created_at = now();
            """, (chat_id, user_id, DEFAULT_PROJECT))
        return DEFAULT_PROJECT
    return None

def list_projects(conn):
    with conn.cursor() as cur:
        # сначала пытаемся взять из docs (если она есть и наполнена)
        try:
            cur.execute("SELECT DISTINCT project FROM docs ORDER BY project;")
            rows = cur.fetchall()
            if rows:
                return [r[0] for r in rows]
        except Exception:
            pass
        # fallback — что когда-либо встречалось в chat_context
        cur.execute("SELECT DISTINCT project FROM chat_context ORDER BY project;")
        rows = cur.fetchall()
    return [r[0] for r in rows]

# --- Telegram helper ----------------------------------------------------------
def send_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()

# --- RAG helpers --------------------------------------------------------------
def fetch_pinned_context(conn, chat_id, user_id, project, limit=4):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT dv.id, d.title, dv.version, dv.content_md
            FROM pins p
            JOIN doc_versions dv ON dv.id = p.doc_version_id
            JOIN docs d ON d.id = dv.doc_id
            WHERE p.chat_id=%s AND p.user_id=%s AND p.project=%s
            ORDER BY p.created_at DESC
            LIMIT %s;
        """, (chat_id, user_id, project, limit))
        return cur.fetchall()  # [(ver_id, title, version, content_md), ...]

def fetch_search_context(conn, project, query, exclude_ids=None, limit=4):
    exclude_ids = list(exclude_ids or [])
    params = [project]
    sql = """
        SELECT doc_version_id, title, version, content_md
        FROM vw_latest_versions
        WHERE project = %s
    """
    if exclude_ids:
        placeholders = ",".join(["%s"] * len(exclude_ids))
        sql += f" AND doc_version_id NOT IN ({placeholders})"
        params.extend(exclude_ids)
    sql += """
          AND (title ILIKE '%%' || %s || '%%'
           OR  content_md ILIKE '%%' || %s || '%%')
        ORDER BY created_at DESC
        LIMIT %s;
    """
    params.extend([query, query, limit])
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def build_context_blocks(rows, max_chars=1800):
    blocks = []
    for (ver_id, title, version, content) in rows:
        snippet = (content or "")[:max_chars]
        blocks.append(f"[id:{ver_id}] {title} • {version}\n{snippet}")
    return blocks

def rag_answer(question: str, blocks: list[str]) -> str:
    ctx = "\n\n---\n\n".join(blocks) if blocks else "НЕТ КОНТЕКСТА."
    user_prompt = (
        f"Вопрос пользователя: {question}\n\n"
        f"Контекст (фрагменты документов):\n{ctx}\n\n"
        "Сформируй краткий ответ по контексту. Если ответа нет в контексте — скажи, что не хватает данных."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        messages=[
            {"role":"system","content":ASSISTANT_SYSTEM},
            {"role":"user","content":user_prompt}
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

# --- Context preview ----------------------------------------------------------
def context_preview(conn, chat_id, user_id, project, question, limit=3):
    pinned = fetch_pinned_context(conn, chat_id, user_id, project, limit=limit)
    pinned_ids = [r[0] for r in pinned]
    search = fetch_search_context(conn, project, question, exclude_ids=pinned_ids, limit=limit)

    def fmt(rows, label):
        if not rows:
            return f"{label}: —"
        lines = []
        for (ver_id, title, version, _content) in rows:
            lines.append(f"[id:{ver_id}] {title} • {version}")
        return f"{label}:\n" + "\n".join(lines)

    return fmt(pinned, "Пины") + "\n\n" + fmt(search, "Поиск")

# --- Health -------------------------------------------------------------------
@app.get("/health")
def health():
    info = {"ok": True, "service": "assistant-memory-bot", "db": False}
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
    chat_id = None
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
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        # /start
        if cmd == "/start":
            send_message(chat_id,
                "Привет! Я Ассистент с внешней памятью.\n"
                "Команды:\n"
                "/help — подсказка\n"
                "/use <Project> — выбрать проект (например, /use Innertrade)\n"
                "/projects — показать список известных проектов\n"
                "/find <запрос> — поиск по документам\n"
                "/pin <id> [note] — закрепить версию\n"
                "/pins — список закреплённого\n"
                "/unpin <id> — снять закреп\n"
                "/show <id> — показать начало контента\n"
                "/context [вопрос] — показать, какие фрагменты пойдут в ответ\n"
                "/reset [pins|project|all] — сброс закрепов/проекта\n"
                "/ask <вопрос> — ответ по памяти (сначала пины, потом поиск)"
            )
            return {"ok": True}

        # /help
        if cmd == "/help":
            send_message(chat_id,
                "Подсказка:\n"
                "/use <Project>\n"
                "/projects — список проектов\n"
                "/find <запрос>\n"
                "/pin <id> [note], /pins, /unpin <id>, /show <id>\n"
                "/context [вопрос] — показать источники\n"
                "/reset [pins|project|all]\n"
                "/ask <вопрос> — ответ на основе закреплённого и найденного контента."
            )
            return {"ok": True}

        # /projects
        if cmd == "/projects":
            conn = get_conn()
            projs = list_projects(conn)
            if not projs:
                send_message(chat_id, "Проектов пока не найдено. Укажи проект через /use <Project> или загрузите документы.")
                return {"ok": True}
            lines = "\n".join(f"• {p}" for p in projs[:50])
            send_message(chat_id, "Доступные проекты:\n" + lines + "\n\nВыбери: /use <Project>")
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
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
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

        # /pin <id> [note]
        if cmd == "/pin":
            if not arg:
                send_message(chat_id, "Формат: /pin <doc_version_id> [note]")
                return {"ok": True}
            parts2 = arg.split(maxsplit=1)
            try:
                ver_id = int(parts2[0])
            except ValueError:
                send_message(chat_id, "id должен быть числом (см. /find)")
                return {"ok": True}
            note = parts2[1].strip() if len(parts2) > 1 else None
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.title, dv.version
                    FROM doc_versions dv
                    JOIN docs d ON d.id = dv.doc_id
                    WHERE dv.id = %s AND d.project = %s
                    LIMIT 1;
                """, (ver_id, project))
                row = cur.fetchone()
            if not row:
                send_message(chat_id, "Версия не найдена в активном проекте.")
                return {"ok": True}
            title, version = row
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO pins (user_id, chat_id, project, doc_version_id, note)
                    VALUES (%s, %s, %s, %s, %s);
                """, (user_id, chat_id, project, ver_id, note))
            send_message(chat_id, f"Закрепил: {title} • {version} (id:{ver_id})")
            return {"ok": True}

        # /pins
        if cmd == "/pins":
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT p.id, p.doc_version_id, d.title, dv.version, COALESCE(p.note,'')
                    FROM pins p
                    JOIN doc_versions dv ON dv.id = p.doc_version_id
                    JOIN docs d ON d.id = dv.doc_id
                    WHERE p.chat_id=%s AND p.user_id=%s AND p.project=%s
                    ORDER BY p.created_at DESC
                    LIMIT 10;
                """, (chat_id, user_id, project))
                rows = cur.fetchall()
            if not rows:
                send_message(chat_id, "Пока ничего не закреплено. Используй /pin <id> из /find.")
                return {"ok": True}
            lines = []
            for i, (pid, ver_id, title, version, note) in enumerate(rows, 1):
                extra = f" — {note}" if note else ""
                lines.append(f"{i}) {title} • {version} (id:{ver_id}){extra}")
            send_message(chat_id, "Закреплено:\n" + "\n".join(lines))
            return {"ok": True}

        # /unpin <id>
        if cmd == "/unpin":
            if not arg:
                send_message(chat_id, "Формат: /unpin <doc_version_id>")
                return {"ok": True}
            try:
                ver_id = int(arg)
            except ValueError:
                send_message(chat_id, "id должен быть числом.")
                return {"ok": True}
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM pins
                    WHERE chat_id=%s AND user_id=%s AND project=%s AND doc_version_id=%s;
                """, (chat_id, user_id, project, ver_id))
                deleted = cur.rowcount
            send_message(chat_id, "Снял закреп." if deleted else "Такой закреп не найден.")
            return {"ok": True}

        # /show <id>
        if cmd == "/show":
            if not arg:
                send_message(chat_id, "Формат: /show <doc_version_id>")
                return {"ok": True}
            try:
                ver_id = int(arg)
            except ValueError:
                send_message(chat_id, "id должен быть числом.")
                return {"ok": True}
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.title, dv.version, dv.content_md
                    FROM doc_versions dv
                    JOIN docs d ON d.id = dv.doc_id
                    WHERE dv.id=%s AND d.project=%s
                    LIMIT 1;
                """, (ver_id, project))
                row = cur.fetchone()
            if not row:
                send_message(chat_id, "Версия не найдена в активном проекте.")
                return {"ok": True}
            title, version, content = row
            snippet = (content or "")[:800]
            send_message(chat_id, f"{title} • {version}\n\n{snippet}")
            return {"ok": True}

        # /context [вопрос]
        if cmd == "/context":
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
                return {"ok": True}
            question = arg or " "
            preview = context_preview(conn, chat_id, user_id, project, question, limit=3)
            send_message(chat_id, "Контекст для ответа:\n" + preview)
            return {"ok": True}

        # /reset [pins|project|all]
        if cmd == "/reset":
            mode = (arg or "").strip().lower()
            conn = get_conn()
            project = get_active_project(conn, chat_id, user_id)  # тут без автоподстановки

            if mode in ("pins", "all"):
                if not project:
                    send_message(chat_id, "Нет активного проекта. Сначала /use <Project> или /reset project.")
                    return {"ok": True}
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM pins
                        WHERE chat_id=%s AND user_id=%s AND project=%s;
                    """, (chat_id, user_id, project))
                if mode == "pins":
                    send_message(chat_id, "Все закрепы по текущему проекту сняты.")
                    return {"ok": True}

            if mode in ("project", "all"):
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM chat_context
                        WHERE chat_id=%s AND user_id=%s;
                    """, (chat_id, user_id))
                send_message(chat_id, "Активный проект сброшен. Укажи заново: /use <Project>")
                return {"ok": True}

            send_message(chat_id,
                "Формат: /reset [pins|project|all]\n"
                "• /reset pins — снять все закрепы в текущем проекте\n"
                "• /reset project — сбросить активный проект\n"
                "• /reset all — и проект, и закрепы"
            )
            return {"ok": True}

        # /ask <question> — RAG ответ
        if cmd == "/ask":
            if not arg or len(arg.strip()) < 2:
                send_message(chat_id, "Сформулируй вопрос чуть конкретнее 🙏")
                return {"ok": True}
            question = arg.strip()
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
                return {"ok": True}
            pinned = fetch_pinned_context(conn, chat_id, user_id, project, limit=4)
            pinned_ids = [r[0] for r in pinned]
            search = fetch_search_context(conn, project, question, exclude_ids=pinned_ids, limit=4)
            blocks = build_context_blocks(pinned) + build_context_blocks(search)
            answer = rag_answer(question, blocks)
            send_message(chat_id, answer)
            return {"ok": True}

        # Любой некомандный текст — трактуем как вопрос к Ассистенту (RAG)
        if not text_raw.startswith("/"):
            question = text_raw.strip()
            if len(question) < 2:
                send_message(chat_id, "Сформулируй вопрос чуть конкретнее 🙏")
                return {"ok": True}
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project> (или задай DEFAULT_PROJECT в ENV).")
                return {"ok": True}
            pinned = fetch_pinned_context(conn, chat_id, user_id, project, limit=4)
            pinned_ids = [r[0] for r in pinned]
            search = fetch_search_context(conn, project, question, exclude_ids=pinned_ids, limit=4)
            blocks = build_context_blocks(pinned) + build_context_blocks(search)
            answer = rag_answer(question, blocks)
            send_message(chat_id, answer)
            return {"ok": True}

        # Нераспознанная команда
        send_message(chat_id, "Команда не распознана. /help")
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
