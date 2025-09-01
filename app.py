import os
import time
import json
import urllib.request
import urllib.parse
from flask import Flask, request
import psycopg
from psycopg import OperationalError
from openai import OpenAI

# --- ENV / Const --------------------------------------------------------------
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
DATABASE_URL     = os.environ["DATABASE_URL"]
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_PROJECT  = os.environ.get("DEFAULT_PROJECT", "").strip()  # опционально
WEBHOOK_PATH     = f"/webhook/{TELEGRAM_TOKEN}"

LLM_MODEL = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-small"  # 1536 dims
ASSISTANT_SYSTEM = """Ты — Ассистент с внешней памятью (RAG).
Отвечай кратко, по делу, опираясь ТОЛЬКО на предоставленный контекст.
Если в контексте нет ответа — прямо скажи, чего не хватает, и предложи уточнить запрос или добавить материалы.
НЕЛЬЗЯ давать финансовые/медицинские рекомендации; это не инвестиционный совет.
Язык ответа: русский."""

# --- Flask & OpenAI -----------------------------------------------------------
app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# простая «аварийная сигнализация»: если словили 429 — 10 минут работаем без LLM
LLM_COOLDOWN_SEC = 600
_last_llm_fail_ts = 0.0

def llm_mark_fail():
    global _last_llm_fail_ts
    _last_llm_fail_ts = time.time()

def llm_allowed():
    return client is not None and ((time.time() - _last_llm_fail_ts) > LLM_COOLDOWN_SEC)

# --- DB helpers ---------------------------------------------------------------
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
        # служебные таблицы
        cur.execute("CREATE TABLE IF NOT EXISTS chat_context (id SERIAL PRIMARY KEY, chat_id BIGINT NOT NULL, user_id BIGINT NOT NULL, project TEXT NOT NULL, created_at TIMESTAMP DEFAULT now());")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_chat_context_chat_user ON chat_context (chat_id, user_id);")
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
        cur.execute("CREATE INDEX IF NOT EXISTS ix_pins_chat_user_proj ON pins (chat_id, user_id, project);")
        # контентные таблицы (минимальная схема)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS docs (
              id SERIAL PRIMARY KEY,
              project TEXT NOT NULL,
              type TEXT NOT NULL,
              title TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT now()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS ix_docs_proj ON docs (project);")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_versions (
              id SERIAL PRIMARY KEY,
              doc_id INT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
              version TEXT NOT NULL,
              content_md TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT now()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS ix_doc_versions_doc ON doc_versions (doc_id);")
        # pgvector индекс для семантики (не обязателен для текущей задачи)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_chunks (
              id BIGSERIAL PRIMARY KEY,
              doc_version_id INT NOT NULL REFERENCES doc_versions(id) ON DELETE CASCADE,
              chunk_no INT NOT NULL,
              content TEXT NOT NULL,
              embedding VECTOR(1536),
              created_at TIMESTAMP DEFAULT now(),
              UNIQUE (doc_version_id, chunk_no)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS ix_doc_chunks_docver ON doc_chunks (doc_version_id);")
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding_hnsw ON doc_chunks USING hnsw (embedding vector_l2_ops);")
        except Exception:
            pass
        # materialized view-стайл: создадим представление latest-версий, если его нет
        # Примечание: CREATE VIEW IF NOT EXISTS доступен в PG15+, в Neon обычно ок.
        try:
            cur.execute("""
                CREATE VIEW IF NOT EXISTS vw_latest_versions AS
                SELECT DISTINCT ON (d.id)
                    d.project,
                    d.type,
                    d.title,
                    dv.id AS doc_version_id,
                    dv.version,
                    dv.content_md,
                    dv.created_at
                FROM docs d
                JOIN doc_versions dv ON dv.doc_id = d.id
                ORDER BY d.id, dv.created_at DESC;
            """)
        except Exception:
            # если уже существует с другим определением — пропустим, код ниже умеет без вьюхи
            pass

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
        cur.execute("SELECT project FROM chat_context WHERE chat_id=%s AND user_id=%s ORDER BY id DESC LIMIT 1;", (chat_id, user_id))
        row = cur.fetchone()
    return row[0] if row else None

def resolve_project(conn, chat_id, user_id):
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
        cur.execute("SELECT DISTINCT project FROM docs ORDER BY project;")
        rows = cur.fetchall()
        if rows:
            return [r[0] for r in rows]
        cur.execute("SELECT DISTINCT project FROM chat_context ORDER BY project;")
        rows = cur.fetchall()
    return [r[0] for r in rows]

# --- Telegram helpers ---------------------------------------------------------
def send_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()

def send_long_message(chat_id: int, text: str, chunk_size: int = 3800):
    text = text or ""
    i = 0
    n = len(text)
    if n == 0:
        send_message(chat_id, "(пусто)")
        return
    while i < n:
        send_message(chat_id, text[i:i+chunk_size])
        i += chunk_size

def telegram_get_file_text(file_id: str) -> str:
    meta_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile?file_id={file_id}"
    with urllib.request.urlopen(meta_url, timeout=10) as r:
        meta = json.loads(r.read().decode("utf-8"))
    file_path = meta["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    with urllib.request.urlopen(file_url, timeout=30) as r:
        data = r.read()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", "ignore")

def get_text_from_reply(msg: dict):
    """Берём текст из сообщения-ответа: либо text/caption, либо скачиваем .md/.txt документ."""
    reply = msg.get("reply_to_message")
    if not reply:
        return None, "Эта команда должна быть ответом на сообщение с текстом или на .md/.txt файл."
    txt = (reply.get("text") or reply.get("caption") or "").strip()
    if txt:
        return txt, None
    doc = reply.get("document")
    if doc:
        fname = (doc.get("file_name") or "").lower()
        if not (fname.endswith(".md") or fname.endswith(".txt") or fname.endswith(".markdown")):
            return None, "Поддерживаются только файлы .md / .txt"
        try:
            text = telegram_get_file_text(doc.get("file_id"))
            return text, None
        except Exception as e:
            return None, f"Не удалось скачать файл: {e}"
    return None, "В ответе нет текста или поддерживаемого файла."

# --- Text chunking & embeddings (необязательное, работает только при наличии ключа) ---
def chunk_text(text: str, max_len: int = 900, overlap: int = 120):
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_len, n)
        chunk = text[i:j]
        cut = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "))
        if cut > max_len * 0.5:
            chunk = chunk[:cut+1]; j = i + len(chunk)
        chunks.append(chunk.strip())
        if j >= n:
            break
        i = max(0, j - overlap)
    return [c for c in chunks if c]

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not llm_allowed():
        raise RuntimeError("LLM cooldown or no API key")
    try:
        resp = client.embeddings.create(model=EMB_MODEL, input=texts)
        return [item.embedding for item in resp.data]
    except Exception as e:
        if "insufficient_quota" in str(e) or "429" in str(e):
            llm_mark_fail()
        raise

def vector_to_sql_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

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
        return cur.fetchall()

def semantic_search_topk(conn, project: str, query: str, k: int = 6):
    try:
        q_emb = embed_texts([query])[0]
    except Exception:
        return []
    q_vec = vector_to_sql_literal(q_emb)
    sql = """
        WITH top AS (
          SELECT ch.doc_version_id, MIN(ch.embedding <=> %s::vector) AS dist
          FROM doc_chunks ch
          JOIN doc_versions dv ON dv.id = ch.doc_version_id
          JOIN docs d ON d.id = dv.doc_id
          WHERE d.project = %s AND ch.embedding IS NOT NULL
          GROUP BY ch.doc_version_id
          ORDER BY dist ASC
          LIMIT %s
        )
        SELECT dv.id, d.title, dv.version, dv.content_md
        FROM top
        JOIN doc_versions dv ON dv.id = top.doc_version_id
        JOIN docs d ON d.id = dv.doc_id
        ORDER BY top.dist ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (q_vec, project, k))
        return cur.fetchall()

def fallback_ilike_search(conn, project, query, exclude_ids=None, limit=4):
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
        try:
            cur.execute(sql, params)
            return cur.fetchall()
        except Exception:
            # если вьюхи нет — берём последний dv для каждого doc вручную
            cur.execute("""
                WITH latest AS (
                  SELECT dv.*
                  FROM doc_versions dv
                  JOIN (
                    SELECT doc_id, MAX(created_at) AS mx
                    FROM doc_versions GROUP BY doc_id
                  ) x ON x.doc_id = dv.doc_id AND x.mx = dv.created_at
                  JOIN docs d ON d.id = dv.doc_id
                  WHERE d.project = %s
                )
                SELECT l.id AS doc_version_id, d.title, l.version, l.content_md
                FROM latest l
                JOIN docs d ON d.id = l.doc_id
                WHERE (d.title ILIKE '%%' || %s || '%%' OR l.content_md ILIKE '%%' || %s || '%%')
                ORDER BY l.created_at DESC
                LIMIT %s;
            """, (project, query, query, limit))
            return cur.fetchall()

def build_context_blocks(rows, max_chars=1800):
    blocks = []
    for (ver_id, title, version, content) in rows:
        snippet = (content or "")[:max_chars]
        blocks.append(f"[id:{ver_id}] {title} • {version}\n{snippet}")
    return blocks

def llm_answer_or_fallback(question: str, blocks: list[str]) -> str:
    if not blocks:
        return "По контексту ничего не нашлось. Добавь материалов или уточни запрос."
    ctx = "\n\n---\n\n".join(blocks)
    if llm_allowed():
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.3,
                messages=[
                    {"role":"system","content":ASSISTANT_SYSTEM},
                    {"role":"user","content":(
                        f"Вопрос пользователя: {question}\n\n"
                        f"Контекст (фрагменты документов):\n{ctx}\n\n"
                        "Сформируй краткий ответ по контексту. Если ответа нет — перечисли, чего не хватает."
                    )}
                ],
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "insufficient_quota" in str(e) or "429" in str(e):
                llm_mark_fail()
    # fallback: просто выдержки
    head = []
    for b in blocks[:3]:
        lines = b.splitlines()
        head.append("\n".join(lines[:6]))
    return "⚠️ Модель недоступна. Ниже релевантные выдержки:\n\n" + "\n\n---\n\n".join(head)

def summarize_doc(title: str, version: str, content: str, bullets: int = 5) -> str:
    bullets = max(3, min(12, bullets))
    if llm_allowed():
        try:
            prompt = (
                f"Сделай краткое резюме документа в виде {bullets} пунктов.\n"
                f"Заголовок: {title} • {version}\n\nТекст:\n{content[:8000]}\n\n"
                "Требования: кратко, по делу, без воды; не выдумывай факты."
            )
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.2,
                messages=[{"role":"system","content":ASSISTANT_SYSTEM},{"role":"user","content":prompt}],
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "insufficient_quota" in str(e) or "429" in str(e):
                llm_mark_fail()
    # fallback без LLM
    text = (content or "").strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    points = []
    for p in paras[:bullets]:
        points.append("- " + (p.replace("\n", " ")[:220]))
    return "\n".join(points) if points else "⚠️ Недостаточно текста для резюме."

def context_preview(conn, chat_id, user_id, project, question, limit=3):
    pinned = fetch_pinned_context(conn, chat_id, user_id, project, limit=limit)
    pinned_ids = [r[0] for r in pinned]
    sem = semantic_search_topk(conn, project, question or " ", k=limit) if question else []
    sem_ids = [r[0] for r in sem]
    fb  = fallback_ilike_search(conn, project, question or " ", exclude_ids=pinned_ids+sem_ids, limit=limit) if question else []
    def fmt(rows, label):
        if not rows: return f"{label}: —"
        lines = []
        for (ver_id, title, version, _content) in rows:
            lines.append(f"[id:{ver_id}] {title} • {version}")
        return f"{label}:\n" + "\n".join(lines)
    return fmt(pinned, "Пины") + "\n\n" + fmt(sem, "Семантика") + "\n\n" + fmt(fb, "Fallback")

# --- Health -------------------------------------------------------------------
@app.get("/health")
def health():
    info = {"ok": True, "service": "assistant-memory-bot", "db": False, "llm_mode": "full" if llm_allowed() else "degraded"}
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

        if not chat_id or not (text_raw or msg.get("document") or msg.get("caption")):
            return {"ok": True}

        parts = (text_raw or "").split(maxsplit=1)
        cmd = parts[0].lower() if (text_raw and parts) else ""
        arg = parts[1].strip() if (text_raw and len(parts) > 1) else ""

        # /start
        if cmd == "/start":
            send_message(chat_id,
                "Привет! Я Ассистент с внешней памятью.\n"
                "Команды:\n"
                "/help — подсказка\n"
                "/use <Project> — выбрать проект (например, /use Innertrade)\n"
                "/projects — список проектов\n"
                "/find <запрос> — поиск по документам\n"
                "/docs — список документов проекта\n"
                "/new <type> | <title> — создать документ\n"
                "/vers <doc_id> — версии документа\n"
                "/add <doc_id> | <version> — ДОБАВИТЬ версию (реплай на текст или .md/.txt)\n"
                "/delver <doc_version_id> — удалить версию\n"
                "/export <doc_id> — показать последнюю версию целиком\n"
                "/pin <id> [note], /pins, /unpin <id>, /show <id>\n"
                "/context [вопрос], /reset [pins|project|all]\n"
                "/summ <id> [n], /index <id>\n"
                "/ask <вопрос> — RAG ответ (пины → семантика → fallback)"
            )
            return {"ok": True}

        # /help
        if cmd == "/help":
            send_message(chat_id,
                "Контент:\n"
                "/docs, /new <type> | <title>, /vers <doc_id>, /add <doc_id> | <version> (реплай на текст/.md/.txt), /delver <ver_id>, /export <doc_id>\n\n"
                "Контекст и ответы:\n"
                "/find, /pin, /pins, /unpin, /show, /context, /reset [pins|project|all], /summ, /index, /ask"
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

        # /docs — список документов проекта
        if cmd == "/docs":
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project>")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, type, title, to_char(created_at,'YYYY-MM-DD HH24:MI')
                    FROM docs
                    WHERE project=%s
                    ORDER BY created_at DESC
                    LIMIT 20;
                """, (project,))
                rows = cur.fetchall()
            if not rows:
                send_message(chat_id, "Документов пока нет. Создай: /new <type> | <title>")
                return {"ok": True}
            lines = [f"{i}) [id:{doc_id}] {typ} — {title} ({created})"
                     for i,(doc_id,typ,title,created) in enumerate(rows,1)]
            send_message(chat_id, "Документы:\n" + "\n".join(lines))
            return {"ok": True}

        # /new <type> | <title>
        if cmd == "/new":
            if "|" not in arg:
                send_message(chat_id, "Формат: /new <type> | <title>")
                return {"ok": True}
            typ, title = [x.strip() for x in arg.split("|",1)]
            if not typ or not title:
                send_message(chat_id, "Укажи тип и заголовок: /new <type> | <title>")
                return {"ok": True}
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project>")
                return {"ok": True}
            with conn.cursor() as cur:
                # если уже есть такой документ — вернём id
                cur.execute("""
                    SELECT id FROM docs WHERE project=%s AND type=%s AND title=%s LIMIT 1;
                """, (project, typ, title))
                row = cur.fetchone()
                if row:
                    send_message(chat_id, f"Документ уже существует: [id:{row[0]}] {typ} — {title}")
                    return {"ok": True}
                cur.execute("""
                    INSERT INTO docs (project, type, title) VALUES (%s,%s,%s) RETURNING id;
                """, (project, typ, title))
                doc_id = cur.fetchone()[0]
            send_message(chat_id, f"Создан документ [id:{doc_id}] {typ} — {title}\nТеперь добавь версию: /add {doc_id} | v0.1.0 (реплай на текст или .md/.txt)")
            return {"ok": True}

        # /vers <doc_id>
        if cmd == "/vers":
            try:
                doc_id = int(arg)
            except Exception:
                send_message(chat_id, "Формат: /vers <doc_id>")
                return {"ok": True}
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT dv.id, dv.version, LEFT(dv.content_md, 160), to_char(dv.created_at,'YYYY-MM-DD HH24:MI')
                    FROM doc_versions dv
                    WHERE dv.doc_id=%s
                    ORDER BY dv.created_at DESC
                    LIMIT 20;
                """, (doc_id,))
                rows = cur.fetchall()
            if not rows:
                send_message(chat_id, "Версий пока нет. Добавь: /add <doc_id> | <version> (реплай на текст/.md/.txt)")
                return {"ok": True}
            lines = [f"{i}) [ver:{vid}] {ver} — {created}\n↳ {prev.replace(chr(10),' ')}"
                     for i,(vid,ver,prev,created) in enumerate(rows,1)]
            send_message(chat_id, "Версии:\n" + "\n\n".join(lines))
            return {"ok": True}

        # /add <doc_id> | <version>  (контент берём из сообщения, на которое ты отвечаешь)
        if cmd == "/add":
            if "|" not in arg:
                send_message(chat_id, "Формат: /add <doc_id> | <version>\nВажно: команда должна быть ответом (reply) на сообщение с текстом или .md/.txt файлом.")
                return {"ok": True}
            left, version = [x.strip() for x in arg.split("|",1)]
            try:
                doc_id = int(left)
            except Exception:
                send_message(chat_id, "doc_id должен быть числом. Пример: /add 12 | v0.1.0")
                return {"ok": True}
            content, err = get_text_from_reply(msg)
            if err:
                send_message(chat_id, err)
                return {"ok": True}
            conn = get_conn()
            with conn.cursor() as cur:
                # Проверим, что документ существует
                cur.execute("SELECT title FROM docs WHERE id=%s LIMIT 1;", (doc_id,))
                row = cur.fetchone()
                if not row:
                    send_message(chat_id, "Документ не найден. Сначала создай: /new <type> | <title>")
                    return {"ok": True}
                title = row[0]
                # пишем версию
                cur.execute("""
                    INSERT INTO doc_versions (doc_id, version, content_md)
                    VALUES (%s,%s,%s)
                    RETURNING id;
                """, (doc_id, version, content))
                ver_id = cur.fetchone()[0]
                # чистим старые чанки на всякий случай
                cur.execute("DELETE FROM doc_chunks WHERE doc_version_id=%s;", (ver_id,))
            send_message(chat_id, f"Добавлена версия [id:{ver_id}] для документа {title} ({version}).\nИндексацию семантики можно сделать позже: /index {ver_id}")
            return {"ok": True}

        # /delver <doc_version_id>
        if cmd == "/delver":
            try:
                ver_id = int(arg)
            except Exception:
                send_message(chat_id, "Формат: /delver <doc_version_id>")
                return {"ok": True}
            conn = get_conn()
            with conn.cursor() as cur:
                # снимем пины с этой версии
                cur.execute("DELETE FROM pins WHERE doc_version_id=%s;", (ver_id,))
                # удалим чанки (каскад есть, но на всякий)
                cur.execute("DELETE FROM doc_chunks WHERE doc_version_id=%s;", (ver_id,))
                # удалим версию
                cur.execute("DELETE FROM doc_versions WHERE id=%s;", (ver_id,))
                deleted = cur.rowcount
            send_message(chat_id, "Версия удалена." if deleted else "Такой версии не найдено.")
            return {"ok": True}

        # /export <doc_id> — показать последнюю версию целиком
        if cmd == "/export":
            try:
                doc_id = int(arg)
            except Exception:
                send_message(chat_id, "Формат: /export <doc_id>")
                return {"ok": True}
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.title, dv.version, dv.content_md
                    FROM doc_versions dv
                    JOIN docs d ON d.id = dv.doc_id
                    WHERE dv.doc_id=%s
                    ORDER BY dv.created_at DESC
                    LIMIT 1;
                """, (doc_id,))
                row = cur.fetchone()
            if not row:
                send_message(chat_id, "У документа нет версий.")
                return {"ok": True}
            title, version, content = row
            send_long_message(chat_id, f"{title} • {version}\n\n{content or ''}")
            return {"ok": True}

        # /find (как было)
        if cmd == "/find":
            if not arg:
                send_message(chat_id, "Формат: /find ключевые слова")
                return {"ok": True}
            query = arg
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
                return {"ok": True}
            with conn.cursor() as cur:
                try:
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
                except Exception:
                    # без вьюхи
                    cur.execute("""
                        WITH latest AS (
                          SELECT dv.*
                          FROM doc_versions dv
                          JOIN (
                            SELECT doc_id, MAX(created_at) AS mx
                            FROM doc_versions GROUP BY doc_id
                          ) x ON x.doc_id = dv.doc_id AND x.mx = dv.created_at
                          JOIN docs d ON d.id = dv.doc_id
                          WHERE d.project = %s
                        )
                        SELECT d.title, d.type, l.version, l.id AS doc_version_id,
                               LEFT(l.content_md, 160) AS preview
                        FROM latest l
                        JOIN docs d ON d.id = l.doc_id
                        WHERE (d.title ILIKE '%%'||%s||'%%' OR l.content_md ILIKE '%%'||%s||'%%')
                        ORDER BY l.created_at DESC
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

        # пины и просмотр
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
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
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

        if cmd == "/pins":
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
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
            for i, (_pid, ver_id, title, version, note) in enumerate(rows, 1):
                extra = f" — {note}" if note else ""
                lines.append(f"{i}) {title} • {version} (id:{ver_id}){extra}")
            send_message(chat_id, "Закреплено:\n" + "\n".join(lines))
            return {"ok": True}

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
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pins WHERE chat_id=%s AND user_id=%s AND project=%s AND doc_version_id=%s;", (chat_id, user_id, project, ver_id))
            send_message(chat_id, "Снял закреп.")
            return {"ok": True}

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
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.title, dv.version, dv.content_md
                    FROM doc_versions dv
                    JOIN docs d ON d.id = dv.doc_id
                    WHERE dv.id=%s
                    LIMIT 1;
                """, (ver_id,))
                row = cur.fetchone()
            if not row:
                send_message(chat_id, "Версия не найдена.")
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
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
                return {"ok": True}
            question = arg or " "
            preview = context_preview(conn, chat_id, user_id, project, question, limit=3)
            send_message(chat_id, "Контекст для ответа:\n" + preview)
            return {"ok": True}

        # /reset [pins|project|all]
        if cmd == "/reset":
            mode = (arg or "").strip().lower()
            conn = get_conn()
            if mode == "all":
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM pins WHERE chat_id=%s AND user_id=%s;", (chat_id, user_id))
                    cur.execute("DELETE FROM chat_context WHERE chat_id=%s AND user_id=%s;", (chat_id, user_id))
                send_message(chat_id, "Полный сброс: пины очищены, проект сброшен.")
                return {"ok": True}
            project = get_active_project(conn, chat_id, user_id)
            if mode == "pins":
                if not project:
                    send_message(chat_id, "Нет активного проекта. Сначала /use <Project> или /reset project.")
                    return {"ok": True}
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM pins WHERE chat_id=%s AND user_id=%s AND project=%s;", (chat_id, user_id, project))
                send_message(chat_id, "Все закрепы по текущему проекту сняты.")
                return {"ok": True}
            if mode == "project":
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chat_context WHERE chat_id=%s AND user_id=%s;", (chat_id, user_id))
                send_message(chat_id, "Активный проект сброшен. Укажи заново: /use <Project>")
                return {"ok": True}
            send_message(chat_id, "Формат: /reset [pins|project|all]")
            return {"ok": True}

        # /summ <doc_version_id> [n]
        if cmd == "/summ":
            if not arg:
                send_message(chat_id, "Формат: /summ <doc_version_id> [кол-во_пунктов]\nНапр.: /summ 1 7")
                return {"ok": True}
            parts2 = arg.split()
            try:
                ver_id = int(parts2[0])
            except ValueError:
                send_message(chat_id, "id должен быть числом (см. /find).")
                return {"ok": True}
            bullets = 5
            if len(parts2) > 1:
                try:
                    bullets = int(parts2[1])
                except ValueError:
                    bullets = 5
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.title, dv.version, dv.content_md
                    FROM doc_versions dv
                    JOIN docs d ON d.id = dv.doc_id
                    WHERE dv.id=%s
                    LIMIT 1;
                """, (ver_id,))
                row = cur.fetchone()
            if not row:
                send_message(chat_id, "Версия не найдена.")
                return {"ok": True}
            title, version, content = row
            summary = summarize_doc(title, version, content or "", bullets)
            summary = f"{title} • {version}\n\n{summary}\n\nИсточник: [id:{ver_id}] {title} • {version}"
            send_message(chat_id, summary)
            return {"ok": True}

        # /index <doc_version_id> — индексация (семантика, при наличии ключа)
        if cmd == "/index":
            if not arg:
                send_message(chat_id, "Формат: /index <doc_version_id>\nИндексация создаёт чанки и эмбеддинги для семантического поиска.")
                return {"ok": True}
            try:
                ver_id = int(arg)
            except ValueError:
                send_message(chat_id, "id должен быть числом (см. /find).")
                return {"ok": True}
            conn = get_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.title, dv.version, dv.content_md
                    FROM doc_versions dv
                    JOIN docs d ON d.id = dv.doc_id
                    WHERE dv.id=%s
                    LIMIT 1;
                """, (ver_id,))
                row = cur.fetchone()
            if not row:
                send_message(chat_id, "Версия не найдена.")
                return {"ok": True}
            title, version, content = row
            chunks = chunk_text(content or "", max_len=900, overlap=120)
            if not chunks:
                send_message(chat_id, "Пустой документ: нечего индексировать.")
                return {"ok": True}
            with conn.cursor() as cur:
                cur.execute("DELETE FROM doc_chunks WHERE doc_version_id=%s;", (ver_id,))
            inserted = 0
            try:
                batch = 64
                for s in range(0, len(chunks), batch):
                    part = chunks[s:s+batch]
                    embs = embed_texts(part)  # может кинуть при пустом ключе/квоте
                    with conn.cursor() as cur:
                        for i, (chunk, emb) in enumerate(zip(part, embs), start=s):
                            vec = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
                            cur.execute("INSERT INTO doc_chunks (doc_version_id, chunk_no, content, embedding) VALUES (%s,%s,%s,%s::vector);", (ver_id, i, chunk, vec))
                            inserted += 1
                send_message(chat_id, f"Проиндексировано: {inserted} чанков (семантика активна) для {title} • {version} (id:{ver_id}).")
            except Exception as e:
                # fallback — сохраняем без эмбеддингов, чтобы потом можно было переиндексировать
                with conn.cursor() as cur:
                    for i, chunk in enumerate(chunks):
                        cur.execute("INSERT INTO doc_chunks (doc_version_id, chunk_no, content, embedding) VALUES (%s,%s,%s,NULL);", (ver_id, i, chunk))
                        inserted += 1
                send_message(chat_id, f"Сохранил {inserted} чанков без эмбеддингов (нет доступа к API). Позже повтори /index {ver_id}.")
            return {"ok": True}

        # /ask — RAG (пины → семантика → fallback) + источники, с мягким фолбэком без LLM
        if cmd == "/ask":
            if not arg or len(arg.strip()) < 2:
                send_message(chat_id, "Сформулируй вопрос чуть конкретнее 🙏")
                return {"ok": True}
            question = arg.strip()
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
                return {"ok": True}
            pinned = fetch_pinned_context(conn, chat_id, user_id, project, limit=4)
            pinned_ids = [r[0] for r in pinned]
            sem = semantic_search_topk(conn, project, question, k=6)
            sem_ids = [r[0] for r in sem]
            fb  = fallback_ilike_search(conn, project, question, exclude_ids=pinned_ids+sem_ids, limit=4)
            rows = pinned + sem + fb
            blocks = build_context_blocks(rows)
            answer = llm_answer_or_fallback(question, blocks)
            if rows:
                src_lines = [f"- [id:{vid}] {t} • {ver}" for (vid, t, ver, _c) in rows[:8]]
                answer += "\n\nИсточники:\n" + "\n".join(src_lines)
            send_message(chat_id, answer)
            return {"ok": True}

        # Свободный текст → как /ask
        if not text_raw.startswith("/"):
            question = text_raw.strip()
            if len(question) < 2:
                send_message(chat_id, "Сформулируй вопрос чуть конкретнее 🙏")
                return {"ok": True}
            conn = get_conn()
            project = resolve_project(conn, chat_id, user_id)
            if not project:
                send_message(chat_id, "Сначала выбери проект: /use <Project>.")
                return {"ok": True}
            pinned = fetch_pinned_context(conn, chat_id, user_id, project, limit=4)
            pinned_ids = [r[0] for r in pinned]
            sem = semantic_search_topk(conn, project, question, k=6)
            sem_ids = [r[0] for r in sem]
            fb  = fallback_ilike_search(conn, project, question, exclude_ids=pinned_ids+sem_ids, limit=4)
            rows = pinned + sem + fb
            blocks = build_context_blocks(rows)
            answer = llm_answer_or_fallback(question, blocks)
            if rows:
                src_lines = [f"- [id:{vid}] {t} • {ver}" for (vid, t, ver, _c) in rows[:8]]
                answer += "\n\nИсточники:\n" + "\n".join(src_lines)
            send_message(chat_id, answer)
            return {"ok": True}

        # нераспознанная команда
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
