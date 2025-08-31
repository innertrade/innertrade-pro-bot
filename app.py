@app.post(WEBHOOK_PATH)
def webhook():
    chat_id = None  # объявим заранее, чтобы не словить NameError в except
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

        if cmd == "/start":
            send_message(chat_id,
                "Привет! Я бот-память.\n"
                "Команды:\n"
                "/help — справка\n"
                "/use <Project> — выбрать активный проект\n"
                "/find <запрос> — найти документы"
            )
            return {"ok": True}

        if cmd == "/help":
            send_message(chat_id,
                "Справка:\n"
                "/use <Project>\n"
                "/find <запрос>"
            )
            return {"ok": True}

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
                      AND (title ILIKE '%%'||%s||'%%' OR content_md ILIKE '%%'||%s||'%%')
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

        # не команда — подсказываем
        if text_raw.startswith("/"):
            send_message(chat_id, "Команда не распознана. /help")
        else:
            send_message(chat_id, "Это не команда. Используй /help")
        return {"ok": True}

    except Exception as e:
        # отладочное сообщение пользователю, если chat_id успели распарсить
        try:
            if chat_id is not None:
                send_message(chat_id, f"Ошибка: {e}")
        except Exception:
            pass
        # обязательно отвечаем 200, чтобы Telegram не ретраил
        return {"ok": True}
