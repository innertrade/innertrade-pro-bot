import os
from flask import Flask, request

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"

app = Flask(__name__)

@app.get("/health")
def health():
    return {"ok": True}

@app.post(WEBHOOK_PATH)
def webhook():
    update = request.get_json(force=True, silent=True) or {}
    chat_id = update.get("message", {}).get("chat", {}).get("id")
    if chat_id:
        send_message(chat_id, "Привет! Бот работает 🎉")
    return {"ok": True}

def send_message(chat_id, text):
    import urllib.request, urllib.parse
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req) as r:
        r.read()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
