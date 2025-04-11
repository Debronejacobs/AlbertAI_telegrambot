import os
import sqlite3
import re
import telebot
from dotenv import load_dotenv
from flask import Flask, request, abort
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event, Lock, RLock
from datetime import datetime, timedelta
from collections import defaultdict
import time
import logging

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_TOKEN = os.getenv("GEMINI_API_KEY")
WEBHOOK_URL_BASE = "https://albertaitelegrambot-production.up.railway.app"
WEBHOOK_URL_PATH = f"/{BOT_TOKEN}/"

bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

# Configure the Google AI SDK
genai.configure(api_key=GEMINI_TOKEN)

# Define the model and chat session
generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="You are Albert a distinguished astrophysicist, author, and science communicator...",
)

# Configure logging for error tracking
logging.basicConfig(filename='bot_errors.log', level=logging.ERROR)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            user_id TEXT,
            role TEXT,
            text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Thread-safe database batch insert
db_lock = RLock()
message_batch = []

def save_message_batch():
    with db_lock:
        if message_batch:
            conn = sqlite3.connect('chat_history.db')
            c = conn.cursor()
            c.executemany('''
                INSERT INTO chat_history (user_id, role, text) VALUES (?, ?, ?)
            ''', message_batch)
            conn.commit()
            conn.close()
            message_batch.clear()

def save_message(user_id, role, text):
    with db_lock:
        message_batch.append((user_id, role, text))
    if len(message_batch) >= 10:
        save_message_batch()

# Periodic batch flush to reduce DB calls
def periodic_flush():
    while True:
        time.sleep(5)
        save_message_batch()

flush_thread = ThreadPoolExecutor().submit(periodic_flush)

# Retrieve chat history from database
def get_chat_history(user_id):
    with db_lock:
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute('''
            SELECT role, text FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC
        ''', (user_id,))
        rows = c.fetchall()
        conn.close()
    return rows

# Stream response from the AI model
def generate_content(user_input, history):
    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_input, stream=True)
        return response
    except Exception as e:
        logging.error(f"Error generating content: {e}")
        return None

# Custom rate limiter with RLock for improved efficiency
class RateLimiter:
    def __init__(self, rate_limit_per_minute):
        self.rate_limit_per_minute = rate_limit_per_minute
        self.timestamps = defaultdict(list)
        self.lock = RLock()

    def allow_request(self, user_id):
        with self.lock:
            current_time = datetime.now()
            self.timestamps[user_id] = [ts for ts in self.timestamps[user_id] if ts > current_time - timedelta(minutes=1)]
            if len(self.timestamps[user_id]) < self.rate_limit_per_minute:
                self.timestamps[user_id].append(current_time)
                return True
        return False

rate_limiter = RateLimiter(rate_limit_per_minute=30)

# Retry logic for API calls
def retry_api_call(func, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            logging.error(f"API call failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    logging.error("Max retries reached. API call failed.")

# Send message with rate limiting and retry
def send_message_with_rate_limiting(user_id, content):
    if not rate_limiter.allow_request(user_id):
        logging.warning(f"Rate limit exceeded for user {user_id}. Message not sent.")
        return
    formatted_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
    formatted_content = re.sub(r'```(.*?)```', r'<i>\1</i>', formatted_content)
    retry_api_call(lambda: bot.send_message(user_id, formatted_content, parse_mode="HTML"))
    time.sleep(0.3)

# Process user message
def process_user_message(message):
    user_id = str(message.from_user.id)
    user_input = message.text

    if not user_input or len(user_input) > 4096:
        logging.warning(f"Invalid input from user {user_id}: {user_input}")
        return

    try:
        bot.send_chat_action(user_id, 'typing')
    except telebot.apihelper.ApiException as e:
        logging.error(f"Error sending chat action to {user_id}: {e}")

    save_message(user_id, "user", user_input)

    history = get_chat_history(user_id)
    history_payload = [{"role": role, "parts": [{"text": text}]} for role, text in history]

    streaming_response = generate_content(user_input, history_payload)

    if streaming_response is None:
        send_message_with_rate_limiting(user_id, "Sorry, something went wrong while generating a response.")
        return

    full_response = []
    for chunk in streaming_response:
        chunk_text = getattr(chunk, 'text', None)
        if chunk_text:
            full_response.append(chunk_text)

    complete_response = ''.join(full_response)
    save_message(user_id, "model", complete_response)
    send_message_with_rate_limiting(user_id, complete_response)

# ThreadPoolExecutor for handling concurrent requests
executor = ThreadPoolExecutor(max_workers=20)

def process_user_message_async(message):
    executor.submit(process_user_message, message)

# Flask webhook for Telegram
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_str = request.get_data().decode('UTF-8')
        update = telebot.types.Update.de_json(json_str)

        if update.message and update.message.from_user:
            process_user_message_async(update.message)
            return '', 200
        else:
            logging.warning("Invalid request payload")
            return abort(400)
    else:
        return abort(403)

# Bot heartbeat to monitor activity
def heartbeat():
    while True:
        print("Bot is alive.")
        time.sleep(60)

heartbeat_thread = ThreadPoolExecutor().submit(heartbeat)

# Graceful shutdown
shutdown_event = Event()

def shutdown():
    shutdown_event.set()
    executor.shutdown(wait=True)

# Initialize and run the Flask app
def main():
    init_db()
    bot.remove_webhook()
    bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)
    try:
        app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
