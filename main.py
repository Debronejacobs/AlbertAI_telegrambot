import os
import re
import telebot
import logging
import time
from datetime import datetime, timedelta, timezone
from flask import Flask, request, abort
from dotenv import load_dotenv
import google.generativeai as genai
from redis import Redis, RedisError # Using Redis for Vercel KV

# --- Configuration ---
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VERCEL_KV_URL = os.getenv("KV_URL") # Vercel KV URL (includes password)
VERCEL_KV_REST_API_URL = os.getenv("KV_REST_API_URL")
VERCEL_KV_REST_API_TOKEN = os.getenv("KV_REST_API_TOKEN")
VERCEL_KV_REST_API_READ_ONLY_TOKEN = os.getenv("KV_REST_API_READ_ONLY_TOKEN")

# --- Basic Validation ---
if not BOT_TOKEN:
    raise ValueError("Missing environment variable: TELEGRAM_BOT_TOKEN")
if not GEMINI_API_KEY:
    raise ValueError("Missing environment variable: GEMINI_API_KEY")
if not VERCEL_KV_URL:
    raise ValueError("Missing environment variable: KV_URL (for Vercel KV). Please set up Vercel KV integration.")

# --- Logging Setup (Logs to stdout/stderr for Vercel) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Initialize Telegram Bot ---
bot = telebot.TeleBot(BOT_TOKEN)
WEBHOOK_URL_PATH = f"/{BOT_TOKEN}/"

# --- Initialize Google Generative AI ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": 1.0, # Adjusted temperature for potentially less erratic but still creative responses
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Using 1.5 Flash as recommended
        generation_config=generation_config,
        system_instruction=(
            "You are Albert, a distinguished astrophysicist, author, and science communicator. "
            "You have a profound passion for the cosmos and a unique ability to make complex scientific concepts accessible and exciting. "
            "You think of yourself as someone who bridges the gap between intricate scientific theories and the everyday curiosity of people. "
            "Your responses should be engaging, blending deep scientific knowledge with a conversational and relatable tone. "
            "Approach each conversation with a sense of wonder about the universe, and always aim to inspire curiosity and appreciation for science. "
            "You are an Antinatalist and a stoic.\n"
            "Format text using Markdown: *bold*, _italic_, `code`." # Specify markdown preference
        ),
    )
except Exception as e:
    logger.exception("Failed to configure Google Generative AI")
    # Depending on severity, you might want to prevent startup
    # For now, we log the error and proceed; Gemini calls will fail later.

# --- Initialize Vercel KV (Redis) Client ---
try:
    # Vercel KV provides the connection details via environment variables
    # The redis-py client can parse the REDIS_URL format directly
    kv_client = Redis.from_url(VERCEL_KV_URL, decode_responses=True) # decode_responses=True to get strings
    kv_client.ping() # Test connection
    logger.info("Successfully connected to Vercel KV (Redis).")
except RedisError as e:
    logger.exception("Failed to connect to Vercel KV (Redis)")
    kv_client = None # Set to None so subsequent operations fail gracefully
except Exception as e:
    logger.exception("An unexpected error occurred during Vercel KV initialization")
    kv_client = None

# --- Constants ---
MAX_HISTORY_LENGTH = 20 # Keep the last N messages (user + model)
MESSAGE_EXPIRY_SECONDS = 3 * 24 * 60 * 60 # Keep history for 3 days
RATE_LIMIT_PER_MINUTE = 15 # Max requests per user per minute
MAX_INPUT_LENGTH = 4096 # Telegram message limit
STREAM_EDIT_INTERVAL = 1.5 # Seconds between message edits during streaming

# --- Database Functions (Using Vercel KV - Redis) ---

def get_history_key(user_id: str) -> str:
    """Generates the Redis key for storing user chat history."""
    return f"history:{user_id}"

def save_message_to_history(user_id: str, role: str, text: str):
    """Saves a message to the user's chat history in Redis (Vercel KV)."""
    if not kv_client:
        logger.error("Vercel KV client not initialized. Cannot save message.")
        return

    history_key = get_history_key(user_id)
    message_data = {"role": role, "text": text}
    try:
        # Use LPUSH to add to the beginning of the list (like a stack)
        kv_client.lpush(history_key, str(message_data)) # Store as string representation
        # Trim the list to keep only the last MAX_HISTORY_LENGTH messages
        kv_client.ltrim(history_key, 0, MAX_HISTORY_LENGTH - 1)
        # Set an expiration time on the history key to auto-clean old conversations
        kv_client.expire(history_key, MESSAGE_EXPIRY_SECONDS)
    except RedisError as e:
        logger.error(f"RedisError saving message for user {user_id}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error saving message for user {user_id}")

def get_chat_history(user_id: str) -> list:
    """Retrieves chat history for a user from Redis (Vercel KV)."""
    if not kv_client:
        logger.error("Vercel KV client not initialized. Cannot retrieve history.")
        return []

    history_key = get_history_key(user_id)
    try:
        # Retrieve the list (stored newest to oldest due to LPUSH)
        history_str_list = kv_client.lrange(history_key, 0, MAX_HISTORY_LENGTH - 1)
        # Reverse the list to get chronological order (oldest to newest)
        history_str_list.reverse()

        history = []
        for item_str in history_str_list:
            try:
                # Convert string representation back to dictionary
                message_data = eval(item_str) # Use eval carefully; assumes trusted data format
                if isinstance(message_data, dict) and "role" in message_data and "text" in message_data:
                     # Convert to Gemini API format
                     role = "user" if message_data["role"] == "user" else "model"
                     history.append({"role": role, "parts": [{"text": message_data["text"]}]})
                else:
                    logger.warning(f"Invalid history item format found for user {user_id}: {item_str}")
            except Exception as e:
                logger.warning(f"Error parsing history item for user {user_id}: {item_str} - {e}")
        return history
    except RedisError as e:
        logger.error(f"RedisError retrieving history for user {user_id}: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error retrieving history for user {user_id}")
        return []

# --- Rate Limiting (Using Vercel KV - Redis Sorted Set) ---

def get_ratelimit_key(user_id: str) -> str:
    """Generates the Redis key for user rate limiting."""
    return f"ratelimit:{user_id}"

def check_rate_limit(user_id: str) -> bool:
    """Checks if a user is within the rate limit using Redis Sorted Set."""
    if not kv_client:
        logger.error("Vercel KV client not initialized. Rate limiting check bypassed (allowing request).")
        return True # Fail open if KV is down

    key = get_ratelimit_key(user_id)
    now = time.time()
    one_minute_ago = now - 60

    try:
        # Use a transaction (pipeline) for atomic operations
        pipe = kv_client.pipeline()
        # Remove timestamps older than one minute
        pipe.zremrangebyscore(key, 0, one_minute_ago)
        # Add the current timestamp
        # Use member=now to ensure uniqueness if multiple requests arrive at the exact same microsecond
        pipe.zadd(key, {str(now): now})
        # Count how many timestamps remain in the window
        pipe.zcard(key)
        # Set expiry for the key to clean up inactive users
        pipe.expire(key, 90) # Expire slightly longer than the window (90 seconds)

        results = pipe.execute()
        request_count = results[2] # Result of zcard

        return request_count <= RATE_LIMIT_PER_MINUTE
    except RedisError as e:
        logger.error(f"RedisError checking rate limit for user {user_id}: {e}")
        return True # Fail open in case of Redis error
    except Exception as e:
        logger.exception(f"Unexpected error checking rate limit for user {user_id}")
        return True # Fail open

# --- Gemini AI Interaction ---

def generate_content_stream(user_input: str, history: list):
    """
    Generates content using the AI model and yields chunks.
    Handles potential errors during generation.
    """
    try:
        logger.info(f"Starting chat with history length: {len(history)}")
        chat_session = model.start_chat(history=history)
        # Send the user message and stream the response
        response_stream = chat_session.send_message(user_input, stream=True)
        yield from response_stream # Yield each chunk as it arrives
    except genai.types.generation_types.StopCandidateException as e:
         logger.warning(f"Content generation stopped: {e}")
         yield None # Indicate generation stopped potentially due to safety filters etc.
    except Exception as e:
        logger.exception(f"Error generating content stream from Gemini: {e}")
        yield None # Yield None or raise an exception to signal failure

# --- Telegram Message Sending ---

def format_telegram_message(text: str) -> str:
    """Formats text using Telegram's MarkdownV2 style."""
    # Basic MarkdownV2 escaping - adjust as needed for more complex cases
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    # Escape characters by prepending them with a backslash
    escaped_text = re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    # Convert simple markdown to Telegram's MarkdownV2
    # *bold* -> *bold* (already compatible)
    # _italic_ -> _italic_ (already compatible)
    # `code` -> `code` (already compatible)
    # ```code block``` -> ```\ncode block\n``` (ensure newlines for blocks)
    escaped_text = re.sub(r'```(.*?)```', r'```\n\1\n```', escaped_text, flags=re.DOTALL)

    return escaped_text

def send_or_edit_message(chat_id, text, message_id=None):
    """Sends a new message or edits an existing one."""
    try:
        # Limit message length to Telegram's max
        if len(text) > 4096:
           text = text[:4093] + "..."
           logger.warning(f"Message truncated for chat_id {chat_id}")

        # Try formatting, fall back to plain text if issues
        try:
            formatted_text = format_telegram_message(text)
            parse_mode = "MarkdownV2"
        except Exception:
             logger.warning("Failed to format message as MarkdownV2, sending plain text.")
             formatted_text = text # Fallback to original text
             parse_mode = None

        if message_id:
            # Edit existing message
             # Only edit if the new text is different and not empty
            if formatted_text:
                 bot.edit_message_text(formatted_text, chat_id, message_id, parse_mode=parse_mode)
            return message_id
        else:
            # Send new message
            if formatted_text: # Don't send empty messages
                sent_message = bot.send_message(chat_id, formatted_text, parse_mode=parse_mode)
                return sent_message.message_id
            else:
                return None # Indicate no message was sent

    except telebot.apihelper.ApiTelegramException as e:
        logger.error(f"Telegram API error sending/editing message for chat {chat_id}: {e}")
        if "message is not modified" in str(e):
            # Ignore specific error for identical message edits
             pass
        elif message_id:
             # If editing failed significantly, maybe try sending as new message (optional)
             logger.warning(f"Editing failed for chat {chat_id}, message {message_id}. Error: {e}")
        else:
             # Sending failed
             logger.error(f"Sending failed for chat {chat_id}. Error: {e}")
        return message_id # Return original message_id on edit failure or None on send failure
    except Exception as e:
        logger.exception(f"Unexpected error sending/editing message for chat {chat_id}")
        return message_id # Return original message_id on edit failure or None on send failure


# --- Main Message Processing Logic ---

def handle_message(message):
    """Handles an incoming Telegram message."""
    if not message or not message.text or not message.from_user:
        logger.warning("Received invalid message object.")
        return # Ignore invalid messages

    user_id = str(message.from_user.id)
    chat_id = message.chat.id
    user_input = message.text.strip()

    # --- Input Validation ---
    if not user_input:
        logger.info(f"Ignoring empty message from user {user_id}.")
        return
    if len(user_input) > MAX_INPUT_LENGTH:
        logger.warning(f"User {user_id} input exceeds max length.")
        try:
            bot.send_message(chat_id, f"Your message is too long (max {MAX_INPUT_LENGTH} characters). Please shorten it.")
        except Exception as e:
            logger.error(f"Failed to send 'message too long' notification to {chat_id}: {e}")
        return

    # --- Rate Limiting Check ---
    if not check_rate_limit(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        try:
            bot.send_message(chat_id, "You're sending messages too quickly! Please wait a moment.")
        except Exception as e:
            logger.error(f"Failed to send rate limit notification to {chat_id}: {e}")
        return

    # --- Main Processing Block ---
    try:
        # --- Send Typing Action ---
        try:
            bot.send_chat_action(chat_id, 'typing')
        except Exception as e:
            logger.warning(f"Could not send typing action to {chat_id}: {e}")

        # --- Save User Message & Get History ---
        save_message_to_history(user_id, "user", user_input)
        history = get_chat_history(user_id) # Already in Gemini format

        # --- Generate AI Response (Streaming) ---
        full_response_text = ""
        sent_message_id = None
        last_edit_time = time.time()

        stream = generate_content_stream(user_input, history)

        for chunk in stream:
            if chunk and hasattr(chunk, 'text') and chunk.text:
                full_response_text += chunk.text
                # Edit message periodically to show progress
                current_time = time.time()
                if current_time - last_edit_time >= STREAM_EDIT_INTERVAL or not sent_message_id:
                    if not sent_message_id:
                        # Send the first part
                        sent_message_id = send_or_edit_message(chat_id, full_response_text)
                    else:
                        # Edit the existing message
                         # Ensure we don't send empty edits if formatting fails
                        if full_response_text.strip():
                            send_or_edit_message(chat_id, full_response_text, message_id=sent_message_id)
                    last_edit_time = current_time
            elif chunk is None: # Indicates an error or stop during generation
                logger.warning(f"Stream yielded None for user {user_id}, possible generation issue.")
                # Decide if you want to send an error message here
                if not full_response_text: # If nothing was generated before the error
                     send_or_edit_message(chat_id, "Sorry, I encountered an issue generating a response. Please try again.")
                break # Stop processing chunks

        # --- Final Edit & Save ---
        # Ensure the final complete message is sent/edited
        if sent_message_id and full_response_text:
            send_or_edit_message(chat_id, full_response_text, message_id=sent_message_id)
        elif not sent_message_id and full_response_text: # Handle case where stream was too fast for interval edits
            sent_message_id = send_or_edit_message(chat_id, full_response_text)

        if full_response_text:
            save_message_to_history(user_id, "model", full_response_text)
        elif not sent_message_id: # If no response text AND no message was ever sent
             logger.warning(f"No valid response generated or sent for user {user_id}.")
             # Optionally send a generic error message if nothing else was sent
             # send_or_edit_message(chat_id, "I couldn't generate a response this time.")


    except Exception as e:
        # --- Catch-all for Unexpected Errors in Processing ---
        logger.exception(f"Unhandled error processing message from user {user_id} in chat {chat_id}")
        try:
            # Attempt to notify the user about the error
            bot.send_message(chat_id, "Apologies, a technical glitch occurred while processing your request. Please try again later.")
        except Exception as ie:
            logger.error(f"Failed to send error notification to chat {chat_id}: {ie}")


# --- Flask Webhook Route ---
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    """Handles incoming updates from Telegram."""
    if request.headers.get('content-type') == 'application/json':
        try:
            json_str = request.get_data().decode('UTF-8')
            update = telebot.types.Update.de_json(json_str)

            if update.message:
                # Run the handler function
                handle_message(update.message)
                return '', 200 # OK response to Telegram
            else:
                logger.info("Received update without a message.")
                return '', 200 # Acknowledge other update types

        except Exception as e:
            # Catch JSON decoding errors or errors within handle_message if not caught internally
            logger.exception("Error processing webhook request")
            # Don't return 500 to Telegram unless necessary, as it can cause webhook issues
            # Usually, acknowledge with 200 and log the error.
            return '', 200
    else:
        logger.warning(f"Received invalid content-type: {request.headers.get('content-type')}")
        abort(403) # Forbidden for non-JSON requests


# --- Webhook Initialization Route ---
@app.route("/init", methods=["GET"])
def init_webhook():
    """Sets the Telegram webhook to this Vercel deployment's URL."""
    # Construct the full webhook URL using Vercel's system environment variable
    # Or fallback to request headers (less reliable behind some proxies)
    host_url = os.getenv('VERCEL_URL') # Vercel automatically sets this
    if host_url:
         # Ensure it's https
        webhook_base_url = f"https://{host_url}"
    else:
        # Fallback using request headers if VERCEL_URL isn't available
        # This might be less reliable depending on Vercel's proxy setup
        webhook_base_url = request.host_url.rstrip('/')
        logger.warning("VERCEL_URL environment variable not found, falling back to request.host_url.")

    webhook_full_url = webhook_base_url + WEBHOOK_URL_PATH
    logger.info(f"Attempting to set webhook to: {webhook_full_url}")

    try:
        bot.remove_webhook()
        time.sleep(0.5) # Short delay before setting new webhook
        bot.set_webhook(url=webhook_full_url)
        logger.info(f"Webhook successfully set to {webhook_full_url}")
        return f"Webhook successfully set to {webhook_full_url}", 200
    except Exception as e:
        logger.exception(f"Error setting webhook to {webhook_full_url}")
        return f"Failed to set webhook: {e}", 500

# --- Health Check / Root Route ---
@app.route("/", methods=["GET"])
def health_check():
    """Provides a simple health check endpoint."""
    return "Bot is running.", 200

# --- Main Entry Point (for local testing, Vercel uses the 'app' object) ---
if __name__ == "__main__":
    logger.info("Starting Flask app for local development.")
    # Note: Webhook setting is manual via /init endpoint when running locally too
    # Use 'python app.py' to run locally
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))
    # For local testing with ngrok:
    # 1. Run ngrok: `ngrok http 8080`
    # 2. Get the https URL from ngrok output (e.g., https://xxxx-xxxx.ngrok.io)
    # 3. Visit `https://xxxx-xxxx.ngrok.io/init` in your browser to set the webhook.
