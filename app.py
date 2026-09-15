from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime
import sqlite3
import re

# NLP
from nltk.stem import PorterStemmer

# Transformers
from transformers import pipeline

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot.db"

stemmer = PorterStemmer()

# FAQ knowledge base
FAQS = [
    {
        "intent": "product_info",
        "patterns": ["what products do you offer", "what products are available", "products", "product information", "tell me about products"],
        "response": "We offer a range of products across different categories. Please provide the product name or category you are interested in, and I can help with available information."
    },
    {
        "intent": "product_availability",
        "patterns": ["is the product available", "product availability", "is this item available", "do you have this product"],
        "response": "Product availability can vary. Please provide the product name or item you want to check."
    },
    {
        "intent": "order_tracking",
        "patterns": ["how can i track my order", "track my order", "order tracking", "where is my order", "check order status", "delivery status"],
        "response": "You can track your order using the order-tracking option in the application. Enter your order ID to view the latest delivery status."
    },
    {
        "intent": "delivery_time",
        "patterns": ["how long does delivery take", "when will my order arrive", "delivery time", "how long is delivery", "when will my package arrive"],
        "response": "Delivery time depends on your location and the selected delivery option. You can check the estimated delivery date from your order details."
    },
    {
        "intent": "delayed_order",
        "patterns": ["my order is delayed", "order delayed", "delivery is late", "package is late", "late delivery"],
        "response": "I'm sorry about the delay. Please check the latest tracking status using your order ID. If the status has not changed, customer support can investigate the delivery."
    },
    {
        "intent": "payment_methods",
        "patterns": ["what payment methods do you accept", "payment methods", "can i pay using upi", "do you accept credit cards", "payment options"],
        "response": "Common payment options include UPI, credit/debit cards and other methods enabled by the application. Available options are shown at checkout."
    },
    {
        "intent": "payment_failed",
        "patterns": ["my payment failed", "payment failed", "payment is not working", "payment error", "transaction failed"],
        "response": "If a payment failed, please verify your payment details and try again. If money was deducted but the order was not confirmed, check the payment status and contact support if needed."
    },
    {
        "intent": "return",
        "patterns": ["how can i return a product", "return a product", "return policy", "i want to return", "can i return my order"],
        "response": "You can request a return from the order details page when the item is eligible. The applicable return conditions are shown with the order."
    },
    {
        "intent": "refund",
        "patterns": ["how long does a refund take", "refund", "when will i get my refund", "refund status", "money back"],
        "response": "Refund processing time depends on the payment method and provider. After the refund is initiated, the status can be checked from your order or payment details."
    },
    {
        "intent": "cancel_order",
        "patterns": ["can i cancel my order", "cancel my order", "cancel order", "i want to cancel"],
        "response": "If your order has not entered the shipping process, you may be able to cancel it from the order details page. Otherwise, you may need to use the return process after delivery."
    },
    {
        "intent": "damaged_product",
        "patterns": ["received a damaged product", "product is damaged", "damaged item", "item arrived damaged", "broken product"],
        "response": "I'm sorry the product arrived damaged. Please keep the packaging and report the issue through customer support or the return option with photos if requested."
    },
    {
        "intent": "support_hours",
        "patterns": ["what are your support hours", "support hours", "when is customer support available", "customer service hours"],
        "response": "Customer support availability depends on the service. Please check the application's Contact or Support section for the current support hours."
    },
    {
        "intent": "complaint",
        "patterns": ["i want to complain", "file a complaint", "report a problem", "i have a complaint", "bad service"],
        "response": "I'm sorry you're having a problem. Please provide your order or issue details through the application's support/contact section so the team can investigate."
    },
    {
        "intent": "greeting",
        "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
        "response": "Hello! 👋 I'm your AI customer-support assistant. How can I help you?"
    },
    {
        "intent": "features",
        "patterns": ["what can you do", "features", "help", "what do you do"],
        "response": "I can answer FAQs, understand common customer-support questions, remember the recent conversation context, detect sentiment, and record interaction logs."
    },
    {
        "intent": "registration",
        "patterns": ["how to register", "create account", "sign up", "registration"],
        "response": "To register, open the application's registration page and provide the required details such as your name, email and password."
    },
    {
        "intent": "login",
        "patterns": ["how to login", "login", "sign in", "log in"],
        "response": "Use your registered email and password on the login page. If you cannot sign in, try the password-reset option."
    },
    {
        "intent": "password",
        "patterns": ["forgot password", "reset password", "change password", "password"],
        "response": "If you forgot your password, choose 'Forgot Password' on the login page and follow the password-reset instructions."
    },
    {
        "intent": "support",
        "patterns": ["contact support", "customer support", "support", "help desk"],
        "response": "You can contact customer support through the application's support/contact section. Please include your issue and relevant details."
    },
    {
        "intent": "thanks",
        "patterns": ["thank you", "thanks", "thank"],
        "response": "You're welcome! 😊 I'm happy to help."
    },
    {
        "intent": "goodbye",
        "patterns": ["bye", "goodbye", "see you"],
        "response": "Goodbye! 👋 Have a great day!"
    },
]

# The Transformer model is loaded lazily so Flask starts quickly.
sentiment_model = None
generator_model = None

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            sentiment TEXT,
            intent TEXT,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def normalize(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return set(stemmer.stem(w) for w in words)

def detect_intent(message):
    tokens = normalize(message)
    best_intent = "unknown"
    best_response = None
    best_score = 0.0

    for faq in FAQS:
        for pattern in faq["patterns"]:
            p_tokens = normalize(pattern)
            if not p_tokens:
                continue
            overlap = len(tokens & p_tokens) / len(p_tokens)
            exact_bonus = 1.0 if pattern.lower() in message.lower() else 0.0
            score = overlap + exact_bonus
            if score > best_score:
                best_score = score
                best_intent = faq["intent"]
                best_response = faq["response"]

    # Avoid false positives from very weak matches.
    if best_score < 0.45:
        return "unknown", None
    return best_intent, best_response

def get_sentiment(text):
    """Transformer sentiment analysis with a simple fallback."""
    global sentiment_model
    try:
        if sentiment_model is None:
            sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        result = sentiment_model(text[:512])[0]
        label = result["label"].upper()
        return f"{label} ({result['score']:.2f})"
    except Exception:
        positive = {"good", "great", "happy", "excellent", "love", "thank", "thanks", "helpful", "awesome"}
        negative = {"bad", "angry", "hate", "terrible", "awful", "failed", "delay", "delayed", "problem", "broken", "damaged"}
        words = set(re.findall(r"[a-zA-Z]+", text.lower()))
        pos = len(words & positive)
        neg = len(words & negative)
        if pos > neg:
            return "POSITIVE (fallback)"
        if neg > pos:
            return "NEGATIVE (fallback)"
        return "NEUTRAL (fallback)"

def generate_contextual_reply(message, history):
    """
    Uses a Transformers text-generation model when available.
    If the model is unavailable/offline, the application safely falls back
    to its FAQ response.
    """
    global generator_model

    try:
        if generator_model is None:
            generator_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-small"
            )

        context_lines = []
        for item in history[-4:]:
            context_lines.append(f"User: {item['user']}")
            context_lines.append(f"Assistant: {item['bot']}")

        prompt = (
            "You are a concise and helpful customer support assistant. "
            "Answer the user's latest question using the conversation context. "
            "Do not invent company policies. If information is unknown, say so. "
            "Conversation:\n" + "\n".join(context_lines) +
            f"\nUser: {message}\nAssistant:"
        )
        result = generator_model(
            prompt,
            max_new_tokens=80,
            do_sample=False
        )[0]["generated_text"].strip()

        if result:
            return result
    except Exception:
        pass

    return None

def log_chat(session_id, user_message, bot_response, sentiment, intent):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO chat_logs
           (session_id, user_message, bot_response, sentiment, intent, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            user_message,
            bot_response,
            sentiment,
            intent,
            datetime.now().isoformat(timespec="seconds")
        )
    )
    con.commit()
    con.close()

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    session_id = str(data.get("session_id", "web-session"))

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    # Conversation context is maintained by the browser and sent with each request.
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []

    intent, faq_response = detect_intent(message)

    # IMPORTANT: Return known FAQ answers immediately.
    # Transformer models can take time to download/load on the first request,
    # so they are never allowed to delay normal FAQ answers.
    if faq_response:
        response = faq_response
        sentiment = get_sentiment(message)
    else:
        # Transformers are used only for questions outside the FAQ knowledge base.
        sentiment = get_sentiment(message)
        response = generate_contextual_reply(message, history)
        if not response:
            response = (
                "I don't have enough information to answer that confidently. "
                "Please rephrase your question or contact customer support."
            )

    log_chat(session_id, message, response, sentiment, intent)

    return jsonify({
        "response": response,
        "sentiment": sentiment,
        "intent": intent
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "AI-Powered Chatbot"})

@app.get("/logs")
def logs():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, user_message, bot_response, sentiment, intent, created_at "
        "FROM chat_logs ORDER BY id DESC LIMIT 100"
    ).fetchall()
    con.close()
    return jsonify([dict(row) for row in rows])

if __name__ == "__main__":
    init_db()
    print("AI-Powered Chatbot starting...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=True)