# AI-Powered Chatbot — Full CodeTech Requirements

This project is designed to satisfy the CodeTech Technologies **AI-Powered Chatbot** project requirements.

## Requirements covered

### Description
- Intelligent customer-support / FAQ chatbot
- Natural Language Processing (NLP)
- Context-aware conversation handling

### Technologies
- Python
- NLTK
- Transformers
- Flask
- SQLite

### Outcome
- Contextual responses
- User interaction logs
- Browser-based chat UI
- Sentiment analysis
- Intent detection
- Health endpoint and log endpoint

## How the features work

1. **NLTK** normalizes and stems user text for FAQ intent matching.
2. **Transformers** provides:
   - DistilBERT sentiment analysis
   - FLAN-T5 response generation for unknown questions
3. **Flask** provides the web application and `/chat` API.
4. **SQLite** stores every interaction with timestamp, intent and sentiment.
5. **Conversation context** is kept in the browser and sent to the Flask API so the Transformer can use recent messages.
6. A reliable FAQ knowledge base provides direct answers for common customer-support questions.

## Run on Windows laptop

Open Command Prompt inside this project folder:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

### First Transformer run
The first time a Transformer feature is used, Hugging Face downloads the required models. An internet connection is needed for that first download. After the models are available locally, the application can use them from the local cache.

If a Transformer model cannot be loaded, the application safely falls back to its FAQ/NLP response instead of crashing.

## Useful endpoints

- `/` — chatbot interface
- `/chat` — chatbot API
- `/health` — application health check
- `/logs` — latest 100 interaction logs

## Database

`chatbot.db` is created automatically and contains the `chat_logs` table.

## Project title for submission

**AI-Powered Customer Support Chatbot Using NLP and Transformers**

## Important

This is an educational/customer-support demonstration. Do not put real passwords, confidential customer information, or production secrets into this project.


## Troubleshooting: chatbot page opens but no response

Make sure the PowerShell window running Flask remains open.

Run:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

This fixed version returns FAQ answers immediately. Transformer models are loaded only when a question is outside the built-in FAQ set, preventing the first message from appearing stuck while a large model downloads.

Try first:
- Hello
- What can you do?
- How do I register?
- I forgot my password
- How can I contact support?

For Transformer-generated answers, keep an internet connection available the first time so the Hugging Face models can be downloaded.


## Added customer-support FAQs

The updated version includes non-account topics such as:
- Products and product availability
- Order tracking
- Delivery time and delayed delivery
- Payment methods and failed payments
- Returns and refunds
- Order cancellation
- Damaged products
- Support hours and complaints

Example questions:
- What products do you offer?
- How can I track my order?
- When will my order arrive?
- What payment methods do you accept?
- How can I return a product?
- How long does a refund take?
- My order is delayed. What should I do?
- I received a damaged product.
