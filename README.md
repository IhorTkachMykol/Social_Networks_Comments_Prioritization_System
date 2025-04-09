# 🧠 Intelligent Text Classifier & Conversational AI Assistant

An advanced NLP project for **text preprocessing**, **classification**, and **interactive chat** powered by **LangChain** and **OpenAI GPT-3.5**.

---

## 🚀 Features

✅ **Multi-step Text Classification**:
- `emoji` — detects if input contains only emojis
- `keyword` — identifies keywords using Haystack + XLM-RoBERTa
- `spam` — flags spam content using a trained Naive Bayes classifier
- `normal` — routes normal input to the chatbot for a meaningful reply

✅ **Emoji Handling**:
- Converts emojis to descriptive text (`❤️` → `red_heart`)
- Optionally removes emojis for cleaner processing

✅ **Conversational AI with Context**:
- Powered by LangChain and OpenAI GPT-3.5
- Uses vector search via **ChromaDB** for contextualized answers
- Maintains **chat history** for better, smarter replies

---

## 🛠️ Project Structure

src/ ├── data_preprocessing/ │ └── remove_emoji.py # Emoji removal and conversion ├── 
keywords_classifier/ │ └── detect_keywords.py # Keyword detection with Haystack ├── 
spam_classifier/ │ └── spam_classifier.py # Spam detection with scikit-learn ├── 
langchain/ │ └── my_langchain_history.py # LangChain history-aware QA └── main.py # Entry point for classification + chat

---

## 📦 Requirements

### 🐍 Python 3.10+

Install via `environment.yml` (recommended):
```bash
conda env create -f environment.yml
conda activate py310
```
Or using pip:
```
pip install -r requirements.txt
```
💬 Usage
1. Run the Classifier
```
python main.py
```
2. Example Output
You: ❤️🔥😂
Classified as: emoji

If not emoji/keyword/spam, a chat will start:

You: what is Alice in Wonderland about?
Assistant: It's a fantasy novel about a girl named Alice who falls into a magical world...

🧠 Chroma DB Setup

To load your own knowledge base into the chatbot:

    Add .txt files into the data/ directory

    Run:
    ```
    python create_chroma_db.py
    ```
🧪 Model Details

    Spam Classifier: CountVectorizer + MultinomialNB (trained model: pipe_spam_cls_model.pkl)

    Keyword Detector: Haystack TransformersQueryClassifier with ukr-models/xlm-roberta-base-uk

    LLM: OpenAI gpt-3.5-turbo via LangChain

    Vector DB: ChromaDB with local persistence

🔐 Environment Variables

Set your OpenAI API Key:
```
export OPENAI_API_KEY=your_openai_key_here
```

Or use a .env file + python-dotenv.
✨ TODO / Improvements

    ✅ Add emoji only detection logic

    ⚠ Improve CLI user experience (add options & flags)

    📊 Add Streamlit or web UI interface

    🧪 Add unit tests

👨‍💻 Author

Developed by Serhii Kolotukhin 
📍 www.linkedin.com/in/serhii-kolotuhkin-25648a166 

Powered by open-source LLM tools
📄 License

MIT License — free to use, modify, and distribute.

