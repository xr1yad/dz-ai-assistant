import streamlit as st
import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
import numpy as np

# ================= إعدادات أساسية =================
st.set_page_config(page_title="المساعد الدراسي الجزائري", layout="wide")

st.title("🤖 المساعد الدراسي الجزائري الذكي")
st.write("مرحبًا! أنا مساعدك الدراسي. اسألني أي سؤال من المناهج أو من الإنترنت التعليمي بالعربية 🇩🇿")

# نموذج ذكاء لغوي من Hugging Face (مجاني)
import json

# سنستخدم واجهة API من Hugging Face بدلاً من تشغيل النموذج محليًا
HF_API_TOKEN = "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"

def query_huggingface(prompt):
    api_url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.6}}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        try:
            return result[0]["generated_text"]
        except Exception:
            return json.dumps(result)
    else:
        return f"⚠️ حدث خطأ في واجهة Hugging Face API: {response.status_code}"

# قاعدة بيانات المناهج
DB_DIR = "chroma_db"
os.makedirs(DB_DIR, exist_ok=True)
# دالة جديدة للحصول على تمثيلات النصوص (embeddings) من Hugging Face مجانًا
HF_API_TOKEN = "ضع_رمز_HuggingFace_الذي_نسخته_من_قبل_هنا"

def get_embeddings(texts):
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    embeddings = []
    for text in texts:
        payload = {"inputs": text}
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            emb = response.json()
            # نأخذ المتوسط لتقليل الحجم
            emb_vector = np.mean(emb, axis=0)
            embeddings.append(emb_vector)
        else:
            embeddings.append(np.zeros(384))  # fallback
    return np.array(embeddings).astype("float32")

client = chromadb.Client(Settings(persist_directory=DB_DIR))
try:
    collection = client.get_collection("curriculum")
except:
    collection = client.create_collection("curriculum")

# ================= وظيفة البحث في الإنترنت =================
def search_internet(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {
        "q": f"{query} site:education.gov.dz OR site:onefd.edu.dz OR site:wikipedia.org OR site:britannica.com",
        "num": 5
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        results = []
        for item in data.get("organic", []):
            results.append(f"{item.get('title')} - {item.get('link')}")
        return "\n".join(results)
    else:
        return "لم أتمكن من البحث في الإنترنت الآن."

# ================= واجهة المستخدم =================
user_input = st.chat_input("🗨️ اكتب سؤالك هنا...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        st.write("🔎 جاري البحث عن الإجابة...")

        # 1. البحث أولاً في المناهج الدراسية
        q_emb = embed_model.encode([user_input])
        results = collection.query(query_embeddings=q_emb.tolist(), n_results=2)
        docs = results.get("documents", [[]])[0]

        if docs:
            context = " ".join(docs)
        else:
            context = ""

        # 2. إذا لم نجد إجابة في المناهج → نبحث في الإنترنت
        if not context.strip():
            st.write("لم أجد الإجابة في المناهج، أبحث الآن في الإنترنت التعليمي...")
            web_results = search_internet(user_input)
            context = web_results

        # 3. توليد الإجابة النهائية
        prompt = f"السؤال: {user_input}\n\nالمراجع:\n{context}\n\nأجب بالعربية الفصحى بإيجاز وبأسلوب واضح للطلاب."
        answer = query_huggingface(prompt)

        st.success("✏️ الإجابة:")
        st.write(answer)
