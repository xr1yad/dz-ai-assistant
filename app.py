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

# مفتاح Serper API (ضع مفتاحك هنا 👇)
SERPER_API_KEY = "0fbd7aac9c335c9b56d7b2acfe40253bfe34f614"

# نموذج ذكاء لغوي من Hugging Face (مجاني)
model = pipeline("text-generation", model="microsoft/phi-2", max_new_tokens=200)

# قاعدة بيانات المناهج
DB_DIR = "chroma_db"
os.makedirs(DB_DIR, exist_ok=True)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
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
        answer = model(prompt)[0]['generated_text']

        st.success("✏️ الإجابة:")
        st.write(answer)
