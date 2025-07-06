import os
import streamlit as st
import datetime
import pyttsx3
import speech_recognition as sr
import threading

from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

from ner_utils import extract_named_entities, format_entities_as_flashcards
from study_planner import create_study_plan
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)

def speak(text):
    def run():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=run).start()


def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            return query
        except Exception as e:
            st.error("Speech recognition failed.")
            return ""

# Load PDFs
@st.cache_resource
def load_data():
    books_dir = "books"
    all_texts = []
    embedding_model = HuggingFaceEmbeddings(model_name="********************")  # Replace with your model name
    for filename in os.listdir(books_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(books_dir, filename))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            all_texts.extend(chunks)
    vectorstore = FAISS.from_documents(all_texts, embedding_model)
    return vectorstore, all_texts

# LLM
@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id="********************",  # Replace with your model repo ID
        task="text-generation",
        temperature=0.7,
        max_new_tokens=500,
        huggingfacehub_api_token=""
    )

# Setup
st.set_page_config(page_title="📚 AI Study Assistant", layout="wide")
st.title("📚 AI-Powered Educational Assistant")

llm = load_llm()
vectorstore, all_texts = load_data()

# Prompts
qa_prompt = PromptTemplate.from_template("""
You are a helpful study assistant. Answer the question using only the provided context.
If the answer is not present in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)
summary_chain = LLMChain(llm=llm, prompt=qa_prompt)

flashcard_prompt = PromptTemplate.from_template("""
Generate 3 educational flashcards from the following text.

Text:
{text}

Format:
Question 1: ...
Answer   1: ...
Question 2: ...
Answer   2: ...
Question 3: ...
Answer   3: ...
""")
flashcard_chain = LLMChain(llm=llm, prompt=flashcard_prompt)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 Ask a Question", "📅 Study Plan", "🧠 Flashcards/NER", "📝 Summarization"])

with tab1:
    st.subheader("🧠 Ask a Question")
    use_voice = st.toggle("Use Microphone")
    user_query = ""

    if use_voice:
        if st.button("🎙️ Start Listening"):
            user_query = recognize_speech()
            st.text_input("Recognized Query:", user_query)
    else:
        user_query = st.text_input("Type your question")

    if st.button("🔍 Get Answer") and user_query:
        response = qa_chain.invoke({"query": user_query})
        answer = response["result"].strip()
        st.success(answer)
        speak(answer)

with tab2:
    st.subheader("📅 Generate Study Plan")
    topic = st.text_input("Enter topic")
    date = st.date_input("Select deadline", min_value=datetime.date.today())

    if st.button("📆 Generate Plan") and topic:
        plan = create_study_plan(topic, str(date))
        st.code(plan, language="markdown")
        speak("Here is your study plan.")

with tab3:
    st.subheader("🧠 NER & Flashcards")
    chunk_index = st.number_input("Select chunk index", 0, len(all_texts) - 1, 0)

    if st.button("🔍 Extract Named Entities"):
        chunk_text = all_texts[chunk_index].page_content
        entities = extract_named_entities(chunk_text)
        flashcards = format_entities_as_flashcards(entities)
        for i, (q, a) in enumerate(flashcards, 1):
            st.markdown(f"**{i}. Q: {q}**  \nA: {a}")
            speak(f"Q: {q}. A: {a}")

    if st.button("🧠 Generate Flashcards"):
        chunk_text = all_texts[chunk_index].page_content
        cards = flashcard_chain.run(text=chunk_text)
        st.code(cards)
        speak("Flashcards generated.")

with tab4:
    st.subheader("📝 Summarization")
    start = st.number_input("Start chunk index", 0, len(all_texts) - 1, 0)
    end = st.number_input("End chunk index", start + 1, len(all_texts), start + 2)

    if st.button("📖 Summarize"):
        for i in range(start, end):
            chunk = all_texts[i]
            summary = summary_chain.run(context=chunk.page_content, question="Summarize this.")
            st.markdown(f"**Summary {i+1}:** {summary}")
            speak(f"Summary of chunk {i+1}")
            speak(summary)
