import os
import fitz  # PyMuPDF
import warnings
import pyttsx3
import speech_recognition as sr
import pyaudio
import spacy
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ner_utils import extract_named_entities, format_entities_as_flashcards
from study_planner import create_study_plan

warnings.filterwarnings("ignore", category=FutureWarning)

# TTS Setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Speech-to-text
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Listening... (speak your query)")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            print(f"📝 You said: {query}")
            return query
        except sr.UnknownValueError:
            print("❌ Sorry, I couldn't understand that.")
        except sr.RequestError:
            print("⚡ STT service is down or unavailable.")
    return ""

# API Token
HUGGINGFACEHUB_API_TOKEN = "********************"  # Replace with your Hugging Face API token

# Load and Process PDFs
books_dir = "books"
all_texts = []
all_documents = []

embedding_model = HuggingFaceEmbeddings(model_name="****************") # Replace with your model name

print("📚 Loading and processing all books in folder:", books_dir)
for filename in os.listdir(books_dir):
    if filename.endswith(".pdf"):
        book_path = os.path.join(books_dir, filename)
        book_name = os.path.splitext(filename)[0]
        print(f"📘 Processing: {book_name}")

        loader = PyPDFLoader(book_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["book"] = book_name
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["book"] = book_name

        all_documents.extend(documents)
        all_texts.extend(chunks)

# Show available books
available_books = {os.path.splitext(f)[0] for f in os.listdir(books_dir) if f.endswith(".pdf")}
print("📚 Available books:", ", ".join(sorted(available_books)))

# Build FAISS index
print("📂 Storing vectors using FAISS...")
vectorstore = FAISS.from_documents(all_texts, embedding_model)
retriever = vectorstore.as_retriever()

# LLM Connection
llm = HuggingFaceEndpoint(
    repo_id="********************",  # Replace with your model repo ID
    task="text-generation",
    temperature=0.7,
    max_new_tokens=500,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# QA Prompt (FIXED)
qa_prompt = PromptTemplate.from_template("""
You are a helpful study assistant. Answer the question using only the provided context.
If the answer is not present in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""")

# RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# Flashcard Prompt
flashcard_prompt = PromptTemplate.from_template("""
Generate 3 educational flashcards from the following text.
Each flashcard should include a question and a concise answer.

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
summary_chain = LLMChain(llm=llm, prompt=qa_prompt)  # Using the same prompt for summaries

# Book filter parser
def extract_book_name_and_query(user_input):
    pattern = r'\[book:(.*?)\]'
    match = re.search(pattern, user_input)
    if match:
        book_name = match.group(1).strip()
        query = re.sub(pattern, '', user_input).strip()
        return book_name, query
    return None, user_input.strip()

# CLI Actions
def summarize_all(start=0, end=2):
    print(f"\n📘 Summarizing chunks {start + 1} to {end}...\n")
    for i, chunk in enumerate(all_texts[start:end], start=start):
        print(f"➡ Summary of chunk {i + 1}:")
        summary = summary_chain.run(context=chunk.page_content, question="Summarize this.")
        print(summary, "\n")
        speak(f"Summary of chunk {i + 1}")
        speak(summary)

def generate_flashcards_for_chunk(chunk_index):
    if chunk_index < 0 or chunk_index >= len(all_texts):
        print("❌ Invalid chunk index.")
        return
    print(f"\n🧠 Flashcards from chunk {chunk_index + 1}:\n")
    cards = flashcard_chain.run(text=all_texts[chunk_index].page_content)
    print(cards)
    for line in cards.splitlines():
        line = line.strip()
        if line.lower().startswith(("q", "a")):
            speak(line)

# CLI Loop
while True:
    choice = input("🎧 Press Enter to type or 'speak' for voice input: ").strip().lower()
    if choice == "speak":
        user_input = recognize_speech()
    else:
        user_input = input("🤔 Ask a question (or type 'summarize', 'flashcard [index]', 'ner [index]', 'nerpage [index]', 'nerpagebook <BookName> <page_index>', or 'plan <topic> <YYYY-MM-DD>' 'exit'): ").strip()

    if not user_input:
        continue
    elif user_input.lower() == "exit":
        print("👋 Goodbye!")
        break
    elif user_input.lower().startswith("summarize"):
        summarize_all(0, 2)
    elif user_input.lower().startswith("flashcard"):
        try:
            index = int(user_input.split(" ")[1])
            generate_flashcards_for_chunk(index)
        except (IndexError, ValueError):
            print("❌ Usage: flashcard <chunk_index>")
    elif user_input.lower().startswith("ner "):
        try:
            index = int(user_input.split(" ")[1])
            if 0 <= index < len(all_texts):
                print(f"\n🔍 Named Entities & Flashcards from chunk {index + 1}:\n")
                chunk_text = all_texts[index].page_content
                entities = extract_named_entities(chunk_text)
                flashcards = format_entities_as_flashcards(entities)
                for i, (q, a) in enumerate(flashcards, 1):
                    print(f"{i}. Q: {q}\n   A: {a}")
            else:
                print("❌ Invalid chunk index.")
        except (IndexError, ValueError):
            print("❌ Usage: ner <chunk_index>")
    
    elif user_input.lower().startswith("nerpage"):
        try:
            page_index = int(user_input.split(" ")[1])
            if 0 <= page_index < len(all_documents):
                print(f"\n📄 Named Entities from page {page_index + 1}:\n")
                page_text = all_documents[page_index].page_content
                entities = extract_named_entities(page_text)
                flashcards = format_entities_as_flashcards(entities)
                for i, (q, a) in enumerate(flashcards, 1):
                    print(f"{i}. Q: {q}\n   A: {a}")
            else:
                print("❌ Invalid page index.")
        except (IndexError, ValueError):
            print("❌ Usage: nerpage <page_index>")
    elif user_input.lower().startswith("plan "):
        try:
            _, topic, date = user_input.split(" ", 2)
            print("🧠 Generating Study Plan...\n")
            plan = create_study_plan(topic, date)
            print(plan)
        except ValueError:
            print("❌ Usage: plan <topic> <YYYY-MM-DD>")
    else:
        book_name, cleaned_query = extract_book_name_and_query(user_input)
        if book_name:
            print(f"🔍 Searching only in book: {book_name}")
            filtered_retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5, "filter": {"book": book_name}}
            )
            filtered_chain = RetrievalQA.from_chain_type(llm=llm, retriever=filtered_retriever)
            response = filtered_chain.invoke({"query": cleaned_query})
        else:
            response = chain.invoke({"query": cleaned_query})
        
        answer = response["result"].strip()
        if answer.lower() in ["i don't know", "i do not know", ""]:
            print("🤷 Sorry, I couldn't find a relevant answer in the uploaded books.")
            speak("Sorry, I couldn't find a relevant answer in the uploaded books.")
        else:
            print("🧠 Answer:", answer, "\n")
            speak(answer)
