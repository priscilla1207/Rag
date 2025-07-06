from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from load_and_split import load_and_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    pdf_path = "Haunting Adeline.pdf"  # Replace with your PDF file path
    split_docs = load_and_split(pdf_path)

    print("🧠 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.")

    print("💾 Storing vectors using FAISS...")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local("faiss_index")
    print("✅ Vector store saved locally as 'faiss_index'")

if __name__ == "__main__":
    main()
