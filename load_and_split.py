from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_and_split(pdf_path):
    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages.")

    print("🔪 Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"✅ Split into {len(split_docs)} chunks.")
    return split_docs
