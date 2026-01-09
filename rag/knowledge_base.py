from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

class MathKnowledgeBase:
    def __init__(self, kb_dir: str = "rag/kb_docs"):
        self.kb_dir = kb_dir
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def load_and_index_docs(self):
        """Load all docs from kb_dir and create vector store"""
        docs_text = []
        
        # Load all text files
        for file_path in Path(self.kb_dir).glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                docs_text.append(f.read())
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_chunks = []
        for doc in docs_text:
            chunks = splitter.split_text(doc)
            all_chunks.extend(chunks)
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            all_chunks,
            embedding=self.embeddings
        )
        
        # Save for later use
        self.vector_store.save_local("rag/faiss_index")
        print(f"✅ Indexed {len(all_chunks)} chunks")
    
    def load_index(self):
        """Load saved vector store"""
        if os.path.exists("rag/faiss_index"):
            self.vector_store = FAISS.load_local(
                "rag/faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
    
    def retrieve(self, query: str, k: int = 3):
        """Retrieve top-k relevant docs"""
        if self.vector_store is None:
            self.load_index()
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results  # [(doc, score), ...]

# Usage in main app
kb = MathKnowledgeBase()
kb.load_and_index_docs()
