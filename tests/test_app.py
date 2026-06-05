import unittest
import shutil
from pathlib import Path
from src.chunking.chunker import RecursiveCharacterTextSplitter
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.ingestion.loader import load_document

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        # Create temp folder for testing
        self.test_dir = Path(__file__).parent / "temp_test_data"
        self.test_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        # Clean up temp folder
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_recursive_chunker(self):
        """Tests that the chunker splits long text and matches configuration constraints."""
        text = "This is sentence one. " * 30  # ~600 chars
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        chunks = splitter.split_text(text)
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 150)
            self.assertTrue(len(chunk) > 0)

    def test_offline_embeddings(self):
        """Tests that the hashing trick embedder generates correct dimension shapes offline."""
        embedder = Embedder(provider="fallback", model="text-embedding-004")
        texts = ["hello world", "rag database design"]
        embs = embedder.embed_documents(texts)
        
        self.assertEqual(len(embs), 2)
        # text-embedding-004 fallback mapping uses 768 dimensions
        self.assertEqual(len(embs[0]), 768)
        self.assertEqual(len(embs[1]), 768)
        
        # Test unit length (L2 normalization)
        norm = sum(x**2 for x in embs[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_fallback_vector_store(self):
        """Tests adding, querying, and clearing the persisted JSON database fallback."""
        store = VectorStore(provider="fallback", persist_dir="tests/temp_test_data/db", collection_name="test_col")
        
        documents = [
            {"content": "Deep learning models are neural networks.", "metadata": {"filename": "test1.txt"}},
            {"content": "RAG architectures use vector databases for similarity search.", "metadata": {"filename": "test2.txt"}}
        ]
        
        embeddings = [
            [1.0 if i == 0 else 0.0 for i in range(768)],
            [1.0 if i == 1 else 0.0 for i in range(768)]
        ]
        
        # Add docs
        store.add_documents(documents, embeddings)
        
        # Verify docs in listing
        all_docs = store.get_all_documents()
        self.assertEqual(len(all_docs), 2)
        
        # Query matching doc 1 (exact vector matching query embedding)
        q_emb = [1.0 if i == 0 else 0.0 for i in range(768)]
        results = store.query(q_emb, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["filename"], "test1.txt")
        self.assertAlmostEqual(results[0]["score"], 1.0)
        
        # Query matching doc 2
        q_emb_2 = [0.0, 1.0] + [0.0]*766
        results_2 = store.query(q_emb_2, top_k=1)
        self.assertEqual(results_2[0]["metadata"]["filename"], "test2.txt")
        
        # Clear store
        store.clear()
        self.assertEqual(len(store.get_all_documents()), 0)

    def test_document_loader_txt(self):
        """Tests that text file parser extracts content correctly."""
        txt_file = self.test_dir / "test.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("FastAPI is a modern web framework.")
            
        docs = load_document(str(txt_file))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "FastAPI is a modern web framework.")
        self.assertEqual(docs[0]["metadata"]["filename"], "test.txt")
        self.assertEqual(docs[0]["metadata"]["type"], "txt")

if __name__ == "__main__":
    unittest.main()
