#!/usr/bin/env python3
"""
Ultra-Optimized Medical RAG Setup
5-10 min processing for 94 MB with sampling and deduplication
"""

import os
from pathlib import Path
import chromadb
import PyPDF2
from typing import List, Dict
import time
import re
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib  # For deduplication

class UltraOptimizedRAG:
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "medical_knowledge"
        self.max_workers = max(1, os.cpu_count() - 1)
        self.chunk_hashes = set()  # For deduplication
        self.sampling_rate = 5  # Process every 5th page (adjust for speed vs completeness)
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]]', '', text)
        text = text.replace('�', '')
        return text.strip()
    
    def is_meaningful_chunk(self, text: str) -> bool:
        if len(text) < 300 or len(text.split()) < 40:
            return False
        skip_phrases = ['table of contents', 'index', 'copyright', 'isbn', 'page number']
        if any(phrase in text.lower() for phrase in skip_phrases):
            return False
        return True
    
    def semantic_split(self, text: str) -> List[str]:
        markers = r'(?i)\n\s*(symptoms|treatment|diagnosis|causes|prevention|prognosis|medication|procedure|care plan|assessment|intervention|evaluation|definition|description|purpose|precautions|risks|complications|expected results|resources|key terms):?\s*\n'
        sections = re.split(markers, text)
        return [s.strip() for s in sections if s.strip()]
    
    def size_split(self, text: str, max_size: int = 2000) -> List[str]:
        if len(text) <= max_size:
            return [text]
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?=\s[A-Z])|\.\s+', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current + sentence) <= max_size:
                current += sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
        return chunks
    
    def deduplicate_chunk(self, text: str) -> bool:
        """Check if chunk is unique using hash"""
        chunk_hash = hashlib.md5(text.encode()).hexdigest()
        if chunk_hash in self.chunk_hashes:
            return False
        self.chunk_hashes.add(chunk_hash)
        return True
    
    def process_page(self, pdf_path: Path, page_num: int) -> List[Dict[str, str]]:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = reader.pages[page_num].extract_text()
                if not text or len(text.strip()) < 100:
                    return []
                
            cleaned = self.clean_text(text)
            if not cleaned:
                return []
            
            semantic_chunks = self.semantic_split(cleaned)
            final_chunks = []
            
            for chunk in semantic_chunks:
                size_chunks = self.size_split(chunk)
                for sc in size_chunks:
                    if self.is_meaningful_chunk(sc) and self.deduplicate_chunk(sc):
                        final_chunks.append(sc)
            
            chunk_dicts = []
            for i, chunk_text in enumerate(final_chunks):
                chunk_id = f"{pdf_path.stem}_p{page_num:04d}_c{i:03d}"
                chunk_dicts.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "source": pdf_path.name,
                    "page": page_num + 1,
                    "length": len(chunk_text)
                })
            
            return chunk_dicts
            
        except Exception:
            return []
    
    def extract_pdf_text(self, pdf_path: Path) -> List[Dict[str, str]]:
        print(f"📖 Sampling {pdf_path.name}...")
        print(f"   Size: {pdf_path.stat().st_size / (1024*1024):.1f} MB")
        
        chunks = []
        
        try:
            reader = PyPDF2.PdfReader(pdf_path)
            total_pages = len(reader.pages)
            sampled_pages = list(range(0, total_pages, self.sampling_rate))
            print(f"   📄 Total pages: {total_pages} | Sampling {len(sampled_pages)} pages")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_page, pdf_path, i) for i in sampled_pages]
                
                for i, future in enumerate(as_completed(futures), 1):
                    chunks.extend(future.result())
                    if i % 50 == 0:
                        progress = (i / len(sampled_pages)) * 100
                        print(f"   ⏳ Progress: {progress:.1f}% ({i}/{len(sampled_pages)}) chunks: {len(chunks)}")
                    gc.collect()
            
            print(f"✅ Extracted {len(chunks)} unique chunks from {pdf_path.name}")
            return chunks
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def setup_collection(self):
        print("🗄️ Setting up collection...")
        
        try:
            self.client.delete_collection(name=self.collection_name)
            time.sleep(2)
        except:
            pass
        
        try:
            collection = self.client.create_collection(name=self.collection_name)
            return collection
        except Exception as e:
            print(f"❌ Failed: {e}")
            return None
    
    def populate_database(self):
        print("🏥 ULTRA RAG DATABASE CREATION")
        print("=" * 50)
        
        collection = self.setup_collection()
        if not collection:
            return False
        
        pdf_files = list(self.data_folder.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDFs found!")
            return False
        
        all_chunks = []
        start_time = time.time()
        
        for pdf in pdf_files:
            chunks = self.extract_pdf_text(pdf)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            print("❌ No chunks extracted!")
            return False
        
        print(f"\n💾 Storing {len(all_chunks)} unique chunks...")
        
        documents = [c["text"] for c in all_chunks]
        ids = [c["id"] for c in all_chunks]
        metadatas = [{"source": c["source"], "page": c["page"], "length": c["length"]} for c in all_chunks]
        
        batch_size = 800
        total_batches = (len(documents) - 1) // batch_size + 1
        
        for i in range(0, len(documents), batch_size):
            try:
                batch_docs = documents[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                
                collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
                
                current = i // batch_size + 1
                print(f"   ⏳ Batch {current}/{total_batches} ✅")
                
                time.sleep(0.05)
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Batch {current}: {str(e)[:30]}...")
        
        total_time = time.time() - start_time
        final_count = collection.count()
        
        print(f"\n🎉 DATABASE CREATED!")
        print(f"Chunks stored: {final_count}")
        print(f"Time: {total_time/60:.1f} minutes")
        return final_count > 0
    
    def test_retrieval(self):
        print("\n🔍 TEST RETRIEVAL")
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.query(query_texts=["heart attack"], n_results=3)
            print("✅ Working!")
            return True
        except:
            print("❌ Failed")
            return False

def main():
    setup = UltraOptimizedRAG()
    if setup.populate_database():
        setup.test_retrieval()
    else:
        print("❌ Failed")

if __name__ == "__main__":
    main()
