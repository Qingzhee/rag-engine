import os
import hashlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import json
from datetime import datetime

load_dotenv()

class AdvancedDocumentIngestor:
    def __init__(self, collection_name: str = "my_documents"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"]
        )
        self.processed_files_log = "processed_files.json"
        
    def load_processed_files_log(self) -> Dict[str, Any]:
        """Load log of previously processed files"""
        if os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, 'r') as f:
                return json.load(f)
        return {}
    
    def save_processed_files_log(self, log: Dict[str, Any]):
        """Save log of processed files"""
        with open(self.processed_files_log, 'w') as f:
            json.dump(log, f, indent=2)
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content for change detection"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=Distance.COSINE
                    )
                )
                print(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def enhanced_text_splitting(self, documents: List[Document]) -> List[Document]:
        """Enhanced text splitting with overlapping and metadata preservation"""
        
        # Different splitters for different content types
        splitters = {
            'default': RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            ),
            'large': RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", " ", ""]
            )
        }
        
        all_chunks = []
        
        for doc in documents:
            # Choose splitter based on document length
            doc_length = len(doc.page_content)
            splitter = splitters['large'] if doc_length > 5000 else splitters['default']
            
            chunks = splitter.split_documents([doc])
            
            # Add enhanced metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk.page_content),
                    'ingestion_date': datetime.now().isoformat()
                })
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def ingest_documents_with_deduplication(
        self, 
        folder_path: str, 
        file_extensions: List[str] = [".pdf"],
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest documents with deduplication and incremental updates
        """
        self.create_collection_if_not_exists()
        
        processed_log = self.load_processed_files_log()
        all_chunks = []
        stats = {
            'total_files': 0,
            'new_files': 0,
            'updated_files': 0,
            'skipped_files': 0,
            'total_chunks': 0,
            'processing_errors': []
        }
        
        # Get all files 
        all_files = []
        for ext in file_extensions:
            pattern = f"**/*{ext}"
            loader = DirectoryLoader(
                folder_path, 
                glob=pattern,
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            all_files.extend([
                os.path.join(root, file) 
                for root, dirs, files in os.walk(folder_path) 
                for file in files if file.endswith(ext)
            ])
        
        stats['total_files'] = len(all_files)
        
        for file_path in all_files:
            try:
                file_hash = self.get_file_hash(file_path)
                file_key = os.path.relpath(file_path, folder_path)
                
                # Check if file needs processing
                if not force_reprocess and file_key in processed_log:
                    if processed_log[file_key].get('hash') == file_hash:
                        print(f"Skipping {file_key} (unchanged)")
                        stats['skipped_files'] += 1
                        continue
                    else:
                        print(f"Reprocessing {file_key} (modified)")
                        stats['updated_files'] += 1
                else:
                    print(f"Processing new file: {file_key}")
                    stats['new_files'] += 1
                
                # Load document
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    # Add file-level metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source_file': file_key,
                            'file_hash': file_hash,
                            'file_size': os.path.getsize(file_path)
                        })
                    
                    # Enhanced text splitting
                    chunks = self.enhanced_text_splitting(docs)
                    all_chunks.extend(chunks)
                    
                    # Update processed files log
                    processed_log[file_key] = {
                        'hash': file_hash,
                        'processed_date': datetime.now().isoformat(),
                        'chunks_count': len(chunks)
                    }
                    
                    print(f"Processed {file_key}: {len(chunks)} chunks")
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                print(error_msg)
                stats['processing_errors'].append(error_msg)
        
        if not all_chunks:
            print("No new documents to ingest.")
            return stats
        
        # Ingest into Qdrant
        try:
            Qdrant.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
            )
            
            stats['total_chunks'] = len(all_chunks)
            print(f"Successfully ingested {len(all_chunks)} chunks into collection '{self.collection_name}'")
            
            # Save updated log
            self.save_processed_files_log(processed_log)
            
        except Exception as e:
            error_msg = f"Error ingesting into Qdrant: {str(e)}"
            print(error_msg)
            stats['processing_errors'].append(error_msg)
        
        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.vectors_count,
                'status': collection_info.status
            }
        except Exception as e:
            return {'error': str(e)}

def ingest_documents_advanced(
    folder_path: str = "data",
    collection_name: str = "my_documents",
    force_reprocess: bool = False
) -> Dict[str, Any]:
    """Main function to ingest documents with advanced features"""
    
    ingestor = AdvancedDocumentIngestor(collection_name)
    return ingestor.ingest_documents_with_deduplication(
        folder_path=folder_path,
        force_reprocess=force_reprocess
    )

if __name__ == "__main__":
    stats = ingest_documents_advanced("data", "my_documents")
    print("\nIngestion Statistics:")
    print(json.dumps(stats, indent=2))