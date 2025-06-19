"""
Configuration file for Advanced RAG System
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RAGConfig:
    """Configuration class for RAG system"""
    
    # Collection settings
    collection_name: str = "my_documents"
    
    # LLM settings
    llm_model: str = "gpt-3.5-turbo"  # tiktoken version only supports gpt-3.5-turbo
    llm_temperature: float = 0.1
    max_tokens: int = 1000
    
    # Retrieval settings
    retrieval_k: int = 6  # Number of documents to retrieve
    retrieval_fetch_k: int = 20  # Number for MMR selection
    search_type: str = "mmr"  # "similarity" or "mmr"
    
    # Text splitting settings
    chunk_size: int = 800
    chunk_overlap: int = 100
    large_doc_chunk_size: int = 1200
    large_doc_chunk_overlap: int = 150
    large_doc_threshold: int = 5000 
    
    # Memory settings
    memory_max_tokens: int = 800
    conversation_summary_enabled: bool = True
    
    # File processing settings
    supported_extensions: List[str] = None
    force_reprocess: bool = False
    
    # UI settings
    show_sources: bool = True
    max_sources_display: int = 3
    source_preview_length: int = 200
    
    # Advanced features
    use_compression: bool = True
    track_costs: bool = True
    enable_logging: bool = True
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [".pdf", ".txt", ".md", ".docx"]
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create config from environment variables"""
        return cls(
            collection_name=os.getenv("RAG_COLLECTION_NAME", "my_documents"),
            llm_model=os.getenv("RAG_LLM_MODEL", "gpt-3.5-turbo"),
            llm_temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("RAG_MAX_TOKENS", "1000")),
            retrieval_k=int(os.getenv("RAG_RETRIEVAL_K", "6")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
            memory_max_tokens=int(os.getenv("RAG_MEMORY_MAX_TOKENS", "800")),
            use_compression=os.getenv("RAG_USE_COMPRESSION", "true").lower() == "true",
            track_costs=os.getenv("RAG_TRACK_COSTS", "true").lower() == "true",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'collection_name': self.collection_name,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'max_tokens': self.max_tokens,
            'retrieval_k': self.retrieval_k,
            'retrieval_fetch_k': self.retrieval_fetch_k,
            'search_type': self.search_type,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'large_doc_chunk_size': self.large_doc_chunk_size,
            'large_doc_chunk_overlap': self.large_doc_chunk_overlap,
            'large_doc_threshold': self.large_doc_threshold,
            'memory_max_tokens': self.memory_max_tokens,
            'conversation_summary_enabled': self.conversation_summary_enabled,
            'supported_extensions': self.supported_extensions,
            'force_reprocess': self.force_reprocess,
            'show_sources': True,
            'max_sources_display': self.max_sources_display,
            'source_preview_length': self.source_preview_length,
            'use_compression': self.use_compression,
            'track_costs': self.track_costs,
            'enable_logging': self.enable_logging,
        }

default_config = RAGConfig()
env_config = RAGConfig.from_env()