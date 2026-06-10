"""
Document loader module.
Fetches Wikipedia articles and prepares them for the RAG pipeline.
"""

import os
import json
import logging
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain.schema import Document

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and chunks documents from Wikipedia for RAG ingestion."""

    def __init__(
        self,
        topics: Optional[List[str]] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.topics = topics or Config.DEFAULT_TOPICS
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.raw_documents: List[Document] = []
        self.chunked_documents: List[Document] = []

    def load_wikipedia_articles(self, max_docs_per_topic: int = 1) -> List[Document]:
        """
        Fetch Wikipedia articles for configured topics.

        Args:
            max_docs_per_topic: Maximum documents to fetch per topic.

        Returns:
            List of LangChain Document objects.
        """
        all_docs = []
        for topic in self.topics:
            try:
                logger.info(f"Loading Wikipedia article: {topic}")
                loader = WikipediaLoader(
                    query=topic,
                    load_max_docs=max_docs_per_topic,
                    doc_content_chars_max=15000,
                )
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_topic"] = topic
                all_docs.extend(docs)
                logger.info(f"  → Loaded {len(docs)} doc(s) for '{topic}'")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load '{topic}': {e}")
                continue

        self.raw_documents = all_docs
        logger.info(f"Total raw documents loaded: {len(all_docs)}")
        return all_docs

    def chunk_documents(
        self, documents: Optional[List[Document]] = None
    ) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.

        Args:
            documents: Documents to chunk. Uses self.raw_documents if None.

        Returns:
            List of chunked Document objects.
        """
        docs = documents or self.raw_documents
        if not docs:
            raise ValueError("No documents to chunk. Load documents first.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        # Add chunk index metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunked_documents = chunks
        logger.info(
            f"Chunked {len(docs)} documents → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def save_documents_metadata(self, filepath: Optional[str] = None):
        """Save document metadata to JSON for reference."""
        Config.ensure_dirs()
        filepath = filepath or os.path.join(Config.DATA_DIR, "documents_meta.json")

        meta = {
            "topics": self.topics,
            "raw_document_count": len(self.raw_documents),
            "chunked_document_count": len(self.chunked_documents),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "documents": [
                {
                    "topic": doc.metadata.get("source_topic", "unknown"),
                    "title": doc.metadata.get("title", "unknown"),
                    "content_length": len(doc.page_content),
                }
                for doc in self.raw_documents
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Document metadata saved to {filepath}")

    def load_and_chunk(self) -> List[Document]:
        """Convenience method: load from Wikipedia and chunk in one step."""
        self.load_wikipedia_articles()
        self.chunk_documents()
        self.save_documents_metadata()
        return self.chunked_documents
