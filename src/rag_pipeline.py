"""
RAG Pipeline module.
Implements a configurable Retrieval-Augmented Generation pipeline
using LangChain, ChromaDB, and OpenAI.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)

# ── Default RAG Prompt ─────────────────────────────────────
DEFAULT_RAG_PROMPT = """You are a knowledgeable assistant. Answer the question based ONLY on the following context. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    Configurable RAG pipeline with LangChain + ChromaDB + OpenAI.
    Supports parameter sweeps for evaluation experiments.
    """

    def __init__(
        self,
        model_name: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        retrieval_k: int = None,
        prompt_template: str = None,
        collection_name: str = "rag_eval_docs",
    ):
        self.model_name = model_name or Config.OPENAI_MODEL
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.retrieval_k = retrieval_k or Config.RETRIEVAL_K
        self.prompt_template = prompt_template or DEFAULT_RAG_PROMPT
        self.collection_name = collection_name

        # Initialize components
        api_key = Config.OPENAI_API_KEY
        if not api_key or "your-openai" in api_key.lower():
            api_key = "dummy-key-for-initialization"
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            api_key=api_key,
        )
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None

    @property
    def config_summary(self) -> Dict[str, Any]:
        """Return a dictionary summarizing the current pipeline configuration."""
        return {
            "model": self.model_name,
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "prompt_template": self.prompt_template[:80] + "...",
        }

    def ingest_documents(self, documents: List[Document]):
        """
        Embed and store document chunks in ChromaDB.

        Args:
            documents: Pre-chunked LangChain Document objects.
        """
        Config.ensure_dirs()
        logger.info(
            f"Ingesting {len(documents)} chunks into ChromaDB "
            f"(collection: {self.collection_name})"
        )

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=Config.CHROMA_DIR,
            collection_name=self.collection_name,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )
        logger.info("Documents ingested successfully.")

    def load_existing_vectorstore(self):
        """Load a previously persisted ChromaDB vectorstore."""
        if not os.path.exists(Config.CHROMA_DIR):
            raise FileNotFoundError(
                f"No ChromaDB found at {Config.CHROMA_DIR}. "
                "Run ingest_documents() first."
            )

        self.vectorstore = Chroma(
            persist_directory=Config.CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )
        logger.info("Loaded existing vectorstore from disk.")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The user question.

        Returns:
            List of retrieved Document objects.
        """
        if self.retriever is None:
            raise RuntimeError("No retriever available. Ingest documents first.")
        return self.retriever.invoke(query)

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate an answer given a query and retrieved context.

        Args:
            query: The user question.
            context_docs: Retrieved documents to use as context.

        Returns:
            Generated answer string.
        """
        context_text = "\n\n".join(doc.page_content for doc in context_docs)
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_text, "question": query})

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """
        Full RAG pipeline: retrieve → generate.

        Args:
            question: The user question.

        Returns:
            Tuple of (answer_string, list_of_retrieved_documents).
        """
        retrieved_docs = self.retrieve(question)
        answer = self.generate(question, retrieved_docs)
        return answer, retrieved_docs

    def batch_query(
        self, questions: List[str], verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run the RAG pipeline on a batch of questions.

        Args:
            questions: List of question strings.
            verbose: Whether to log progress.

        Returns:
            List of dicts with question, answer, and contexts.
        """
        results = []
        total = len(questions)

        for i, question in enumerate(questions):
            if verbose and (i + 1) % 5 == 0:
                logger.info(f"Processing question {i + 1}/{total}")

            try:
                answer, docs = self.query(question)
                results.append(
                    {
                        "question": question,
                        "answer": answer,
                        "contexts": [doc.page_content for doc in docs],
                    }
                )
            except Exception as e:
                logger.error(f"Error on question {i + 1}: {e}")
                results.append(
                    {
                        "question": question,
                        "answer": f"Error: {str(e)}",
                        "contexts": [],
                    }
                )

        return results

    def cleanup(self):
        """Remove the ChromaDB collection (for fresh experiments)."""
        if self.vectorstore is not None:
            try:
                self.vectorstore.delete_collection()
                logger.info(f"Deleted collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
