"""
Synthetic Test Set Generator module.
Uses RAGAS TestsetGenerator to create evaluation datasets from documents,
with a manual fallback for environments where RAGAS generation has issues.
"""

import os
import json
import logging
from typing import List, Dict, Optional

from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)


class TestSetGenerator:
    """
    Generates synthetic QA test sets for RAG evaluation.
    Supports RAGAS-native generation and LLM-based manual generation.
    """

    def __init__(self, testset_size: int = None):
        self.testset_size = testset_size or Config.TESTSET_SIZE
        api_key = Config.OPENAI_API_KEY
        if not api_key or "your-openai" in api_key.lower():
            api_key = "dummy-key-for-initialization"
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0.3,
            api_key=api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=api_key,
        )

    def generate_with_ragas(self, documents: List[Document]) -> List[Dict]:
        """
        Use RAGAS TestsetGenerator to create synthetic QA pairs.

        Args:
            documents: LangChain Document objects to generate from.

        Returns:
            List of dicts with question, ground_truth, and source info.
        """
        try:
            from ragas.testset import TestsetGenerator
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            generator_llm = LangchainLLMWrapper(self.llm)
            generator_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

            generator = TestsetGenerator(
                llm=generator_llm,
                embedding_model=generator_embeddings,
            )

            logger.info(
                f"Generating {self.testset_size} synthetic QA pairs with RAGAS..."
            )
            testset = generator.generate_with_langchain_docs(
                documents, testset_size=self.testset_size
            )

            df = testset.to_pandas()
            results = []
            for _, row in df.iterrows():
                results.append(
                    {
                        "question": row.get("user_input", row.get("question", "")),
                        "ground_truth": row.get(
                            "reference", row.get("ground_truth", "")
                        ),
                    }
                )

            logger.info(f"RAGAS generated {len(results)} QA pairs.")
            return results

        except Exception as e:
            logger.warning(f"RAGAS generation failed: {e}")
            logger.info("Falling back to LLM-based generation...")
            return self.generate_with_llm(documents)

    def generate_with_llm(self, documents: List[Document]) -> List[Dict]:
        """
        Fallback: Use direct LLM calls to generate QA pairs from documents.

        Args:
            documents: LangChain Document objects.

        Returns:
            List of dicts with question and ground_truth.
        """
        all_qa_pairs = []
        # Take a subset of documents to generate from
        sample_docs = documents[: min(len(documents), 20)]
        pairs_per_doc = max(1, self.testset_size // len(sample_docs))

        prompt_template = """Based on the following text, generate {n} diverse question-answer pairs. 
The questions should be specific, factual, and answerable from the text.
The answers should be concise and directly supported by the text.

Text:
{text}

Return ONLY a valid JSON array with objects having "question" and "ground_truth" keys.
Example format:
[
  {{"question": "What is X?", "ground_truth": "X is..."}},
  {{"question": "How does Y work?", "ground_truth": "Y works by..."}}
]

Generate exactly {n} pairs:"""

        for i, doc in enumerate(sample_docs):
            try:
                # Truncate very long documents
                text = doc.page_content[:3000]
                prompt = prompt_template.format(n=pairs_per_doc, text=text)

                response = self.llm.invoke(prompt)
                content = response.content.strip()

                # Extract JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                pairs = json.loads(content)
                if isinstance(pairs, list):
                    all_qa_pairs.extend(pairs)
                    logger.info(
                        f"  Doc {i + 1}/{len(sample_docs)}: "
                        f"generated {len(pairs)} QA pairs"
                    )

            except Exception as e:
                logger.warning(f"  Doc {i + 1}: generation error - {e}")
                continue

        # Trim to requested size
        all_qa_pairs = all_qa_pairs[: self.testset_size]
        logger.info(f"LLM generated {len(all_qa_pairs)} total QA pairs.")
        return all_qa_pairs

    def save_testset(
        self, testset: List[Dict], filepath: Optional[str] = None
    ) -> str:
        """Save generated test set to JSON file."""
        Config.ensure_dirs()
        filepath = filepath or os.path.join(Config.DATA_DIR, "testset.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(testset, f, indent=2, ensure_ascii=False)

        logger.info(f"Test set saved to {filepath} ({len(testset)} pairs)")
        return filepath

    def load_testset(self, filepath: Optional[str] = None) -> List[Dict]:
        """Load a previously saved test set."""
        filepath = filepath or os.path.join(Config.DATA_DIR, "testset.json")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No test set found at {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            testset = json.load(f)

        logger.info(f"Loaded test set from {filepath} ({len(testset)} pairs)")
        return testset

    def generate_and_save(self, documents: List[Document]) -> List[Dict]:
        """Convenience: generate test set and save to disk."""
        testset = self.generate_with_ragas(documents)
        self.save_testset(testset)
        return testset
