from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import voyageai
from dotenv import load_dotenv
from groq import Groq

EMBEDDING_MODEL = "voyage-4-large"


@dataclass(frozen=True)
class DocumentChunk:
    source_index: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Settings:
    voyage_api_key: str | None
    groq_api_key: str | None
    groq_model: str
    embedding_candidate_count: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        try:
            candidate_count = int(os.getenv("EMBEDDING_CANDIDATE_COUNT", "2"))
        except ValueError as exc:
            raise ValueError("EMBEDDING_CANDIDATE_COUNT must be a positive integer") from exc
        return cls(
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            embedding_candidate_count=candidate_count,
        )

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("VOYAGE_API_KEY", self.voyage_api_key),
                ("GROQ_API_KEY", self.groq_api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
        if self.embedding_candidate_count < 1:
            raise ValueError("EMBEDDING_CANDIDATE_COUNT must be a positive integer")


class SemanticSearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._voyage: voyageai.Client | None = None
        self._groq: Groq | None = None

    def rerank_documents(self, query: str, documents: Sequence[str]) -> Any:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not documents:
            raise ValueError("at least one document is required")
        return self._voyage_client().rerank(query, list(documents), model="rerank-3", top_k=3)

    def chunk_documents(self, documents: Sequence[str]) -> list[DocumentChunk]:
        """Split documents at paragraph boundaries, or sentence boundaries if unformatted."""
        chunks: list[DocumentChunk] = []

        for source_index, document in enumerate(documents):
            if not document.strip():
                continue
            paragraphs = [
                part.strip() for part in re.split(r"\n\s*\n", document.strip()) if part.strip()
            ]
            if len(paragraphs) == 1:
                paragraphs = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", paragraphs[0])
                    if sentence.strip()
                ]
            chunks.extend(
                DocumentChunk(source_index, chunk_index, text)
                for chunk_index, text in enumerate(paragraphs)
            )

        if not chunks:
            raise ValueError("at least one non-empty document is required")
        return chunks

    def rank_by_embedding(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not documents:
            raise ValueError("at least one document is required")

        client = self._voyage_client()
        document_vectors: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0
        for document in documents:
            token_count = client.count_tokens([document], model=EMBEDDING_MODEL)
            effective_tokens = min(token_count, 32_000)
            if batch and (batch_tokens + effective_tokens > 100_000 or len(batch) >= 1_000):
                document_vectors.extend(
                    client.embed(
                        texts=batch,
                        model=EMBEDDING_MODEL,
                        input_type="document",
                        truncation=True,
                    ).embeddings
                )
                batch = []
                batch_tokens = 0
            batch.append(document)
            batch_tokens += effective_tokens
        if batch:
            document_vectors.extend(
                client.embed(
                    texts=batch,
                    model=EMBEDDING_MODEL,
                    input_type="document",
                    truncation=True,
                ).embeddings
            )
        query_vector = client.embed(
            texts=[query], model=EMBEDDING_MODEL, input_type="query"
        ).embeddings[0]
        scores = [
            sum(
                document_value * query_value
                for document_value, query_value in zip(vector, query_vector, strict=True)
            )
            for vector in document_vectors
        ]
        return sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    def answer_question(self, query: str, context: str) -> str:
        completion = self._groq_client().chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Based on this information: '{context}', answer: {query}",
                }
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return completion.choices[0].message.content

    def _voyage_client(self) -> voyageai.Client:
        if self._voyage is None:
            self._voyage = voyageai.Client(api_key=self.settings.voyage_api_key)
        return self._voyage

    def _groq_client(self) -> Groq:
        if self._groq is None:
            self._groq = Groq(api_key=self.settings.groq_api_key)
        return self._groq


def run_demo() -> None:
    settings = Settings.from_env()
    try:
        settings.validate()
    except ValueError as exc:
        raise SystemExit(f"{exc}. Copy .env.example to .env and add your keys.") from exc

    documents = [
        (
            "This quarter, our company is focused on building new products, increasing "
            "market share, and cutting costs."
        ),
        (
            "20th-century innovations, from radios to smartphones, centered on electronic "
            "advancements."
        ),
        (
            "Photosynthesis in plants converts light energy into glucose and produces "
            "essential oxygen."
        ),
    ]
    query = "What are my company's goals this quarter?"
    service = SemanticSearchService(settings)
    chunks = service.chunk_documents(documents)
    chunk_texts = [chunk.text for chunk in chunks]
    embedding_ranking = service.rank_by_embedding(query, chunk_texts)
    candidate_count = min(settings.embedding_candidate_count, len(embedding_ranking))
    top_embedding_results = embedding_ranking[:candidate_count]
    print("Semantic search results:")
    for rank, (document_index, score) in enumerate(top_embedding_results, 1):
        chunk = chunks[document_index]
        print(
            f"{rank}. Document {chunk.source_index + 1}: {chunk.text[:50]}... "
            f"- Similarity score: {score:.4f}"
        )
    print()

    candidate_indices = [index for index, _ in top_embedding_results]
    candidates = [chunk_texts[index] for index in candidate_indices]
    results = service.rerank_documents(query, candidates).results
    if not results:
        raise SystemExit("Voyage AI returned no ranked documents.")

    print("Reranked results:")
    for index, result in enumerate(results, 1):
        print(f"{index}. Score: {result.relevance_score:.4f} - {result.document[:50]}...")

    answer = service.answer_question(query, results[0].document)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Voyage AI semantic-search demo.")
    parser.add_argument("--demo", action="store_true", help="Run the sample semantic-search demo.")
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return
    run_demo()


if __name__ == "__main__":
    main()
