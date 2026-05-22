# backend/generation/chain.py
# ─────────────────────────────────────────────────────
# PURPOSE: Connect the LLM to our prompt template
# and build the generation component of the RAG chain.
# This file wires together: Prompt → LLM → Output Parser
# ─────────────────────────────────────────────────────

import logging
import time
from typing import Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
# ChatOpenAI: LangChain's wrapper around OpenAI's
# chat completion API (gpt-4o-mini, gpt-4o, etc.)
# It handles:
#   - API authentication
#   - Request formatting (messages list → API payload)
#   - Response parsing (API response → string)
#   - Retry on rate limits (with exponential backoff)
#   - Streaming support (token-by-token output)
#
# Alternative — local LLMs via Ollama:
#   from langchain_community.llms import Ollama
#   llm = Ollama(model="llama3.1")
#   EVERYTHING ELSE in this file stays IDENTICAL.
#   This is LangChain's greatest strength — swap LLMs
#   by changing one line.

from langchain.schema.output_parser import StrOutputParser
# StrOutputParser: Takes the LLM's response object
# and extracts just the text string from it.
# Without this, you'd get an AIMessage object with
# many fields — we just want the content string.

from langchain.schema.runnable import RunnablePassthrough
# RunnablePassthrough: passes its input unchanged.
# Used in chain composition to route data through
# multiple steps. Think of it as a wire connector.

from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)
# StreamingStdOutCallbackHandler: prints LLM tokens
# to stdout as they're generated (like ChatGPT's
# streaming effect). Used during development to see
# response forming in real time — great for debugging
# slow responses.

from backend.config import OPENAI_API_KEY, OPENAI_MODEL
from backend.generation.prompt import get_rag_prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Structured container for LLM generation output.

    Same design philosophy as RetrievalResult (Phase 6) —
    return rich structured data, not raw strings.
    Callers can inspect metadata without parsing text.
    """

    answer: str
    # The LLM's generated answer text.

    question: str
    # Original question — for logging and API response.

    context_used: str
    # The context that was given to the LLM.
    # Returned so the API can show users WHAT was
    # retrieved, not just what was answered.

    model_used: str
    # Which LLM generated this answer.
    # Important for cost tracking and debugging.

    generation_time: float
    # Seconds taken to generate the response.
    # Track this — if consistently > 5s, consider
    # smaller model or streaming.

    token_estimate: int = 0
    # Rough token count of the prompt + response.
    # Used for cost estimation. Not exact without
    # calling the tokenizer — this is an approximation.

    @property
    def is_uncertain(self) -> bool:
        """
        True if the LLM indicated it couldn't answer
        from the provided context.
        Detects our specific "I don't have enough
        information" response from the system prompt.
        """
        return "i don't have enough information" in \
               self.answer.lower()
        # .lower() for case-insensitive matching.
        # LLMs sometimes capitalize differently.


class LLMChain:
    """
    Manages the LLM and generation pipeline.

    Responsibilities:
      1. Initialize and configure the LLM
      2. Connect prompt template → LLM → output parser
      3. Generate answers from context + question
      4. Track performance metrics

    Design: We build a LangChain "LCEL" chain using
    the pipe operator (|). This is LangChain Expression
    Language — a functional, composable way to build
    pipelines.
    """

    def __init__(
        self,
        model_name: str = OPENAI_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        streaming: bool = False,
    ):
        """
        Args:
            model_name:  OpenAI model identifier.
                         "gpt-4o-mini" — fast, cheap, good
                         "gpt-4o"      — best quality, expensive
                         "gpt-3.5-turbo" — fastest, cheapest

            temperature: Controls randomness of output.
                         0.0 = deterministic (same input →
                               same output, every time)
                         1.0 = very creative/random
                         For RAG: ALWAYS use 0.0
                         We want FACTUAL retrieval, not creativity.
                         Higher temperature increases hallucination
                         risk — exactly what RAG is trying to solve.

            max_tokens:  Maximum tokens in the generated response.
                         1024 ≈ 750 words — enough for most answers.
                         Prevents runaway expensive responses.
                         Set lower for APIs with cost constraints.

            streaming:   If True, print tokens as they generate.
                         Use during development for faster feedback.
                         Use in production with async streaming
                         to improve perceived latency for users.
        """

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            f"Initializing LLM | model={model_name} | "
            f"temperature={temperature} | "
            f"max_tokens={max_tokens}"
        )

        # ── Build the LLM ─────────────────────────────────
        callbacks = []
        if streaming:
            callbacks.append(StreamingStdOutCallbackHandler())
            # Add streaming callback only if requested.
            # In production FastAPI, use a different
            # streaming approach (async generators).

        self.llm = ChatOpenAI(
            model=model_name,
            # Which model to call. Passed directly to
            # OpenAI API as the "model" field.

            temperature=temperature,
            # 0.0 for RAG — we want deterministic,
            # factual answers, not creative variations.

            max_tokens=max_tokens,
            # Hard cap on response length.

            openai_api_key=OPENAI_API_KEY,
            # From config — never hardcoded.

            callbacks=callbacks,
            # List of callback handlers for events
            # like token streaming, errors, etc.

            request_timeout=30,
            # Seconds before giving up on the API call.
            # OpenAI can occasionally be slow — 30s is
            # generous but prevents hanging forever.
        )

        # ── Build the LCEL chain ──────────────────────────
        self.prompt = get_rag_prompt()
        # Our carefully engineered prompt from prompt.py

        self.output_parser = StrOutputParser()
        # Extracts plain string from LLM response object.

        # The chain: prompt | llm | parser
        # This is LangChain Expression Language (LCEL).
        # The | operator chains Runnables together.
        # Data flows left to right:
        #   dict{"context","question"}
        #     → prompt.format_messages() → List[Message]
        #     → llm(messages) → AIMessage
        #     → parser(message) → str
        self.chain = self.prompt | self.llm | self.output_parser

        logger.info("✅ LLM chain ready.")

    def generate(
        self,
        question: str,
        context: str,
    ) -> GenerationResult:
        """
        Generate an answer for a question given context.

        This is called AFTER retrieval — context is the
        formatted string from Retriever.format_context()
        in Phase 6.

        Args:
            question: User's natural language question
            context:  Retrieved and formatted context chunks

        Returns:
            GenerationResult with answer + metadata
        """

        logger.info(
            f"Generating answer | "
            f"question='{question[:60]}' | "
            f"context_length={len(context)} chars"
        )

        start_time = time.time()

        # ── Invoke the chain ──────────────────────────────
        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })
        # chain.invoke() runs the full pipeline:
        # 1. Formats the prompt with context + question
        # 2. Sends to OpenAI API
        # 3. Receives AIMessage response
        # 4. Parses to plain string
        # Returns: str (the LLM's answer)
        #
        # Alternative: chain.stream() for streaming
        # Alternative: chain.ainvoke() for async (FastAPI)

        generation_time = time.time() - start_time

        # ── Estimate token usage ──────────────────────────
        prompt_text = context + question
        token_estimate = len(prompt_text.split()) * 1.3
        # Rough approximation: words × 1.3 ≈ tokens
        # (English averages ~0.75 tokens per word)
        # For exact count: use tiktoken library.
        # We keep this rough to avoid adding a dependency.

        result = GenerationResult(
            answer=answer.strip(),
            # .strip() removes leading/trailing whitespace
            # that LLMs sometimes add.

            question=question,
            context_used=context,
            model_used=self.model_name,
            generation_time=round(generation_time, 3),
            token_estimate=int(token_estimate),
        )

        logger.info(
            f"✅ Generated in {generation_time:.2f}s | "
            f"answer_length={len(answer)} chars | "
            f"uncertain={result.is_uncertain}"
        )

        return result

    def generate_with_sources(
        self,
        question: str,
        context: str,
        source_documents: list,
    ) -> dict:
        """
        Generate an answer AND return source attribution.

        Used by the API layer to return a rich response:
          - The answer
          - Which documents were used
          - Confidence scores per source
          - Generation metadata

        Args:
            question:         User question
            context:          Formatted context string
            source_documents: List[Document] from retriever

        Returns:
            dict with answer + full source attribution
        """

        result = self.generate(question, context)

        # Build source attribution list
        sources = []
        for doc in source_documents:
            sources.append({
                "file_name": doc.metadata.get(
                    "file_name", "unknown"
                ),
                "page": doc.metadata.get("page", "?"),
                "chunk_preview": doc.page_content[:150] + "...",
                # First 150 chars as a preview — enough for
                # users to recognize the source without
                # overwhelming the response.
            })

        return {
            "answer":          result.answer,
            "question":        result.question,
            "sources":         sources,
            "model_used":      result.model_used,
            "generation_time": result.generation_time,
            "is_uncertain":    result.is_uncertain,
            "token_estimate":  result.token_estimate,
        }