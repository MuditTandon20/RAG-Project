# backend/generation/prompt.py
# ─────────────────────────────────────────────────────
# PURPOSE: Define and manage all prompt templates
# used in the RAG system.
# Separating prompts from logic is a key practice —
# prompts change frequently during tuning, and having
# them in one place prevents hunting through code.
# ─────────────────────────────────────────────────────

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
# LangChain prompt classes:
#
# ChatPromptTemplate: Composes system + human messages
#   into the format chat models expect (role/content pairs).
#   Modern LLMs (GPT-3.5+, Claude) are "chat" models —
#   they expect a conversation format, not a single string.
#
# SystemMessagePromptTemplate: Wraps the system instruction.
#   This is the "judge's rules" — sets LLM behavior.
#   Appears as {"role": "system", "content": "..."}
#
# HumanMessagePromptTemplate: Wraps the user turn.
#   Contains context + question.
#   Appears as {"role": "user", "content": "..."}
#
# PromptTemplate: Simple string template with {variables}.
#   Used for non-chat (completion) models.
#   We use it for building sub-components.


# ══════════════════════════════════════════════════════
# Core RAG Prompt Template
# ══════════════════════════════════════════════════════

# The system message — defines the LLM's behavior
SYSTEM_TEMPLATE = """You are an expert document assistant \
for a RAG (Retrieval-Augmented Generation) system.

Your ONLY job is to answer questions based on the \
CONTEXT provided below.

STRICT RULES you must follow:
1. Answer ONLY using information from the CONTEXT.
2. Do NOT use any outside knowledge or assumptions.
3. If the CONTEXT does not contain enough information \
   to answer, respond EXACTLY with:
   "I don't have enough information in the provided \
   documents to answer this question."
4. Always cite your source using the format:
   (Source: [filename], Page [number])
5. Be concise and direct. No unnecessary preamble.
6. If the question has multiple parts, address each part \
   separately using the context.

Remember: An honest "I don't know" is always better \
than a confident wrong answer."""

# The human message template
# {context} and {question} are template variables —
# LangChain fills them in at runtime.
HUMAN_TEMPLATE = """CONTEXT:
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION: {question}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Answer based strictly on the context above:"""
# The horizontal lines (━━━) visually separate context
# from the question. This isn't just aesthetic —
# research shows LLMs perform better with clear
# structural separators in prompts.
# The instruction at the end ("Answer based strictly...")
# is a "trailing nudge" — a final reminder right before
# the LLM starts generating. Very effective.


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Build and return the main RAG ChatPromptTemplate.

    Returns a template with two variables:
      {context}  → formatted retrieved chunks (from Phase 6)
      {question} → user's natural language question

    The template produces a list of messages in the format:
    [
      {"role": "system", "content": "You are an expert..."},
      {"role": "user",   "content": "CONTEXT:\n...\nQUESTION:..."}
    ]
    This is exactly what OpenAI's chat API expects.
    """

    system_message = SystemMessagePromptTemplate.from_template(
        SYSTEM_TEMPLATE
    )
    # from_template(): parses the string and creates a
    # template object. Validates that no accidental
    # {variables} are left unformatted in the system prompt.
    # Our system prompt has no variables — it's fixed.

    human_message = HumanMessagePromptTemplate.from_template(
        HUMAN_TEMPLATE
    )
    # Our human template HAS variables: {context}, {question}
    # LangChain registers these as required inputs.
    # Calling the chain without them raises a clear error.

    prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message,
    ])
    # Combines both into one template.
    # Order matters: system first, then human.
    # This matches how OpenAI expects chat messages.

    return prompt


# ══════════════════════════════════════════════════════
# Condense/Rephrase Prompt (for follow-up questions)
# ══════════════════════════════════════════════════════

CONDENSE_TEMPLATE = """Given a chat history and a \
follow-up question, rephrase the follow-up question \
to be a standalone question that contains all necessary \
context from the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
# This prompt handles a key RAG limitation:
#
# User: "What is the Model X range?"
# Assistant: "The range is 348 miles."
# User: "What about in cold weather?"  ← ambiguous!
#
# "What about in cold weather?" has no context alone.
# This template asks the LLM to rephrase it as:
# "What is the Model X range in cold weather?"
# BEFORE running retrieval.
# This is called "query condensation" — critical for
# multi-turn conversations with RAG.

def get_condense_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt template for condensing
    follow-up questions into standalone queries.
    Used in conversational RAG (multi-turn).
    """

    return ChatPromptTemplate.from_template(CONDENSE_TEMPLATE)


# ══════════════════════════════════════════════════════
# Prompt Inspector — for debugging
# ══════════════════════════════════════════════════════

def preview_prompt(context: str, question: str) -> str:
    """
    Preview what the final prompt looks like before
    sending to the LLM. Invaluable for debugging.

    Args:
        context:  Formatted context string from Phase 6
        question: User's question

    Returns:
        The fully formatted prompt as a string.
    """

    prompt = get_rag_prompt()

    messages = prompt.format_messages(
        context=context,
        question=question,
    )
    # format_messages() fills in {context} and {question}
    # Returns List[BaseMessage] — the actual message objects.

    # Format for human-readable display
    output = []
    for msg in messages:
        role = msg.__class__.__name__.replace(
            "Message", ""
        ).upper()
        # Extracts "SYSTEM" from "SystemMessage",
        # "HUMAN" from "HumanMessage", etc.

        output.append(f"[{role}]\n{msg.content}")
        output.append("─" * 60)

    return "\n".join(output)