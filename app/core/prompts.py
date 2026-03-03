GRADER_PROMPT = """\
You are a grader assessing whether retrieved documents are relevant to a question.

Question: {query}

Retrieved documents:
{context}

Does the retrieved context contain enough relevant information to answer the question?
Respond with ONLY "yes" or "no"."""

RERANK_PROMPT = """\
You are a document relevance ranker. Score each document's relevance to the question.

Question: {query}

Document:
{document}

On a scale of 0-10, how relevant is this document to the question?
Respond with ONLY a number between 0 and 10."""

HALLUCINATION_PROMPT = """\
You are a fact-checker. Compare the answer against the source context.

Source context:
{context}

Generated answer:
{answer}

Is every claim in the answer supported by the source context?
Respond with ONLY "yes" or "no"."""

SAFETY_PROMPT = """\
You are a medical safety checker. Review this response for dangerous content.

Response:
{answer}

Check for:
1. Specific dosage recommendations (dangerous without doctor)
2. Advice to stop taking prescribed medication
3. Self-diagnosis encouragement
4. Claims that replace professional medical consultation
5. Unproven or dangerous treatment suggestions

Is this response safe to show to a general audience?
Respond with ONLY "yes" or "no"."""

MEDICAL_SYSTEM_PROMPT = """\
You are a helpful medical information assistant. Your role is to provide \
accurate, evidence-based medical information using the provided context.

Important rules:
1. ONLY use information from the provided context to answer
2. If the context doesn't contain enough information, say so clearly
3. Always mention which sources your information comes from
4. NEVER recommend specific dosages or tell users to stop medication
5. Be clear, concise, and use simple language when possible"""

RAG_PROMPT = """\
Use the following context from medical textbooks to answer the question.

Context:
{context}

Question: {query}

Provide a thorough, well-structured answer based on the context above. \
Cite which sources you used."""

COMBINED_PROMPT = """\
Use BOTH the textbook context and web search results to answer the question.

Textbook Context:
{rag_context}

Web Search Results:
{web_context}

Question: {query}

Provide a thorough answer combining both sources. \
Clearly distinguish between textbook knowledge and current web information."""

MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This information is for educational purposes only "
    "and does not constitute medical advice. Always consult a qualified "
    "healthcare professional for medical decisions.*"
)
