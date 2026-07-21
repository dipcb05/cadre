from __future__ import annotations

import html
from collections.abc import Sequence

from .models import (
    ContentOrigin,
    Document,
    MessageSegment,
    Provenance,
    TrustLevel,
    TrustedContext,
)


def _escape(value: str) -> str:
    return html.escape(value, quote=False)


def segment_context(
    context: TrustedContext, documents: Sequence[Document]
) -> list[MessageSegment]:
    segments: list[MessageSegment] = []
    if context.system.strip():
        segments.append(
            MessageSegment(
                content=context.system,
                trust=TrustLevel.TRUSTED,
                origin=ContentOrigin.SYSTEM,
            )
        )
    if context.developer.strip():
        segments.append(
            MessageSegment(
                content=context.developer,
                trust=TrustLevel.TRUSTED,
                origin=ContentOrigin.DEVELOPER,
            )
        )
    segments.append(
        MessageSegment(
            content=context.user,
            trust=TrustLevel.UNTRUSTED,
            origin=ContentOrigin.USER,
        )
    )
    for doc in documents:
        segments.append(
            MessageSegment(
                content=doc.text,
                trust=TrustLevel.UNTRUSTED,
                origin=ContentOrigin.RETRIEVAL,
                provenance=Provenance(
                    source_id=doc.source,
                    retrieval_score=doc.score,
                ),
            )
        )
    return segments


def serialize_context(
    context: TrustedContext, documents: Sequence[Document]
) -> str:
    policy = "\n".join(x for x in (context.system, context.developer) if x.strip())
    evidence = "\n\n".join(
        f'<DOC id="{_escape(doc.id)}" source="{_escape(doc.source)}">\n'
        f"{_escape(doc.text)}\n</DOC>"
        for doc in documents
    )
    return (
        f"<POLICY>\n{_escape(policy)}\n</POLICY>\n"
        f"<REQUEST>\n{_escape(context.user)}\n</REQUEST>\n"
        f"<EVIDENCE>\n{evidence}\n</EVIDENCE>"
    )
