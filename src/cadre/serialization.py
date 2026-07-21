from __future__ import annotations

import html
from collections.abc import Sequence

from .models import Document, TrustedContext


def _escape(value: str) -> str:
    return html.escape(value, quote=False)


def serialize_context(context: TrustedContext, documents: Sequence[Document]) -> str:
    policy = "\n".join(x for x in (context.system, context.developer) if x.strip())
    evidence = "\n\n".join(
        f'<DOC id="{_escape(doc.id)}" source="{_escape(doc.source)}">\n{_escape(doc.text)}\n</DOC>'
        for doc in documents
    )
    return (
        f"<POLICY>\n{_escape(policy)}\n</POLICY>\n"
        f"<REQUEST>\n{_escape(context.user)}\n</REQUEST>\n"
        f"<EVIDENCE>\n{evidence}\n</EVIDENCE>"
    )
