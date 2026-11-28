"""Shared query-aware extractive segment selection."""
import re

_CONTEXT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "give", "in", "is",
    "of", "on", "or", "the", "to", "was", "were", "what", "when", "which", "with", "year",
    "answer", "banker", "decimal", "defined", "financial", "following", "here", "only", "places",
    "question", "s", "you", "your",
}
_HEADER_CUE = re.compile(
    r"\b(?:USD|EUR|GBP|CAD|\$|€|£)\s+(?:thousands?|millions?|billions?|trillions?)\b|"
    r"\b(?:fiscal\s+year|year\s+ended|as\s+of|in\s+thousands?|in\s+millions?|"
    r"in\s+billions?|statement\s+of\s+(?:income|operations|cash\s+flows|financial\s+position))\b",
    re.IGNORECASE,
)


def _context_words(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.casefold()) if word not in _CONTEXT_STOPWORDS]


def _query_pairs(question, query):
    pairs = set()
    for text in (question, query):
        words = _context_words(text)
        for index, first in enumerate(words):
            for second in words[index + 1:index + 5]:
                if first != second and not first.isdigit() and not second.isdigit():
                    pairs.add((first, second))
    return pairs


def _financial_value(token):
    value = token.strip(".,;:()$€£")
    if re.fullmatch(r"(?:19|20)\d{2}", value):
        return False
    return bool(re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?%?", value))


def _selection_segments(question, query, text):
    """Return query-ranked, token-boundary excerpts with nearby period/unit headers."""
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return []
    query_terms = set(_context_words(question))
    query_terms.update(_context_words(query))
    query_pairs = _query_pairs(question, query)
    if not query_terms:
        return []
    token_terms = [set(re.findall(r"[a-z0-9]+", token.group().casefold())) for token in tokens]
    anchors = [i for i, terms in enumerate(token_terms) if terms & query_terms]
    if not anchors:
        # The retriever has already ranked this bounded candidate. Keep its complete
        # text as a semantic-search fallback; never turn it into a prefix slice.
        return [{"excerpt": text.strip(), "score": 0, "segments": [(0, len(tokens))],
                 "selection_reason": "retriever_candidate_fallback"}]

    headers = []
    for match in _HEADER_CUE.finditer(text):
        header_index = next((i for i, token in enumerate(tokens) if token.start() <= match.start() < token.end()), None)
        if header_index is None:
            continue
        start = max(0, header_index - 5)
        end = min(len(tokens), header_index + 13)
        headers.append((start, end))

    candidates = {}
    # If a phrase from the question appears beside financial values, preserve the
    # compact table row directly. This takes advantage of labels such as a requested
    # metric followed by its year columns, without depending on a company or task ID.
    for first, second in sorted(query_pairs):
        for first_index, terms_at_first in enumerate(token_terms):
            if first not in terms_at_first:
                continue
            for second_index in range(first_index + 1, min(len(tokens), first_index + 6)):
                if second not in token_terms[second_index]:
                    continue
                value_indices = [i for i in range(second_index + 1, min(len(tokens), second_index + 17))
                                 if _financial_value(tokens[i].group())][:3]
                if not value_indices:
                    continue
                previous_values = [i for i in range(max(0, first_index - 12), first_index)
                                   if _financial_value(tokens[i].group())]
                row_start = previous_values[-1] + 1 if previous_values else max(0, first_index - 6)
                if first_index - row_start > 12:
                    row_start = max(0, first_index - 6)
                row_end = value_indices[-1] + 1
                nearby = [span for span in headers if span[1] <= row_start and row_start - span[1] <= 500]
                pieces = ([max(nearby, key=lambda span: span[1])] if nearby else []) + [(row_start, row_end)]
                segments = _merge_segments(pieces)
                excerpt, _ = _render_segments(text, segments)
                row_terms = set(_context_words(" ".join(token.group() for token in tokens[row_start:row_end]))) & query_terms
                score = 100 + len(value_indices) * 10 + len(row_terms) * 2
                key = tuple(segments)
                current = candidates.get(key)
                if current is None or score > current["score"]:
                    candidates[key] = {"excerpt": excerpt, "score": score, "segments": segments,
                                       "selection_reason": "query_phrase_numeric_row"}

    for anchor in anchors:
        # The local window includes the label, nearby period columns and following values;
        # the extra tail keeps adjacent numeric cells intact. Token boundaries avoid
        # cutting numbers or units.
        start, end = max(0, anchor - 12), min(len(tokens), anchor + 46)
        matched = set().union(*token_terms[start:end]) & query_terms
        score = sum(3 if not term.isdigit() else 0.5 for term in matched)
        score += sum(1 if not term.isdigit() else 0.15
                     for i in range(start, end) for term in token_terms[i] & query_terms)
        value_tokens = [tokens[i].group() for i in range(start, end)
                        if re.search(r"\d", tokens[i].group())
                        and not re.fullmatch(r"(?:19|20)\d{2}", tokens[i].group().strip(".,;:"))]
        score += min(4, len(value_tokens)) * 1.5
        window_words = _context_words(" ".join(token.group() for token in tokens[start:end]))
        pair_hits = 0
        for first, second in query_pairs:
            first_positions = [i for i, word in enumerate(window_words) if word == first]
            second_positions = [i for i, word in enumerate(window_words) if word == second]
            if any(0 < right - left <= 3 for left in first_positions for right in second_positions):
                pair_hits += 1
        score += min(3, pair_hits) * 5
        nearby = [span for span in headers if span[1] <= start and start - span[1] <= 500]
        pieces = []
        if nearby:
            header = max(nearby, key=lambda span: span[1])
            pieces.append(header)
        pieces.append((start, end))
        # Keep both spans intact and make omitted intervening text explicit.
        deduped = []
        for span in sorted(pieces):
            if deduped and span[0] <= deduped[-1][1]:
                deduped[-1] = (deduped[-1][0], max(deduped[-1][1], span[1]))
            elif not deduped or span != deduped[-1]:
                deduped.append(span)
        excerpt = "\n[… omitted text …]\n".join(
            text[tokens[left].start():tokens[right - 1].end()] for left, right in deduped
        )
        key = tuple(deduped)
        current = candidates.get(key)
        if current is None or score > current["score"]:
            candidates[key] = {"excerpt": excerpt, "score": score, "segments": deduped,
                               "selection_reason": "query_term_and_numeric_value_score"}
    return sorted(candidates.values(), key=lambda item: (-item["score"], item["segments"]))


def _merge_segments(segments):
    merged = []
    for start, end in sorted(segments):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        elif not merged or (start, end) != merged[-1]:
            merged.append((start, end))
    return merged


def _render_segments(text, segments):
    tokens = list(re.finditer(r"\S+", text))
    spans = _merge_segments(segments)
    excerpt = "\n[… omitted text …]\n".join(
        text[tokens[start].start():tokens[end - 1].end()] for start, end in spans
    )
    return excerpt, spans
