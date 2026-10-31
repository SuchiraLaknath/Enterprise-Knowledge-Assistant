"""Remove text from retrieved content that speaks to the assistant instead of to a reader.

Rule 5: retrieved text is data. Document tags say so, but a sentence such as "ignore all previous
instructions" should not reach a model at all. Only the offending sentence is removed, so that a
poisoned runbook still gives its useful steps. Research claims, labels, summaries and tool results
pass through here as well, because a poisoned document can travel through a sub-agent's words.
"""

import re

from security.policies import find_matching_category, get_security_policies

# Splits after sentence endings, keeping the ending with its sentence. A closing bracket counts,
# because an injected markdown image often has no full stop after it.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?)])\s+|\n+")
REMOVED_TEXT_NOTICE = "[text removed by the content guard]"


def contains_injection(text: str) -> bool:
    """Return whether any part of this text speaks to the assistant."""
    return find_matching_category(text, get_security_policies().content) is not None


def remove_injected_sentences(text: str) -> str:
    """Return the text with each injected sentence replaced by a visible notice."""
    # The whole text is checked first, because nearly all of it is clean.
    if not contains_injection(text):
        return text

    injection_patterns = get_security_policies().content
    cleaned_sentences = []
    for sentence in SENTENCE_BOUNDARY.split(text):
        if find_matching_category(sentence, injection_patterns):
            cleaned_sentences.append(REMOVED_TEXT_NOTICE)
        else:
            cleaned_sentences.append(sentence)
    return " ".join(cleaned_sentences)
