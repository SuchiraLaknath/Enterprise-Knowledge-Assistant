"""Clean a finished answer before it is rendered as markdown (rule 9).

A rendered image makes the browser fetch its URL with no click, and a link needs only one, so
either could send data to an attacker. The poisoned runbook in the corpus tries exactly this.
Answers never need a web address, because sources are listed separately, so every address is
removed. That one rule also defeats the many markdown spellings of an image or link, which a
pattern for each spelling would miss. Build step 8 adds brand and validity checks here.
"""

import re

# A web address, including the protocol-relative and www forms that browsers also follow.
WEB_ADDRESS_PATTERN = re.compile(r"(?:https?:)?//[^\s)>\]]+|\bwww\.[^\s)>\]]+", re.IGNORECASE)
# A markdown image such as ![alt text](address "title").
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# The empty remains of a link once its address has gone, such as [text]([link removed]).
LINK_WITHOUT_ADDRESS_PATTERN = re.compile(r"\[([^\]]*)\]\(\[link removed\][^)]*\)")

REMOVED_LINK_NOTICE = "[link removed]"
REMOVED_IMAGE_NOTICE = "[image removed]"


def clean_final_answer(answer: str) -> str:
    """Remove everything in an answer that could reach an outside address."""
    # Step 1: remove addresses first, because a crafted image can survive a pattern for images
    # and be rebuilt by it, as in !![x](y)(address).
    answer_without_addresses = WEB_ADDRESS_PATTERN.sub(REMOVED_LINK_NOTICE, answer)

    # Step 2: tidy what is left of images and links.
    answer_without_images = MARKDOWN_IMAGE_PATTERN.sub(
        REMOVED_IMAGE_NOTICE, answer_without_addresses
    )
    answer_with_plain_links = LINK_WITHOUT_ADDRESS_PATTERN.sub(r"\1", answer_without_images)

    # Step 3: escape any remaining image syntax, so that no unusual form can render as an image.
    return answer_with_plain_links.replace("![", "!\\[")
