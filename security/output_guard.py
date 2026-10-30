"""Clean a finished answer before it is shown as formatted text.

Rendering a markdown image makes the browser fetch its URL, so an image in an answer could send
data to an attacker's server. The poisoned runbook in the corpus tries exactly this. Build step 8
extends this guard with brand and validity checks.
"""

import re

# Matches a markdown image such as ![alt text](https://example.com/image.png "title").
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
REMOVED_IMAGE_NOTICE = "[image removed]"


def remove_markdown_images(answer: str) -> str:
    """Replace every markdown image, because no legitimate answer needs a remote image."""
    return MARKDOWN_IMAGE_PATTERN.sub(REMOVED_IMAGE_NOTICE, answer)
