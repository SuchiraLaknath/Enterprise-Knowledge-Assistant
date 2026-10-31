"""Export thumbs-down questions, so that a poor answer can become a golden question.

A person still adds the expected documents, because a rating says that an answer was wrong, not
what the right one was.

Run with: uv run python -m evals.export_candidates
"""

import asyncio
from pathlib import Path

import psycopg
import yaml

from config.settings import get_environment_settings

CANDIDATES_PATH = Path(__file__).parent / "golden_candidates.yaml"
SELECT_THUMBS_DOWN_STATEMENT = """
SELECT DISTINCT role, question FROM answer_feedback WHERE rating = 'down' ORDER BY role, question
"""


async def main() -> None:
    """Write each distinct thumbs-down question to the candidates file."""
    database_url = get_environment_settings().database_url
    async with await psycopg.AsyncConnection.connect(database_url) as connection:
        cursor = await connection.execute(SELECT_THUMBS_DOWN_STATEMENT)
        rows = await cursor.fetchall()

    candidates = [
        {"question": question, "user_role": role, "expected_doc_ids": []} for role, question in rows
    ]
    CANDIDATES_PATH.write_text(yaml.safe_dump({"questions": candidates}, sort_keys=False))
    print(f"Wrote {len(candidates)} candidates to {CANDIDATES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
