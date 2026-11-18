import pytest

from fetcher.rag import LLM, Messages, ask, parse_answer
from fetcher.search import Chunk, Hit

HITS = [Hit(Chunk("csv#0", "https://docs/csv", "csv", "csv.reader reads rows"), 0.9)]
GOOD = '{"answer": "Use csv.reader.", "sources": ["csv#0"], "confident": true}'


def scripted_llm(*replies: str) -> tuple[list[Messages], LLM]:
    """A fake LLM that returns the given replies in order and records what it was sent."""
    calls: list[Messages] = []
    queue = list(replies)

    def llm(messages: Messages) -> str:
        calls.append(messages)
        return queue.pop(0)

    return calls, llm


def test_valid_answer_is_parsed() -> None:
    answer = parse_answer(GOOD, HITS)
    assert answer.sources == ["csv#0"] and answer.confident


@pytest.mark.parametrize(
    "raw",
    [
        "Sure! Use csv.reader.",  # not JSON at all
        '{"answer": "Use csv.reader."}',  # missing fields
        '{"answer": "x", "sources": ["made-up#3"], "confident": true}',  # cites a fake passage
    ],
)
def test_bad_answers_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_answer(raw, HITS)


def test_ask_retries_once_with_the_error() -> None:
    calls, llm = scripted_llm("not json", GOOD)
    answer = ask("how do I read a csv?", HITS, llm)
    assert answer.answer == "Use csv.reader."
    assert len(calls) == 2
    assert "Invalid" in calls[1][-1]["content"]  # second call told the model what went wrong


def test_ask_gives_up_after_max_attempts() -> None:
    _, llm = scripted_llm("nope", "still nope")
    with pytest.raises(RuntimeError):
        ask("q", HITS, llm)
