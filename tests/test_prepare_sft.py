from minillm_forge.cli.prepare_sft import extract_record


def test_extract_record_accepts_solution_lists():
    assert extract_record({"problem": "2+2?", "solutions": ["Final answer: 4"]}) == {
        "problem": "2+2?",
        "solution": "Final answer: 4",
    }


def test_extract_record_accepts_chat_messages():
    record = extract_record(
        {
            "messages": [
                {"role": "user", "content": "3+3?"},
                {"role": "assistant", "content": "Final answer: 6"},
            ]
        }
    )
    assert record == {"problem": "3+3?", "solution": "Final answer: 6"}
