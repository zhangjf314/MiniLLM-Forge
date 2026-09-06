from minillm_forge.data.sft import SFTDataCollator, encode_sft_example


class CharacterTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=True):
        ids = [1] if add_special_tokens else []
        ids.extend(ord(character) % 251 + 4 for character in text)
        return {"input_ids": ids}


def test_sft_masks_system_and_user_but_keeps_assistant():
    tokenizer = CharacterTokenizer()
    encoded = encode_sft_example(tokenizer, user="2+2?", assistant="Final answer: 4")
    first_trainable = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
    assert all(label == -100 for label in encoded["labels"][:first_trainable])
    assert encoded["labels"][first_trainable:] == encoded["input_ids"][first_trainable:]


def test_sft_collator_masks_padding():
    collator = SFTDataCollator(pad_token_id=0, pad_to_multiple_of=4)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        ]
    )
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"][1].tolist() == [-100, 2, -100, -100]
    assert batch["attention_mask"][1].tolist() == [True, True, False, False]
