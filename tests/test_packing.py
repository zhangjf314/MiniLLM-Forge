from minillm_forge.data.packing import pack_token_sequences


def test_packing_crosses_document_boundaries_and_pads_tail():
    packed = list(
        pack_token_sequences(
            [[4, 5], [6, 7, 8]],
            4,
            eos_token_id=2,
            pad_token_id=0,
            drop_remainder=False,
        )
    )
    assert packed == [[4, 5, 2, 6], [7, 8, 2, 0]]
