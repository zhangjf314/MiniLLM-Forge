from minillm_forge.data.contamination import audit_contamination
from minillm_forge.data.dedup import exact_deduplicate


def test_normalized_dedup_handles_case_space_and_punctuation():
    unique, removed = exact_deduplicate(["Solve: 2 + 2", " solve 2+2! ", "Different"])
    assert unique == ["Solve: 2 + 2", "Different"]
    assert removed == 1


def test_contamination_finds_exact_and_near_collisions():
    report = audit_contamination(
        ["If a shirt costs 20 dollars, what is half?", "Solve x + 2 = 5"],
        ["SOLVE x+2=5!", "A shirt costs 20 dollars; what is half of that price?"],
        threshold=0.35,
        ngram_size=4,
    )
    assert report["collision_count"] >= 2
    assert {item["type"] for item in report["collisions"]} >= {"normalized_exact", "ngram"}
