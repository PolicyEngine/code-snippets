import pandas as pd

from agi_checks_helpers import summarize_young_children


def test_summarize_young_children_uses_entity_weights():
    people = pd.DataFrame(
        {
            "household_id": [1, 1, 2, 3, 4],
            "age": [2, 4, 35, 5, 6],
            "person_weight": [1.5, 1.5, 2.0, 0.75, 3.0],
            "household_weight": [1.5, 1.5, 2.0, 0.75, 3.0],
        }
    )

    summary = summarize_young_children(people)

    assert summary == {
        "persons": 8.75,
        "young_children": 3.75,
        "households_with_young_children": 2.25,
    }


def test_summarize_young_children_returns_zero_for_empty_subset():
    people = pd.DataFrame(
        {
            "household_id": [1, 2],
            "age": [6, 40],
            "person_weight": [1.25, 2.5],
            "household_weight": [1.25, 2.5],
        }
    )

    summary = summarize_young_children(people)

    assert summary == {
        "persons": 3.75,
        "young_children": 0.0,
        "households_with_young_children": 0.0,
    }
