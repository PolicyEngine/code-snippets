def summarize_young_children(people):
    young = people[people["age"] < 6]
    young_households = young.drop_duplicates("household_id")

    return {
        "persons": float(people["person_weight"].sum()),
        "young_children": float(young["person_weight"].sum()),
        "households_with_young_children": float(
            young_households["household_weight"].sum()
        ),
    }
