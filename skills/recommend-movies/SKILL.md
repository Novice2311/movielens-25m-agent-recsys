---
name: recommend-movies
description: Select grounded film recommendations using the MovieLens catalog and explicit genre constraints.
---

1. Read explicit constraints and current memory version.
2. Call Catalog.retrieve with the supplied user ID, allowed genres, exclusions and bounded k.
3. Select distinct IDs from the returned records. Cite only a genre in the same record.
4. Require Catalog.verify and the independent validator before calling the output verified.
5. On an unavailable model, return deterministic candidate records and mark the outcome fallback.
6. Never infer runtime, country, awards or plot: MovieLens movies.csv does not provide them.
