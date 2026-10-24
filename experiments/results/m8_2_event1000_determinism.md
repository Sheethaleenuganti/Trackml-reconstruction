# M8.2 Event-1000 Determinism Check

Two complete event-1000 loads, feature constructions, and frozen DBSCAN runs were compared.

- Raw DBSCAN labels exactly equal: **True**
- Canonical hit-to-cluster mappings equal: **True**
- Canonical hit-to-submission-track mappings equal: **False**
- Scores equal: **True**
- Raw hit-ID and feature input hashes equal: **True**
- First raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- Second raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- First canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- Second canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- First saved-submission SHA-256: `21991d5e4085b0b2615a17f5d0462fbd7bcd4c45f9f48be5c80c58b5ff8e4283`
- Second submission SHA-256: `6f1f0de0794d70a88805e26a05de520bc15fe340e9fad1b5b3884ea1704dfb0b`
- First score: `0.13625134527683258`
- Second score: `0.13625134527683258`

---

## Corrected dtype-normalized comparison

The initial comparison above treated TrackML input `hit_id` values (int32) and CSV-round-tripped values (int64) as unequal because pandas `DataFrame.equals` is dtype-sensitive. The verification below normalizes IDs and compares mapping identity. First-run artifacts were preserved unchanged.

# M8.2 Event-1000 Determinism Check

Two complete event-1000 loads, feature constructions, and frozen DBSCAN runs were compared.

- Raw DBSCAN labels exactly equal: **True**
- Canonical hit-to-cluster mappings equal: **True**
- Canonical hit-to-submission-track mappings equal: **True**
- Scores equal: **True**
- Raw hit-ID and feature input hashes equal: **True**
- First raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- Second raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- First canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- Second canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- First saved-submission mapping SHA-256: `2d40716b812ca201239ad22480a0bfce4b943eb42b8361229c50fabf72fb14c9`
- Second submission mapping SHA-256: `2d40716b812ca201239ad22480a0bfce4b943eb42b8361229c50fabf72fb14c9`
- First score: `0.13625134527683258`
- Second score: `0.13625134527683258`

---

## Corrected dtype-normalized comparison

The initial comparison above treated TrackML input `hit_id` values (int32) and CSV-round-tripped values (int64) as unequal because pandas `DataFrame.equals` is dtype-sensitive. The verification below normalizes IDs and compares mapping identity. First-run artifacts were preserved unchanged.

# M8.2 Event-1000 Determinism Check

Two complete event-1000 loads, feature constructions, and frozen DBSCAN runs were compared.

- Raw DBSCAN labels exactly equal: **True**
- Canonical hit-to-cluster mappings equal: **True**
- Canonical hit-to-submission-track mappings equal: **True**
- Scores equal: **True**
- Raw hit-ID and feature input hashes equal: **True**
- First raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- Second raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- First canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- Second canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- First saved-submission mapping SHA-256: `2d40716b812ca201239ad22480a0bfce4b943eb42b8361229c50fabf72fb14c9`
- Second submission mapping SHA-256: `2d40716b812ca201239ad22480a0bfce4b943eb42b8361229c50fabf72fb14c9`
- First score: `0.13625134527683258`
- Second score: `0.13625134527683258`

---

## Corrected dtype-normalized comparison

The initial comparison above treated TrackML input `hit_id` values (int32) and CSV-round-tripped values (int64) as unequal because pandas `DataFrame.equals` is dtype-sensitive. The verification below normalizes IDs and compares mapping identity. First-run artifacts were preserved unchanged.

# M8.2 Event-1000 Determinism Check

Two complete event-1000 loads, feature constructions, and frozen DBSCAN runs were compared.

- Raw DBSCAN labels exactly equal: **True**
- Canonical hit-to-cluster mappings equal: **True**
- Canonical hit-to-submission-track mappings equal: **True**
- Scores equal: **True**
- Raw hit-ID and feature input hashes equal: **True**
- First raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- Second raw label SHA-256: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- First canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- Second canonical mapping SHA-256: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- First saved-submission mapping SHA-256: `2d40716b812ca201239ad22480a0bfce4b943eb42b8361229c50fabf72fb14c9`
- Second submission mapping SHA-256: `2d40716b812ca201239ad22480a0bfce4b943eb42b8361229c50fabf72fb14c9`
- First score: `0.13625134527683258`
- Second score: `0.13625134527683258`
