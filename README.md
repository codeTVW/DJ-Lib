# DJ Library Intelligence

## Installatie in drie stappen

1. Maak een virtuele omgeving aan: `python3.11 -m venv .venv`
2. Activeer de omgeving: `source .venv/bin/activate`
3. Installeer alle dependencies: `pip install -r requirements.txt`

## Gebruik in drie stappen

1. Importeer je lokale muziekbibliotheek: `python3 src/library_importer.py --db-path ./data/db/dj_library.sqlite3`
2. Verrijk de database met analyse: `python3 src/audio_feature_extractor.py --db-path ./data/db/dj_library.sqlite3`, daarna `python3 src/similarity_engine.py --db-path ./data/db/dj_library.sqlite3` en `python3 src/clustering_engine.py --db-path ./data/db/dj_library.sqlite3`
3. Start de desktopapp: `python3 src/main_window.py`

## Modules

- `src/database_init.py`: initialiseert alle vereiste SQLite-tabellen en vult ontbrekende kolommen veilig aan.
- `src/library_importer.py`: detecteert automatisch de beste lokale muziekbron en schrijft unieke tracks weg naar `tracks`.
- `src/audio_feature_extractor.py`: berekent audiofeatures per track en vult alleen ontbrekende featurewaarden aan.
- `src/similarity_engine.py`: bouwt genormaliseerde featurevectoren en schrijft per track de beste similarity-matches weg.
- `src/clustering_engine.py`: zet featurevectoren om naar UMAP-coördinaten en HDBSCAN-clusters met metadata.
- `src/recommendation_engine.py`: geeft voor een track de beste opvolgers terug met mix_type en korte reden.
- `src/crate_generator.py`: maakt functionele crates op basis van audiofeatures, clusters en brugtracks.
- `src/feedback_engine.py`: slaat mixfeedback op en past similarity-scores en mix_types daarop aan.
- `src/collection_analyser.py`: analyseert balans, gaten en patronen in de volledige collectie.
- `src/set_generator.py`: genereert volledige sets voor `club`, `afterhours` en `warmup` trajectories.
- `src/main_window.py`: start de PyQt6-interface met sidebar, clustermap, setbuilder, crates en collectie-analyse.
