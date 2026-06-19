# Afrobeats Hit Predictor

A machine learning project exploring whether a song's audio characteristics can predict its popularity within the Afrobeats genre.

## Project Summary

This project set out to predict whether an Afrobeats track is a "hit" based on its Spotify audio features (danceability, energy, tempo, valence, etc.).

## Data Pipeline & Pivot

The original plan was to collect live data via the Spotify Web API (`spotify_data.py`). During development, the API stopped returning real popularity scores under the available access tier, and a fallback branch in the script silently substituted synthetic, hand-generated popularity values instead of real ones. This was caught during model evaluation — the artificially generated labels were producing misleadingly high accuracy on an artificially small and imbalanced dataset (51 rows, 80/20 class split).

The project pivoted to a verified, public dataset instead: the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle), filtered to the `afrobeat` genre tag — 1,000 tracks with real popularity scores and real audio features.

## Methodology

- **Target variable:** `is_hit` — defined as tracks with popularity above the genre median (median split, ~55/45 class balance)
- **Features:** 9 standard Spotify audio features — danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, duration, explicit flag
- **Model:** Random Forest Classifier (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`)
- **Evaluation:** 80/20 train-test split, stratified by class, compared against a majority-class baseline

## Results

| Metric | Majority-Class Baseline | Random Forest |
|---|---|---|
| Accuracy | 55.0% | 54.0% |
| Macro F1 | — | 0.53 |

**Top features by importance:** instrumentalness, tempo, acousticness, loudness, duration

## Conclusion

The model performs within noise of the majority-class baseline — audio features alone do not meaningfully predict whether an Afrobeats track outperforms its genre peers in popularity. This result is consistent with established "Hit Song Science" research: once genre is held constant, acoustic properties tend to be weak predictors of popularity, which is more strongly driven by factors not present in this dataset — artist reputation, marketing, and playlist placement.

## Key Takeaways

- A real-world data pipeline failure (API restrictions) was caught and documented rather than hidden, leading to a methodology pivot
- A synthetic-data bug that initially appeared to produce a strong model (82% accuracy) was identified and corrected before drawing conclusions
- The final, honest result is a null finding — which is still a valid and informative outcome in data science

## Files

- `spotify_data.py` — original Spotify Web API collection script (deprecated; superseded by Kaggle data due to API restrictions)
- `model.py` — final modeling pipeline (data loading, feature engineering, training, evaluation)
- `eda.py` — exploratory data analysis
- `spotify_tracks_full.csv` — Kaggle Spotify Tracks Dataset (source data)

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn
python3 model.py
```
