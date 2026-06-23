import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

print("--- Starting Exploratory Data Analysis & Feature Engineering ---")


try:
    df = pd.read_csv("afrobeats_tracks.csv")
    print(f"Successfully loaded dataset. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'afrobeats_tracks.csv' not found. Please verify the filename.")
    exit()


if 'release_month' in df.columns:
    df['release_month'] = df['release_month'].fillna(1).astype(int)

# Ensure data types are consistent
if 'release_year' in df.columns:
    df['release_year'] = df['release_year'].astype(int)


print("\n--- Running Feature Engineering Pipeline ---")

if 'duration_ms' in df.columns:
    df['duration_secs'] = df['duration_ms'] / 1000.0

if 'release_month' in df.columns:
    df['release_quarter'] = pd.cut(
        df['release_month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=[1, 2, 3, 4]
    ).astype(int)

if 'total_tracks_on_album' in df.columns and 'release_year' in df.columns:
    df['tracks_per_year'] = df['total_tracks_on_album'] / (2026 - df['release_year'] + 1)



print("\n--- Generating and Saving Insights/Plots ---")


plt.figure(figsize=(8, 5))
sns.histplot(df['popularity'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Track Popularity')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('popularity_dist.png')
plt.close()

if 'explicit' in df.columns:
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='explicit', y='popularity', data=df, palette='Set2')
    plt.title('Explicit Content vs Track Popularity')
    plt.xlabel('Is Explicit')
    plt.ylabel('Popularity Score')
    plt.tight_layout()
    plt.savefig('explicit_vs_popularity.png')
    plt.close()


columns_to_drop = []
if 'duration_ms' in df.columns: columns_to_drop.append('duration_ms')
if 'release_date' in df.columns: columns_to_drop.append('release_date')

features_df = df.drop(columns=columns_to_drop)

print("\n=== Verify New Engineered Data ===")
print("New Engineered DataFrame Columns:")
print(features_df.columns.tolist())

print("\nEngineered Dataset Sample:")
preview_cols = [c for c in ['name', 'artist', 'duration_secs', 'release_year', 'release_month'] if c in features_df.columns]
if preview_cols:
    print(features_df[preview_cols].head())


features_df.to_csv("cleaned_afrobeats_tracks.csv", index=False)
print("\nEDA complete. Plots saved and 'cleaned_afrobeats_tracks.csv' is generated!")