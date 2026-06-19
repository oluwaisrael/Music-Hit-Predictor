import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_theme(style="whitegrid")

# ==========================================
# 1. LOAD DATA & INITIAL CLEANING
# ==========================================
print("--- Starting Exploratory Data Analysis & Feature Engineering ---")

# Load the raw dataset
try:
    df = pd.read_csv("afrobeats_tracks.csv")
    print(f"Successfully loaded dataset. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'afrobeats_tracks.csv' not found. Please verify the filename.")
    exit()

# Handle missing values in critical columns
# Filling missing release months with 1 (January) as a baseline configuration
if 'release_month' in df.columns:
    df['release_month'] = df['release_month'].fillna(1).astype(int)

# Ensure data types are consistent
if 'release_year' in df.columns:
    df['release_year'] = df['release_year'].astype(int)


# ==========================================
# 2. FEATURE ENGINEERING PIPELINE
# ==========================================
print("\n--- Running Feature Engineering Pipeline ---")

# A. Convert duration from milliseconds to seconds for clean scale interpretation
if 'duration_ms' in df.columns:
    df['duration_secs'] = df['duration_ms'] / 1000.0

# B. Extract seasonal release markers if you want to explore Q1-Q4 impacts
if 'release_month' in df.columns:
    df['release_quarter'] = pd.cut(
        df['release_month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=[1, 2, 3, 4]
    ).astype(int)

# C. Calculate an interaction feature: tracks relative to album age.
# Using 2026 baseline and adding +1 to prevent zero-division errors for 2026 drops.
if 'total_tracks_on_album' in df.columns and 'release_year' in df.columns:
    df['tracks_per_year'] = df['total_tracks_on_album'] / (2026 - df['release_year'] + 1)


# ==========================================
# 3. VISUALIZATIONS (EXPLORATORY)
# ==========================================
print("\n--- Generating and Saving Insights/Plots ---")

# Visualization 1: Popularity Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['popularity'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Track Popularity')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('popularity_dist.png')
plt.close()

# Visualization 2: Explicit content flag vs Popularity
if 'explicit' in df.columns:
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='explicit', y='popularity', data=df, palette='Set2')
    plt.title('Explicit Content vs Track Popularity')
    plt.xlabel('Is Explicit')
    plt.ylabel('Popularity Score')
    plt.tight_layout()
    plt.savefig('explicit_vs_popularity.png')
    plt.close()


# ==========================================
# 4. PREPARE TRAINING FEATURE SET & SAVE
# ==========================================
# Drop original raw text or redundant columns that aren't safe or useful for modeling.
# We retain metadata tracking attributes inside features_df briefly for your sanity prints,
# then write the final set to the training file.
columns_to_drop = []
if 'duration_ms' in df.columns: columns_to_drop.append('duration_ms')
if 'release_date' in df.columns: columns_to_drop.append('release_date')

features_df = df.drop(columns=columns_to_drop)

# ==========================================
# 5. VERIFY NEW FEATURES & SAVE
# ==========================================
print("\n=== Verify New Engineered Data ===")
print("New Engineered DataFrame Columns:")
print(features_df.columns.tolist())

print("\nEngineered Dataset Sample:")
preview_cols = [c for c in ['name', 'artist', 'duration_secs', 'release_year', 'release_month'] if c in features_df.columns]
if preview_cols:
    print(features_df[preview_cols].head())

# Save the clean dataset to disk for model.py
features_df.to_csv("cleaned_afrobeats_tracks.csv", index=False)
print("\nEDA complete. Plots saved and 'cleaned_afrobeats_tracks.csv' is generated!")