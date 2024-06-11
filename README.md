# Wines Quality Analysis

## Foreground & Goals

This Jupyter Notebook contains analysis on the red variants of the Portuguese "Vinho Verde" wine. The dataset contains several physicochemical The purpose of the aforementioned analysis is two-fold:

Goals:
1. Figure out what physicochemical variables exceptional variants (with a quality > 6) of the Vinho Verde wine have as opposed to non-exceptional variants?
2. Create a statistical model that would be able to explain how the physicochemical variables affect the quality of the "Vinho Verde" wine to a high degree.

## Table of Contents

1. Data Input and Preparation
2. Data Analysis & Structural Analysis
2. Modelling
3. Conclusions

## Variables Used

### Physicochemical features:

- `Fixed Acidity` - which represents acids that do not evaporate rapidly and are a biproduct of the fermentation process.
- `Volatile Acidity` - which represents the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
- `Citric Acid` - an acid supplement during the fermentation process to help winemakers boost the acidity of their wine especially grapes grown in warmer climates (found in small quantities, citric acid can add 'freshness' and flavor to wines).
- `Residual Sugar` - the amount of sugar remaining after fermentation stops.
- `Chlorides` - the amount of salt in the wine.
- `Free Sulfur Dioxide` - amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine.
- `Density` - the density of water is close to that of water depending on the percent alcohol and sugar content.
- `pH` - describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic).
- `Sulphates` - a wine additive which can contribute to sulfur dioxide gas (S02) levels.

### Target Variable: Quality

The sensory quality of the wine.

## Conclusions Made

- Exceptional Quality wines seem to have higher `fixed_acidity`, significantly lower `volatile_acidity`, higher `citric_acid`, higher `sulphate` concentration & more `alcohol`.
- Exceptional Wines seem to also have less variance in each of the physicochemical variable distributions.
- For the most part these are the relevant effect of each independent variable on the quality of the wine:
    - `Volatile acidity` negatively impacts the quality of the wine.
    - The amount of `sulphates` in the wine have a higher likelihood of pushing the wine to exceptional range or pull it out of the worst wine quality class - it's a bit of a wildcard feature.
    - `pH` only has stronger impact for already exceptional wines. The effect is positive.
    - `Total sulfur dioxide` has a negative impact towards the quality of the wine. Probably because it creates a harsh, bitter taste.
    - `Citric acid` has a positive effect on the quality of the wine that is stronger for the better wines.
    - `Residual sugar` has a positive effect on the quality of the wine. The measure itself refers to the amount of sugar left after fermentation, which migh mean that less sweet wines have higher chances of being rated better quality.
    - `Alcohol` - the higher the alcohol level, the better the quality is. And the effect seems to be stronger for better quality wines. 
- For target variables that have discrete values of up <10 distinct values, classification models perform much better in accuracy than regression models with polynomials and interaction terms included.

## Links

Link to the Kaggle database I got my inspiration from: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

My Personal GitHub: https://github.com/kaspa-r
My Personal LinkedIn: https://www.linkedin.com/in/kasparas-rutkauskas/