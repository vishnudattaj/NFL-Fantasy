# Fantasy Football Ranking Generator  

A machine learning–powered tool for generating, comparing, and evaluating fantasy football player rankings.  
This project combines player historical stats, ESPN projections, and custom ML models to identify undervalued and overvalued players for the upcoming season.  

---

## How It Works  

The project is composed of three main scripts:  

### `rankingGenerator.py`  
- Trains **XGBoost regression models** (wrapped in `MultiOutputRegressor`) on past player stats.  
- Generates predictions for **QB, RB, WR, TE** for the 2024 season.  
- Outputs position-specific predictions (`qb_predictions.csv`, `rb_predictions.csv`, etc.) ranked by fantasy points.  

### `adpGenerator.py`  
- Uses the `espn_api` package to fetch ESPN’s projected points for free agents.  
- Exports ESPN-based positional rankings (`espn_qb_predictions.csv`, `espn_rb_predictions.csv`, etc.).  

### `playerValuer.py`  
- Compares the **model-generated rankings** with **ESPN’s rankings**.  
- Identifies **undervalued** and **overvalued** players based on rank differences.  
- Outputs results into a single Excel file (`ranking_comparison.xlsx`) with separate sheets per position.  

---

## Outputs  

- `*_predictions.csv` → Model-based predictions for each position.  
- `espn_*_predictions.csv` → ESPN projections for each position.  
- `ranking_comparison.xlsx` → Combined comparison, split into undervalued/overvalued players.  

---

## Tech Stack  

- **Python**  
- **Pandas** – data wrangling  
- **XGBoost** – regression modeling  
- **scikit-learn** – training & evaluation utilities  
- **espn_api** – fetch ESPN projections  

---

## Features  

- Machine learning–based player stat predictio
- ESPN projections scraping & integration
- Identification of undervalued and overvalued players 
- Export to Excel for easy draft prep

