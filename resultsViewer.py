import pandas as pd
from scipy.stats import spearmanr

def get_metrics(projDF, actualDF, topN):
    projDF = projDF.head(topN)[["player_name"]].copy()
    actualDF = actualDF[["player_name"]].copy()

    projDF["modelRank"] = projDF.index + 1
    actualDF["actualRank"] = actualDF.index + 1

    players = projDF.merge(actualDF, on="player_name", how="inner")
    players["rankDiff"] = players["modelRank"] - players["actualRank"]
    players["absDiff"] = players["rankDiff"].abs()
    big_misses = players.sort_values(by='absDiff', ascending=False).head(5)
    big_misses.reset_index(drop=True, inplace=True)

    result = spearmanr(players["modelRank"], players["actualRank"])

    return players["absDiff"].mean(), result.statistic, big_misses[['player_name', 'modelRank', 'actualRank', 'rankDiff']]

positions = [["qb", 20], ["rb", 50], ["wr", 60], ["te", 20]]

for pos_name, limit in positions:
    projDF = pd.read_csv(f'{pos_name}_predictions.csv')
    espnDF = pd.read_csv(f'espn_{pos_name}_predictions.csv')
    actualDF = pd.read_csv(f'espn_{pos_name}_final.csv')

    if pos_name == "te":
        projDF.replace("Kyle Pitts", "Kyle Pitts Sr.", inplace=True)

    modelMAE, modelRho, modelMiss = get_metrics(projDF, actualDF, limit)
    espnMAE, espnRho, espnMiss = get_metrics(espnDF, actualDF, limit)

    print(f"--- {pos_name.upper()} ---")
    print(f"MAE Improvement (Higher is Better): {espnMAE - modelMAE:.2f}")
    print(f"Spearman (Higher is Better): Model: {modelRho:.3f} | ESPN: {espnRho:.3f}")
    print(f"Spearman Improvement (Positive is Better): {modelRho - espnRho:.3f}\n")
    print(f"Top 5 'Misses' for {pos_name} (MODEL):")
    print(modelMiss)
    print(f"Top 5 'Misses' for {pos_name} (ESPN):")
    print(espnMiss)
    print()
