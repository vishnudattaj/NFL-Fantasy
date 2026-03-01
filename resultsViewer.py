import pandas as pd

def rankDiff(projDF, actualDF):
    projDF = projDF[["player_name"]].copy()
    actualDF = actualDF[["player_name"]].copy()

    projDF["modelRank"] = projDF.index + 1
    actualDF["actualRank"] = actualDF.index + 1

    players = projDF.merge(actualDF, on="player_name", how="inner")
    players["rankDiff"] = players["modelRank"] - players["actualRank"]
    players["absDiff"] = players["rankDiff"].abs()

    return players["absDiff"].mean()

positions = ["qb", "rb", "wr", "te"]

for pos in positions:
    projFile = f'{pos}_predictions.csv'
    espnFile = f'espn_{pos}_predictions.csv'
    actualFile = f'espn_{pos}_final.csv'

    projDF = pd.read_csv(projFile)
    espnDF = pd.read_csv(espnFile)
    actualDF = pd.read_csv(actualFile)

    modelDiff = rankDiff(projDF, actualDF)
    espnDiff = rankDiff(espnDF, actualDF)

    print(f"{pos.upper()}: ESPN - Model RankDiff = {espnDiff - modelDiff:.2f}")
