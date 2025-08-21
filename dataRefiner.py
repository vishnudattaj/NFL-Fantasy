import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def pastFeatures(featureList, excludeList, dataFrame, draft):
    dataFrame = pd.merge(dataFrame.drop(columns=["draft_year"]), draft, on='player_id', how='left')
    dataFrame['experience'] = dataFrame['season'] - dataFrame['draft_year']

    for feature in featureList:
        if feature not in excludeList:
            dataFrame[f"Past-1-{feature}"] = dataFrame.groupby('player_id')[feature].shift(1)

    dataFrame.dropna(axis=1, how='all', inplace=True)
    dataFrame.fillna(0, inplace=True)

    return dataFrame[(dataFrame["season"] > 2012)].reset_index(drop=True)


def currentDataExtractor(dataFrame, excludeList, model, trainColumns, playerType):
    featureList = dataFrame.columns.tolist()
    dataFrame = dataFrame[(dataFrame["season"] == 2024)]

    for feature in featureList:
        if feature not in excludeList:
            if "Past" not in feature:
                dataFrame[f"Past-1-{feature}"] = dataFrame[feature]

    X = dataFrame[[col for col in featureList if "Past" in col]]

    y_pred = model.predict(X)

    pred_df = pd.DataFrame(
        y_pred,
        columns=trainColumns
    )

    pred_df["player_id"] = dataFrame["player_id"].values
    pred_df["player_name"] = dataFrame["player_name"].values
    if playerType == "QB":
        pred_df["fantasy_pts"] = (pred_df["passing_yards"] / 25) + (pred_df["pass_touchdown"] * 4) + (pred_df["rushing_yards"] / 10) + (pred_df["rush_touchdown"] * 6) - (pred_df["interception"] * 2) - (pred_df["fumble"] * 2)
    else:
        pred_df["fantasy_pts"] = (pred_df["rushing_yards"] / 10) + (pred_df["rush_touchdown"] * 6) + (pred_df["receiving_yards"] / 10) + (pred_df["receiving_touchdown"] * 6) + (pred_df["receptions"] * 1) - (pred_df["fumble"] * 2)
    pred_df = pred_df.sort_values(by="fantasy_pts", ascending=False).reset_index(drop=True)
    pred_df = pred_df[["player_id", "player_name", "fantasy_pts"] + trainColumns]
    return pred_df


def trainModel(trainingDF, trainColumns):
    X = trainingDF[[col for col in trainingDF.columns.tolist() if "Past" in col]]
    y = trainingDF[trainColumns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(XGBRegressor(learning_rate=0.1, max_depth=3))

    return model.fit(X_train, y_train)


yearlyPlayer = pd.read_csv('yearly_player_stats_offense.csv')
yearlyPlayer = yearlyPlayer[(yearlyPlayer['season_type'] == "REG")]
yearlyPlayer.drop(["season_type", "games_played_season", "games_played_career"], axis=1, inplace=True)

yearlyFeatures = yearlyPlayer.columns.tolist()

excludeFeatures = ["player_id", "player_name", "position", "birth_year", "draft_round", "draft_pick", "draft_ovr", "height", "weight", "college", "season", "team", "conference", "division"]
averages = ["passer_rating", "adot", "comp_pct", "int_pct", "pass_td_pct", "ypa", "rec_td_pct", "yptarget", "ayptarget", "ypr", "rush_td_pct", "ypc", "td_pct", "yptouch", "age", "years_exp", "team_pass_attempts_share", "team_complete_pass_share", "team_passing_yards_share", "team_pass_touchdown_share", "team_targets_share", "team_receptions_share", "team_receiving_yards_share", "team_receiving_air_yards_share", "team_receiving_touchdown_share", "team_rush_attempts_share", "team_rushing_yards_share", "team_rush_touchdown_share"]
draft_year_df = yearlyPlayer[['player_id', 'draft_year']].drop_duplicates()

yearlyPlayer = pastFeatures(yearlyFeatures, excludeFeatures, yearlyPlayer, draft_year_df)

yearlyQB = yearlyPlayer[(yearlyPlayer['position'] == "QB")].reset_index(drop=True)
yearlyRB = yearlyPlayer[(yearlyPlayer['position'] == "RB")].reset_index(drop=True)
yearlyWR = yearlyPlayer[(yearlyPlayer['position'] == "WR")].reset_index(drop=True)
yearlyTE = yearlyPlayer[(yearlyPlayer['position'] == "TE")].reset_index(drop=True)

yearlyQB_training = yearlyQB[(yearlyQB['experience'] > 0)]
yearlyRB_training = yearlyRB[(yearlyRB['experience'] > 0)]
yearlyWR_training = yearlyWR[(yearlyWR['experience'] > 0)]
yearlyTE_training = yearlyTE[(yearlyTE['experience'] > 0)]

qbDF = currentDataExtractor(yearlyQB, excludeFeatures, trainModel(yearlyQB_training, ["passing_yards", "rushing_yards", "rush_touchdown", "pass_touchdown", "interception", "fumble"]), ["passing_yards", "rushing_yards", "rush_touchdown", "pass_touchdown", "interception", "fumble"], "QB")
qbDF.drop(columns=["player_id"], inplace=True)
qbDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("qb_predictions.csv")

rbDF = currentDataExtractor(yearlyRB, excludeFeatures, trainModel(yearlyRB_training, ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"]), ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "RB")
rbDF.drop(columns=["player_id"], inplace=True)
rbDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("rb_predictions.csv")

wrDF = currentDataExtractor(yearlyWR, excludeFeatures, trainModel(yearlyWR_training, ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"]), ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "WR")
wrDF.drop(columns=["player_id"], inplace=True)
wrDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("wr_predictions.csv")

teDF = currentDataExtractor(yearlyTE, excludeFeatures, trainModel(yearlyTE_training, ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"]), ["receiving_yards", "rushing_yards", "rush_touchdown", "receptions", "receiving_touchdown", "fumble"], "TE")
teDF.drop(columns=["player_id"], inplace=True)
teDF.sort_values(by='fantasy_pts', ascending=False).reset_index(drop=True).to_csv("te_predictions.csv")
