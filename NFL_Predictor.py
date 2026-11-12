from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy
import pandas


#constants
url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
back_games = 5


def load(url): #to clean data

    data = pandas.read_csv(url)

    change = pandas.to_datetime(data["gameday"])
    data["gameday"] = change

    team_name_change = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

    change2 = data["home_team"].replace(team_name_change)
    data["home_team"] = change2

    change3 = data["away_team"].replace(team_name_change)
    data["away_team"] = change3

    change4 = data["season"]>=2002
    data = data[change4]

    data = data.sort_values(by=["gameday", "game_id"])
    data = data.reset_index(drop=True)

    change5 = data["result"].notna()
    data["played"] = change5

    return data

def elo_change(data):
    elo={}
    prev_season = None
    home_elo=[]
    away_elo=[]

    for i in data.itertuples():
        if i.season!=prev_season:
            for team in elo:
                new_rating = (elo[team]*(1-1/3)+1500*1/3)
                elo[team]=new_rating
            prev_season = i.season

        elo_h = elo.get(i.home_team,1500)
        elo_a = elo.get(i.away_team, 1500)

        home_elo.append(elo_h)
        away_elo.append(elo_a)

        if not i.played:
            continue
#including home feild advantage in the elo calculation
        if getattr(i,"location","Home")=="Neutral":
            advantage = 0
        else:
            advantage = 50

        expected = 1/(1+10**(-(elo_h+advantage-elo_a)/400))

        if i.result>0:
            actual = 1
        elif i.result<0:
            actual = 0
        else:
            actual = 0.5

        if i.result>0:
            elo_differnece = elo_h + advantage -elo_a
        else:
            elo_differnece = elo_a - advantage - elo_h
#use a modified elo system to take into account the win difference
        temp = numpy.log(abs(i.result)+1)*2.2
        margin_of_victory = temp/(elo_differnece*0.001+2.2)

        final_change = 20*margin_of_victory*(actual-expected)
        elo[i.home_team] = elo_h +final_change
        elo[i.away_team] = elo_a - final_change

    data["home_elo"] = home_elo
    data["away_elo"] = away_elo

    return data

#adjusts based on recent games
def recent(data):
#create a copy of data to modify without affecting dataset
    home_data = data[["game_id","gameday","home_team","home_score","away_score"]].copy()
    home_data.columns = ["game_id","gameday","team","scored","conced"]
    home_data["is_home"]=1
    away_data = data[["game_id", "gameday", "away_team", "away_score", "home_score"]].copy()
    away_data.columns = ["game_id", "gameday", "team", "scored", "conced"]
    away_data["is_home"]=0

    combine = pandas.concat([home_data, away_data], ignore_index=True)
    combine = combine.sort_values(["team","gameday","game_id"])

    seperate_teams = combine.groupby("team")

    def calc_rolling_avg(scores):
        prev_scores = scores.shift(1)

        rolling_avg = prev_scores.rolling(5,min_periods=1).mean()

        return rolling_avg

    combine["scored_rolling"]= seperate_teams["scored"].transform(calc_rolling_avg)
    combine["conced_rolling"]= seperate_teams["conced"].transform(calc_rolling_avg)

    home_games = combine[combine["is_home"]==1]
    home_recent = home_games[["game_id","scored_rolling","conced_rolling"]].copy()
    home_recent.columns=["game_id","home_scored","home_conced"]

    away_games = combine[combine["is_home"]==0]
    away_recent = away_games[["game_id","scored_rolling","conced_rolling"]].copy()
    away_recent.columns =["game_id","away_scored","away_conced"]

    data = data.merge(home_recent,on="game_id",how="left")
    data = data.merge(away_recent,on="game_id",how="left")

    return data

def filter_traning_data(data):
    data1 = data.copy()

    elo_difference = data1["home_elo"]-data1["away_elo"]
    data1["elo_diff"] = elo_difference

    total_points_difference = (data1["home_scored"]-data1["home_conced"])-((data1["away_scored"]-data1["away_conced"]))
    data1["total_diff"] = total_points_difference

    rest_days_difference = (data1["home_rest"]-data1["away_rest"]).clip(-7,7)
    # use clip so unusaly long rests dont make to much of a differnece
    data1["rest_days_difference"] = rest_days_difference

    data1["divisonal_game"] = data1["div_game"].fillna(0)
    #not entirely usefull might remove

    netural_check = []
    for location in data1["location"]:
        if location == "Neutral":
            netural_check.append(1)
        else:
            netural_check.append(0)
    data1["netural"]=netural_check #converts true/false into 1,0

    playoff_check = []
    for playoff in data1["game_type"]:
        if playoff != "REG":
            playoff_check.append(1)
        else:
            playoff_check.append(0)
    data1["playoff"] = playoff_check

    traning_data = ["elo_diff","total_diff","home_scored","home_conced","away_scored","away_conced",
                    "rest_days_difference","divisonal_game","netural","playoff"]

    data1[traning_data] = data1[traning_data].fillna(data1[traning_data].median())
    #just to make it so the machine learning always has values even if not exactly right

    return data1, traning_data

def evaluate_model(model_name,actual,prob):

    prediction = []
    for value in prob:
        if value>0.5:
            prediction.append(True)
        else:
            prediction.append(False)

    accuracy = accuracy_score(actual,prediction)
    loss = log_loss(actual,prob)
    brier = brier_score_loss(actual,prob)

    print(model_name + "-- accuracy: "+ str(round(accuracy,4))+" log loss: "+str(round(loss,4))+" Brier: "+str(round(brier,4)))

def main():
    data = load(url)
    data = elo_change(data)
    data = recent(data)
    data, traning_data = filter_traning_data(data)

    home_win = []
    for game in data["result"]:
        if game > 0:
            home_win.append(1)
        else:
            home_win.append(0)
    data["home_win"] = home_win

    played_games = data[data["played"]&(data["result"]!=0)].copy()




    season_count = played_games.groupby("season").size()
    full_seasons = season_count[season_count>=260]
    recent_full_season = full_seasons.index.max()
    partition = recent_full_season - back_games + 1

    # split into traning and testing data

    traning = played_games[played_games["season"]<partition]
    testing = played_games[(played_games["season"]>=partition)&(played_games["season"]<=recent_full_season)]


    print("traning on " +str(len(traning)) +" games from "+ str(traning["season"].min()) + " - "+ str(traning["season"].max()))
    print("testing on " + str(len(testing)) + " games from " + str(testing["season"].min()) + " - " + str(
        testing["season"].max()))

    #do the machine learning part

    models = {}
    logistic_regression = make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=1000))
    models["Logistic_regression"] = logistic_regression
    gradient_boosting = HistGradientBoostingClassifier(max_depth=3,learning_rate=0.03,max_iter=250,l2_regularization=5.0,random_state=0)
    models["Gradient_boosting"] = gradient_boosting



    elo_probability = 1/(1+10**(-(testing["elo_diff"]+50)/400))

    evaluate_model("elo only",testing["home_win"],elo_probability)

    scores = {}
    for i in models:
        model = models[i]
        model.fit(traning[traning_data],traning["home_win"]) #traning data and target data

        all_prob = model.predict_proba(testing[traning_data])
        probabilities = []
        for game in all_prob:
            probabilities.append(game[1])
        scores[i] = log_loss(testing["home_win"],probabilities)
        evaluate_model(i,testing["home_win"],probabilities)

    top_model = min(scores,key=scores.get)

    chosen_model = models[top_model]
    chosen_model.fit(played_games[traning_data],played_games["home_win"])

    print("Chosen model:" + top_model)

    upcomming_games = data[~data["played"]].copy()
    next = upcomming_games.sort_values("gameday").iloc[0] #selects first row

    week_games = upcomming_games[(upcomming_games["season"]==next["season"])&(upcomming_games["week"]==next["week"])].copy()

    all_prob2 = chosen_model.predict_proba(week_games[traning_data])
    home_prob = []
    for i in all_prob2:
        home_prob.append(i[1])
    week_games["p_home"] = home_prob

    for i,game in week_games.sort_values("gameday").iterrows():

        if game["p_home"] >= 0.5:
            pick = game["home_team"]
            prob = game["p_home"]
        else:
            pick = game["away_team"]
            prob = 1-game["p_home"]

        percent_prob = round(prob * 100,1)
        print(
            str(game["away_team"]) +" @ "+
            str(game["home_team"]) +"   " +
            str(pick)+"   " +
            str(percent_prob)+"%"
        )

if __name__ == "__main__":
    main()
