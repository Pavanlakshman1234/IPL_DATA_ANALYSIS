# ============================================================
# IPL DATA ANALYSIS - 2026
# Beginner Data Analytics Project
# ============================================================

# Install these once in Terminal / Command Prompt if needed:
# pip install pandas numpy matplotlib

# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 2: Load the dataset
file_path = "IPL_2026_Data_Analysis.csv"
df = pd.read_csv(file_path)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# STEP 3: Understand the dataset
print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all"))

# STEP 4: Clean the data
df = df.drop_duplicates()

# Remove accidental spaces from text columns
text_columns = [
    "Venue", "Team1", "Team2", "Toss_Winner",
    "Toss_Decision", "Winner", "Player_of_Match",
    "Match_Stage"
]

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()

# Convert numeric columns
numeric_columns = [
    "Win_By_Runs", "Win_By_Wickets",
    "Team1_Runs", "Team2_Runs", "Total_Match_Runs"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Missing winner = no-result match
df["Winner"] = df["Winner"].fillna("No Result")

# STEP 5: Basic KPIs
total_matches = df["Match_ID"].nunique()
decided_matches = (df["Match_Result_Type"] == "Decided").sum()
no_result_matches = (df["Match_Result_Type"] == "No Result").sum()
tie_matches = (df["Match_Result_Type"] == "Tie").sum()
total_runs = df["Total_Match_Runs"].sum()
average_match_runs = df["Total_Match_Runs"].mean()
total_venues = df["Venue"].nunique()
total_teams = len(set(df["Team1"]).union(set(df["Team2"])))

print("\n========== KEY PERFORMANCE INDICATORS ==========")
print(f"Total Matches: {total_matches}")
print(f"Decided Matches: {decided_matches}")
print(f"No Result Matches: {no_result_matches}")
print(f"Tied Matches: {tie_matches}")
print(f"Total Runs: {total_runs:,.0f}")
print(f"Average Runs per Match: {average_match_runs:,.2f}")
print(f"Number of Teams: {total_teams}")
print(f"Number of Venues: {total_venues}")

# STEP 6: Team win analysis
team_wins = (
    df[df["Winner"] != "No Result"]
    .groupby("Winner")
    .size()
    .reset_index(name="Wins")
    .rename(columns={"Winner":"Team"})
    .sort_values("Wins", ascending=False)
)

print("\n========== TEAM WINS ==========")
print(team_wins)

# STEP 7: Team match participation
Team1_count = df.groupby("Team1").size().reset_index(name="Matches")
Team1_count.columns = ["Team", "Matches"]

Team2_count = df.groupby("Team2").size().reset_index(name="Matches")
Team2_count.columns = ["Team", "Matches"]

team_matches = (
    pd.concat([Team1_count, Team2_count])
    .groupby("Team")["Matches"]
    .sum()
    .reset_index()
    .sort_values("Matches", ascending=False)
)

print("\n========== TEAM MATCHES ==========")
print(team_matches)

# STEP 8: team win percentage
print("team_matches columns;",team_matches.columns.tolist())
print('team wins columns:', team_wins.columns.tolist())
team_performance = team_matches.merge(
    team_wins,
    on="Team",
    how="left"
)

team_performance["Wins"] = team_performance["Wins"].fillna(0)

team_performance["Win_Percentage"] = (
    team_performance["Wins"] /
    team_performance["Matches"] * 100
)

team_performance = team_performance.sort_values(
    "Win_Percentage", ascending=False
)

print("\n========== TEAM PERFORMANCE ==========")
print(team_performance)

# STEP 9: Toss analysis
toss_wins = (
    df.groupby("Toss_Winner")
      .size()
      .reset_index(name="Toss_Wins")
      .sort_values("Toss_Wins", ascending=False)
)

toss_match_result = df[
    (df["Winner"] != "No Result") &
    (df["Toss_Winner"] == df["Winner"])
]

toss_advantage = len(toss_match_result) / decided_matches * 100

print("\n========== TOSS ANALYSIS ==========")
print(toss_wins)
print(f"Toss winner also won the match in {toss_advantage:.2f}% of decided matches.")

# STEP 10: Toss decision analysis
toss_decision = (
    df.groupby("Toss_Decision")
      .size()
      .reset_index(name="Matches")
      .sort_values("Matches", ascending=False)
)

print("\n========== TOSS DECISION ==========")
print(toss_decision)

# STEP 11: Player of the Match analysis
pom = (
    df[df["Player_of_Match"].notna()]
    .groupby("Player_of_Match")
    .size()
    .reset_index(name="Awards")
    .sort_values("Awards", ascending=False)
)

print("\n========== TOP PLAYER OF THE MATCH ==========")
print(pom.head(10))

# STEP 12: Venue analysis
venue_analysis = (
    df.groupby("Venue")
      .agg(
          Matches=("Match_ID", "nunique"),
          Average_Runs=("Total_Match_Runs", "mean"),
          Total_Runs=("Total_Match_Runs", "sum")
      )
      .reset_index()
      .sort_values("Matches", ascending=False)
)

print("\n========== VENUE ANALYSIS ==========")
print(venue_analysis.head(10))

# STEP 13: Match stage analysis
stage_analysis = (
    df.groupby("Match_Stage")
      .agg(
          Matches=("Match_ID", "nunique"),
          Average_Runs=("Total_Match_Runs", "mean")
      )
      .reset_index()
      .sort_values("Matches", ascending=False)
)

print("\n========== MATCH STAGE ANALYSIS ==========")
print(stage_analysis)

# STEP 14: Winning margin analysis
run_wins = df[df["Win_By_Runs"] > 0]
wicket_wins = df[df["Win_By_Wickets"] > 0]

print("\n========== WINNING MARGIN ==========")
print(f"Matches won by runs: {len(run_wins)}")
print(f"Matches won by wickets: {len(wicket_wins)}")

# STEP 15: Monthly analysis
monthly = (
    df.groupby(["Year", "Month_Number", "Month"])
      .agg(
          Matches=("Match_ID", "nunique"),
          Total_Runs=("Total_Match_Runs", "sum"),
          Average_Runs=("Total_Match_Runs", "mean")
      )
      .reset_index()
      .sort_values(["Year", "Month_Number"])
)

print("\n========== MONTHLY ANALYSIS ==========")
print(monthly)

# STEP 16: Highest-scoring matches
highest_scoring = (
    df[["Match_No", "Team1", "Team2", "Total_Match_Runs"]]
    .sort_values("Total_Match_Runs", ascending=False)
    .head(10)
)

print("\n========== TOP 10 HIGH-SCORING MATCHES ==========")
print(highest_scoring)

# STEP 17: Business / cricket insights
print("\n========== IMPORTANT INSIGHTS ==========")

if not team_wins.empty:
    print(
        "Most match wins:",
        team_wins.iloc[0]["Team"],
        "-",
        team_wins.iloc[0]["Wins"]
    )

if not pom.empty:
    print(
        "Most Player of the Match awards:",
        pom.iloc[0]["Player_of_Match"],
        "-",
        pom.iloc[0]["Awards"]
    )

if not venue_analysis.empty:
    print(
        "Venue with most matches:",
        venue_analysis.iloc[0]["Venue"],
        "-",
        venue_analysis.iloc[0]["Matches"]
    )

if not highest_scoring.empty:
    print(
        "Highest combined score:",
        highest_scoring.iloc[0]["Team1"],
        "vs",
        highest_scoring.iloc[0]["Team2"],
        "-",
        highest_scoring.iloc[0]["Total_Match_Runs"],
        "runs"
    )

# STEP 18: Create charts

# Chart 1 - Team wins
plt.figure(figsize=(10, 6))
plt.bar(team_wins["Team"], team_wins["Wins"])
plt.title("IPL 2026 - Wins by Team")
plt.xlabel("Team")
plt.ylabel("Number of Wins")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Chart 2 - Toss decisions
plt.figure(figsize=(7, 5))
plt.bar(toss_decision["Toss_Decision"], toss_decision["Matches"])
plt.title("IPL 2026 - Toss Decision")
plt.xlabel("Toss Decision")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.show()

# Chart 3 - Matches by venue
top_venues = venue_analysis.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_venues["Venue"], top_venues["Matches"])
plt.title("Top Venues by Number of Matches")
plt.xlabel("Matches")
plt.ylabel("Venue")
plt.tight_layout()
plt.show()

# Chart 4 - Average runs by match stage
plt.figure(figsize=(8, 5))
plt.bar(stage_analysis["Match_Stage"], stage_analysis["Average_Runs"])
plt.title("Average Runs by Match Stage")
plt.xlabel("Match Stage")
plt.ylabel("Average Combined Runs")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Chart 5 - Total runs by match
plt.figure(figsize=(10, 6))
plt.plot(df["Match_No"], df["Total_Match_Runs"], marker="o")
plt.title("Total Runs by Match")
plt.xlabel("Match Number")
plt.ylabel("Combined Runs")
plt.tight_layout()
plt.show()

# STEP 19: Export analysis results for Power BI / Excel
team_wins.to_csv("ipl_team_wins.csv", index=False)
team_performance.to_csv("ipl_team_performance.csv", index=False)
toss_wins.to_csv("ipl_toss_wins.csv", index=False)
toss_decision.to_csv("ipl_toss_decision.csv", index=False)
pom.to_csv("ipl_player_of_match.csv", index=False)
venue_analysis.to_csv("ipl_venue_analysis.csv", index=False)
stage_analysis.to_csv("ipl_stage_analysis.csv", index=False)
monthly.to_csv("ipl_monthly_analysis.csv", index=False)
highest_scoring.to_csv("ipl_highest_scoring_matches.csv", index=False)

print("\n========== PROJECT COMPLETED ==========")
print("All analysis CSV files have been exported.")
