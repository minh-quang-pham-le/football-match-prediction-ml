import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def compute_all_features(df, team_attr, players):

    # A) Cơ bản
    df['match_year']  = df['date'].dt.year
    df['match_month'] = df['date'].dt.month

    # B) Kết quả & thông số bàn thắng
    df['goal_diff']   = df['home_team_goal'] - df['away_team_goal']
    df['total_goals'] = df['home_team_goal'] + df['away_team_goal']
    df['outcome'] = df.apply(
        lambda row: 'Win' if row['home_team_goal'] > row['away_team_goal']
        else 'Draw' if row['home_team_goal'] == row['away_team_goal']
        else 'Loss',
        axis=1
    )
    
    # C) Phong độ & động lực (5 trận gần nhất có trọng số)
    weights = np.array([1, 2, 3, 4, 5])
    
    home = (
        df[['match_api_id','date','home_team_api_id','home_team_goal','away_team_goal']]
        .rename(columns={
            'home_team_api_id':'team_id',
            'home_team_goal':'goals_for',
            'away_team_goal':'goals_against'
        })
    )
    away = (
        df[['match_api_id','date','away_team_api_id','away_team_goal','home_team_goal']]
        .rename(columns={
            'away_team_api_id':'team_id',
            'away_team_goal':'goals_for',
            'home_team_goal':'goals_against'
        })
    )
    home['is_win'] = (home['goals_for'] > home['goals_against']).astype(int)
    away['is_win'] = (away['goals_for'] > away['goals_against']).astype(int)

    all_matches = pd.concat([home, away]).sort_values('date')
    
    def weighted_rolling(s: pd.Series) -> pd.Series:
        return (
            s.shift(1)
            .rolling(window=len(weights), min_periods=len(weights))
            .apply(lambda arr: np.dot(arr, weights) / weights.sum(), raw=True)
        )
        
    # Tính các feature weighted form và goals
    all_matches['home_wins_last5']    = all_matches.groupby('team_id')['is_win'].transform(weighted_rolling)
    all_matches['home_avg_gs_last5']  = all_matches.groupby('team_id')['goals_for'].transform(weighted_rolling)
    all_matches['home_avg_gc_last5']  = all_matches.groupby('team_id')['goals_against'].transform(weighted_rolling)

    recent = all_matches[['match_api_id','team_id','home_wins_last5','home_avg_gs_last5','home_avg_gc_last5']]
 
    # Merge trở lại df
    df = df.merge(
        recent.rename(columns={'team_id':'home_team_api_id'}),
        on=['match_api_id','home_team_api_id'],
        how='left'
    ).merge(
        recent.rename(columns={
            'team_id':'away_team_api_id',
            'home_wins_last5':'away_wins_last5',
            'home_avg_gs_last5':'away_avg_gs_last5',
            'home_avg_gc_last5':'away_avg_gc_last5'
        }),
        on=['match_api_id','away_team_api_id'],
        how='left'
    )
    
    # Drop 5 trận đầu tiên (không đủ 5 trận lịch sử) cho cả home và away
    df = df.dropna(subset=[
        'home_wins_last5','home_avg_gs_last5','home_avg_gc_last5',
        'away_wins_last5','away_avg_gs_last5','away_avg_gc_last5'
    ]).reset_index(drop=True)

    # D) Head-to-head (3 trận gần nhất với Laplace Smoothing)
    alpha = 2
    p0 = 0.5

    # 1. Tạo key và bảng tạm
    df['h2h_key'] = df.apply(lambda x: tuple(sorted([x['home_team_api_id'], x['away_team_api_id']])), axis=1)
    h2h = df[['match_api_id','date','h2h_key','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']].copy()
    h2h['home_win'] = (h2h['home_team_goal'] > h2h['away_team_goal']).astype(int)
    h2h = h2h.sort_values('date')

    # 2. Tính số trận thắng và số trận đối đầu trong 3 lần trước
    h2h['h2h_wins_last3'] = h2h.groupby('h2h_key')['home_win'] \
        .transform(lambda s: s.shift(1).fillna(0).rolling(3, min_periods=1).sum())
    h2h['h2h_nobs'] = h2h.groupby('h2h_key')['home_win'] \
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).count())

    # 3. Laplace smoothing
    h2h['h2h_win_rate_last3'] = (h2h['h2h_wins_last3'] + alpha * p0) / (h2h['h2h_nobs'] + alpha)

    # 4. Merge vào df
    df = df.merge(  
        h2h[['match_api_id','h2h_wins_last3','h2h_nobs','h2h_win_rate']],
        on='match_api_id', how='left'
    )


    # E) Tactical features (median mùa giải trước)
    attrs = [
        'buildUpPlaySpeed','buildUpPlayPassing',
        'chanceCreationPassing','chanceCreationShooting',
        'defencePressure','defenceAggression','defenceTeamWidth'
    ]
    
    # 1) Hàm lấy prev_season từ "YYYY/YYYY+1"
    def prev_season_label(season_str: str) -> str:
        y0, y1 = map(int, season_str.split('/'))
        return f"{y0-1}/{y1-1}"

    df['prev_season'] = df['season'].apply(prev_season_label)

    # 2) Tạo season label đúng kiểu châu Âu cho team_attr
    def make_season_label(d: pd.Timestamp) -> str:
        start = d.year if d.month >= 7 else d.year-1
        return f"{start}/{start+1}"

    team_attr['season'] = team_attr['date'].apply(make_season_label)

    # 3) Lấy bản ghi cuối mỗi (team_api_id, season)
    team_attr_last = (
    team_attr
    .sort_values(['team_api_id', 'season', 'date'])
    .drop_duplicates(subset=['team_api_id','season'], keep='last')
    .loc[:, ['team_api_id','season'] + attrs]
    )
    
    # 4) Merge home / away
    for side in ('home','away'):
        df = df.merge(
            team_attr_last.rename(columns={
                'team_api_id':           f'{side}_team_api_id',
                'season':                'prev_season',
                **{col: f'{side}_{col}' for col in attrs}
            }),
            on=[f'{side}_team_api_id','prev_season'],
            how='left'
        )

    # F) Tactical diffs
    df['diff_speed']    = df['home_buildUpPlaySpeed']    - df['away_buildUpPlaySpeed']
    df['diff_shooting'] = df['home_chanceCreationShooting'] - df['away_chanceCreationShooting']
    df['diff_pressure'] = df['home_defencePressure']     - df['away_defencePressure']

    # G) Player static info (avg height/weight/age)
    recs = []
    for side in ['home','away']:
        for i in range(1,12):
            recs.append(
                df[['match_api_id','date',f'{side}_player_{i}']]
                .rename(columns={f'{side}_player_{i}':'player_api_id'})
                .assign(side=side)
            )
    long = pd.concat(recs, ignore_index=True)
    long = long.merge(
        players[['player_api_id','birthday','height','weight']],
        on='player_api_id', how='left'
    )
    long['age'] = (pd.to_datetime(long['date']) - pd.to_datetime(long['birthday'])).dt.days // 365
    agg = (
        long
        .groupby(['match_api_id','side'])
        .agg(
            avg_height=('height','mean'),
            avg_weight=('weight','mean'),
            avg_age=('age','mean')
        )
        .reset_index()
    )
    pivot = agg.pivot(index='match_api_id', columns='side')
    pivot.columns = [f"{side}_{metric}" for metric, side in pivot.columns]
    pivot = pivot.reset_index()
    df = df.merge(pivot, on='match_api_id', how='left')

    # H) Additional context
    df['home_advantage'] = 1
    df['rest_days'] = df.groupby('home_team_api_id')['date'].diff().dt.days.fillna(0)
    df['max_stage'] = df.groupby(['league_id','season'])['stage'].transform('max')
    df['match_importance'] = df['stage'] / df['max_stage']
    imp_norm = df['match_importance']
    df['match_imp_cat'] = pd.qcut(
        imp_norm,
        q=[0, 0.33, 0.66, 1.0],
        labels=[1, 2, 3]
    ).astype(int)

    df.drop(columns=['max_stage'], inplace=True)

    return df

def add_player_skill_features_asof(df, player_attr):
    
    player_attr = (player_attr
                   .rename(columns={'date': 'attr_date'})
                   .sort_values(['player_api_id', 'attr_date']))
    
    attrs = [
        'overall_rating','potential','acceleration','sprint_speed',
        'finishing','crossing','heading_accuracy','dribbling','stamina','strength',
        'interceptions','standing_tackle','sliding_tackle',
        'long_passing','short_passing','ball_control',
        'long_shots','shot_power','curve','free_kick_accuracy','volleys',
        'penalties'
    ]

    parts = []
    for side in ('home', 'away'):
        for i in range(1, 12):
            parts.append(
                df[['match_api_id', 'date', f'{side}_player_{i}']]
                  .rename(columns={'date': 'match_date',
                                   f'{side}_player_{i}': 'player_api_id'})
                  .assign(side=side, position=i)
            )
    long = (pd.concat(parts, ignore_index=True)
              .sort_values(['player_api_id', 'match_date']))

    # Merge_asof to pick the nearest attr_date <= match_date
    merged = pd.merge_asof(
        long,
        player_attr[['player_api_id', 'attr_date'] + attrs],
        left_on='match_date',
        right_on='attr_date',
        by='player_api_id',
        direction='backward'
    )

    # Composite feature calculations
    merged['pace']              = merged[['acceleration','sprint_speed']].mean(axis=1)
    merged['tackle']            = merged[['standing_tackle','sliding_tackle']].mean(axis=1)
    merged['passing_skill']     = merged[['short_passing','long_passing', 'ball_control']].mean(axis=1)
    merged['dribbling_skill']   = merged[['dribbling','ball_control']].mean(axis=1)
    merged['shooting_skill']    = merged[['long_shots','shot_power','curve','free_kick_accuracy','volleys']].mean(axis=1)
    merged['physical']          = merged[['strength','stamina']].mean(axis=1)
    merged['defensive_skill']   = merged[['interceptions','tackle']].mean(axis=1)

    # Select final feature columns to aggregate
    feat_cols = [
        'overall_rating','potential','pace','passing_skill','dribbling_skill',
        'shooting_skill','finishing','physical','defensive_skill',
        'crossing','heading_accuracy','penalties'
    ]

    # Aggregate to match-level per side
    agg = (
        merged
        .groupby(['match_api_id','side'])[feat_cols]
        .mean()
        .reset_index()
    )

    # Pivot so we have home_avg_* and away_avg_* columns
    pivot = agg.pivot(index='match_api_id', columns='side')
    pivot.columns = [f"{side}_avg_{col}" for col, side in pivot.columns]
    pivot = pivot.reset_index()

    # Merge back to original df
    df = df.merge(pivot, on='match_api_id', how='left')
    return df