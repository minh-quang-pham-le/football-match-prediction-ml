import pandas as pd
import numpy as np

def compute_all_features(df: pd.DataFrame, team_attr: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:

    # A) Cơ bản
    df['match_year']  = df['date'].dt.year
    df['match_month'] = df['date'].dt.month

    # B) Kết quả (Target)
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
        h2h[['match_api_id','h2h_wins_last3','h2h_nobs','h2h_win_rate_last3']],
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

    # G) Thông tin cơ bản của player (avg height/weight/age)
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

    # H) Các features thêm
    
    # Tính match_importance
    df['match_importance'] = df['stage'] / df.groupby(['league_id','season'])['stage'].transform('max')
    
    # Chia 3 pha (early / mid / late) theo mùa
    df['pct_stage'] = df.groupby(['league_id','season'])['stage'].rank(pct=True)
    df['match_phase'] = (
        pd.qcut(
            df['pct_stage'],
            q=[0, 0.33, 0.66, 1.0],
            labels=[1, 2, 3],
            duplicates='drop'
        ).astype(int)
    )
    df = df.drop(columns=['pct_stage']) 

    return df

def get_player_stats(matches: pd.DataFrame, player_attr: pd.DataFrame) -> pd.DataFrame:
    # 1) Prepare player_stats: rename date and sort
    ps = (
        player_attr
        .rename(columns={'date':'attr_date'})
        .sort_values(['player_api_id','attr_date'])
    )
    ps['player_api_id'] = ps['player_api_id'].astype(int)

    attrs = [
        'overall_rating','potential','acceleration','sprint_speed',
        'finishing','crossing','heading_accuracy','dribbling','stamina','strength',
        'interceptions','standing_tackle','sliding_tackle',
        'long_passing','short_passing','ball_control',
        'long_shots','shot_power','curve','free_kick_accuracy','volleys',
        'penalties'
    ]

    # 2) Unpivot matches into long form
    recs = []
    for side in ('home','away'):
        for i in range(1,12):
            col = f'{side}_player_{i}'
            if col not in matches:
                continue
            tmp = (
                matches[['match_api_id','date', col]]
                .dropna(subset=[col])
                .rename(columns={'date':'date', col:'player_api_id'})
            )
            tmp['player_api_id'] = tmp['player_api_id'].astype(int)
            tmp['side']         = side
            tmp['position']     = i
            tmp['is_match']     = True
            recs.append(tmp)
    match_long = pd.concat(recs, ignore_index=True)

    # 3) Prepare attribute events
    attr_long = (
        ps[['player_api_id','attr_date'] + attrs]
        .rename(columns={'attr_date':'date'})
    )
    attr_long['side']         = np.nan
    attr_long['position']     = np.nan
    attr_long['match_api_id'] = np.nan
    attr_long['is_match']     = False

    # 4) Concatenate and forward‐fill per player
    all_long = pd.concat([attr_long, match_long], sort=False)
    all_long = all_long.sort_values(['player_api_id','date','is_match'])
    all_long[attrs] = all_long.groupby('player_api_id')[attrs].ffill()

    # 5) Filter back to match rows
    merged = all_long[all_long['is_match']].drop(columns=['is_match'])

    # 6) Compute average per match & side
    avg_stats = (
        merged
        .groupby(['match_api_id','side'])[attrs]
        .mean()
        .unstack('side')  # creates MultiIndex columns (attr, side)
    )

    # Rename the MultiIndex columns via the (attr,side)→flat_name map
    new_cols = {
        (attr, side): f'{side}_avg_{attr}'
        for attr in attrs
        for side in ('home','away')
    }
    avg_stats = avg_stats.rename(columns=new_cols)

    # Flatten any remaining MultiIndex entries to a single-level list of strings
    avg_stats.columns = [
        col if isinstance(col, str) else new_cols.get(col)
        for col in avg_stats.columns
    ]

    # Now bring match_api_id back as a column
    avg_stats = avg_stats.reset_index()
    return avg_stats

def add_composite_features(fifa_stats: pd.DataFrame) -> pd.DataFrame:
    stats = fifa_stats.copy()
    for side in ('home','away'):
        # pace = avg(acceleration, sprint_speed)
        stats[f'{side}_pace'] = stats[[f'{side}_avg_acceleration', f'{side}_avg_sprint_speed']].mean(axis=1)
        # tackle = avg(standing_tackle, sliding_tackle)
        stats[f'{side}_tackle'] = stats[[f'{side}_avg_standing_tackle', f'{side}_avg_sliding_tackle']].mean(axis=1)
        # passing_skill = avg(short_passing, long_passing, ball_control)
        stats[f'{side}_passing_skill'] = stats[
            [f'{side}_avg_short_passing', f'{side}_avg_long_passing', f'{side}_avg_ball_control']
        ].mean(axis=1)
        # dribbling_skill = avg(dribbling, ball_control)
        stats[f'{side}_dribbling_skill'] = stats[
            [f'{side}_avg_dribbling', f'{side}_avg_ball_control']
        ].mean(axis=1)
        # shooting_skill = avg(long_shots, shot_power, curve, free_kick_accuracy, volleys)
        stats[f'{side}_shooting_skill'] = stats[
            [f'{side}_avg_long_shots', f'{side}_avg_shot_power', f'{side}_avg_curve',
             f'{side}_avg_free_kick_accuracy', f'{side}_avg_volleys']
        ].mean(axis=1)
        # physical = avg(strength, stamina)
        stats[f'{side}_physical'] = stats[
            [f'{side}_avg_strength', f'{side}_avg_stamina']
        ].mean(axis=1)
        # defensive_skill = avg(interceptions, tackle)
        stats[f'{side}_defensive_skill'] = stats[
            [f'{side}_avg_interceptions', f'{side}_tackle']
        ].mean(axis=1)
        # direct avg features
        stats[f'{side}_finishing']          = stats[f'{side}_avg_finishing']
        stats[f'{side}_crossing']           = stats[f'{side}_avg_crossing']
        stats[f'{side}_heading_accuracy']   = stats[f'{side}_avg_heading_accuracy']
        stats[f'{side}_penalties']          = stats[f'{side}_avg_penalties']

    return stats

def merge_player_features(matches: pd.DataFrame,
                          fifa_stats: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()

    # 1) Rename avg overall & potential if present
    for side in ('home', 'away'):
        avg_over = f'{side}_avg_overall_rating'
        avg_pot  = f'{side}_avg_potential'
        if avg_over in fifa_stats.columns:
            fifa_stats = fifa_stats.rename(columns={avg_over: f'{side}_overall_rating'})
        if avg_pot in fifa_stats.columns:
            fifa_stats = fifa_stats.rename(columns={avg_pot: f'{side}_potential'})

    # 2) Define final feature names
    feat_cols = [
        'overall_rating', 'potential', 'pace', 'passing_skill', 'dribbling_skill',
        'shooting_skill', 'finishing', 'physical', 'defensive_skill',
        'crossing', 'heading_accuracy', 'penalties'
    ]

    # 3) Build list of columns to merge
    merge_cols = ['match_api_id']
    for side in ('home', 'away'):
        for feat in feat_cols:
            col = f'{side}_{feat}'
            if col in fifa_stats.columns:
                merge_cols.append(col)

    # 4) Subset and merge
    to_merge = fifa_stats[merge_cols]
    df = df.merge(to_merge, on='match_api_id', how='left')
    return df

if __name__ == "__main__":
    # 1) Load data
    matches = pd.read_csv('data/processed/df_1.csv', parse_dates=['date'])
    players = pd.read_csv('data/raw/players.csv', parse_dates=['birthday'])
    team_attr = pd.read_csv('data/raw/team_attributes.csv', parse_dates=['date'])

    # 2) Compute all features
    df = compute_all_features(matches, team_attr, players)

    # 3) Get player stats
    player_stats = get_player_stats(matches, team_attr)

    # 4) Add composite features
    player_stats = add_composite_features(player_stats)

    # 5) Merge player features into main dataframe
    df = merge_player_features(df, player_stats)

    # 6) Save the final dataframe
    df.to_csv('data/processed/df_2.csv', index=False)