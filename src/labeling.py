import pandas as pd
import numpy as np
import os

def create_target_label(df, look_ahead_start=1, look_ahead_end=3):
    """
    복합적인 위기 정의에 따라 타겟 변수(is_at_risk)를 생성합니다.
    위기 발생 1~3개월 전 시점을 위험(1)으로 레이블링합니다.

    Args:
        df (pd.DataFrame): 피처가 생성된 데이터프레임
        look_ahead_start (int): 예측 시작 시점 (개월)
        look_ahead_end (int): 예측 종료 시점 (개월)

    Returns:
        pd.DataFrame: 'is_at_risk' 컬럼이 추가된 데이터프레임
    """
    print("타겟 레이블링 시작...")
    df = df.sort_values(by=['ENCODED_MCT', 'TA_YM']).reset_index(drop=True)
    METRIC_COL = 'RC_M1_SAA_score'  # 성과 지표 컬럼명

    # --- 위기 정의 1: 폐업 (CLOSE_DT가 해당 월에 속하는 경우) ---
    df['crisis_event_closure'] = False
    # 폐업일(MCT_ME_D)이 있는 가맹점에 대해서만 계산
    closed_merchants = df[df['MCT_ME_D'].notna()]
    # 폐업월(YYYY-MM)과 기준년월(YYYY-MM)이 정확히 일치하는 시점을 True로 설정
    df.loc[closed_merchants.index, 'crisis_event_closure'] = \
        (closed_merchants['MCT_ME_D'].dt.strftime('%Y-%m') == closed_merchants['TA_YM'].dt.strftime('%Y-%m'))


    # --- 위기 정의 2: 임계 성과 하락 ---
    # 3개월 이동 평균 및 표준편차 계산
    rolling_3m = df.groupby('ENCODED_MCT')[METRIC_COL].rolling(window=3, min_periods=1)
    rolling_mean = rolling_3m.mean().reset_index(0, drop=True)
    rolling_std = rolling_3m.std().reset_index(0, drop=True)

    # 임계값 정의: 평균 - 1.5 * 표준편차
    threshold = rolling_mean - 1.5 * rolling_std
    
    # 임계값 이하로 하락했는지 여부
    df['perf_decline'] = df[METRIC_COL] < threshold
    
    # 2개월 연속 하락 여부 확인 (shift(1)은 바로 전 달을 의미)
    df['crisis_event_perf'] = df['perf_decline'] & df.groupby('ENCODED_MCT')['perf_decline'].shift(1).fillna(False)

    # --- 최종 위기 이벤트 정의 (폐업 또는 성과 하락) ---
    df['crisis_event'] = df['crisis_event_closure'] | df['crisis_event_perf']


    # --- 조기 레이블링 ---
    # 각 가맹점별 '첫' 위기 발생 시점 찾기
    crisis_dates = df[df['crisis_event']].groupby('ENCODED_MCT')['TA_YM'].min().reset_index()
    crisis_dates.rename(columns={'TA_YM': 'crisis_month'}, inplace=True)

    # 원본 데이터에 위기 발생 월 정보 병합
    df = pd.merge(df, crisis_dates, on='ENCODED_MCT', how='left')

    # is_at_risk 레이블 생성 (기본값은 0)
    df['is_at_risk'] = 0
    
    # crisis_month가 있는 경우 (위기 가맹점)에만 레이블링 수행
    mask_crisis_merchant = df['crisis_month'].notna()
    
    month_diff = (df.loc[mask_crisis_merchant, 'crisis_month'].dt.year - df.loc[mask_crisis_merchant, 'TA_YM'].dt.year) * 12 + \
                 (df.loc[mask_crisis_merchant, 'crisis_month'].dt.month - df.loc[mask_crisis_merchant, 'TA_YM'].dt.month)
    label_mask = (month_diff >= look_ahead_start) & (month_diff <= look_ahead_end)
    df.loc[mask_crisis_merchant & label_mask, 'is_at_risk'] = 1
    
    print("타겟 레이블링 완료.")
    if 1 in df['is_at_risk'].value_counts(normalize=True):
        print(f"위험 클래스(1) 비율: {df['is_at_risk'].value_counts(normalize=True)[1]:.4f}")
    else:
        print("위험 클래스(1)가 데이터에 존재하지 않습니다. 위기 정의나 기간 설정을 확인해보세요.")
        
    # 중간 계산에 사용된 컬럼들 제거
    df = df.drop(columns=[
        'crisis_event_closure', 'perf_decline', 
        'crisis_event_perf', 'crisis_event', 'crisis_month'
    ], errors='ignore')

    return df

if __name__ == '__main__':
    try:
        from data_loader import load_and_merge_data
        from feature_engineering import create_features
        
        PATH_INFO = '../data/big_data_set1_f.csv'
        PATH_SALES = '../data/big_data_set2_f.csv'
        PATH_CUSTOMER = '../data/big_data_set3_f.csv'
        
        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
        featured_df = create_features(abt)
        labeled_df = create_target_label(featured_df)

        print("\n레이블링 결과 샘플 (is_at_risk == 1):")
        # is_at_risk가 1인 샘플이 있는 경우에만 출력
        if not labeled_df[labeled_df['is_at_risk'] == 1].empty:
            print(labeled_df[labeled_df['is_at_risk'] == 1][['ENCODED_MCT', 'TA_YM', 'RC_M1_SAA_score', 'is_at_risk']].head())
        else:
            print("위험으로 레이블링된 샘플이 없습니다.")

        # 생성된 데이터프레임을 CSV 파일로 저장
        output_dir = '../BigContest2025-main/data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'labeled_data.csv')
        labeled_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    except (ImportError, ModuleNotFoundError):
        print("오류: 'data_loader.py' 또는 'feature_engineering.py' 파일을 찾을 수 없습니다.")
