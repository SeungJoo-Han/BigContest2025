import pandas as pd
import numpy as np

def create_target_label(df, look_ahead_start=3, look_ahead_end=6):
    """
    복합적인 위기 정의에 따라 타겟 변수(is_at_risk)를 생성합니다.
    위기 발생 3~6개월 전 시점을 위험(1)으로 레이블링합니다.
    (기존 코드의 오류를 모두 수정한 최종 버전)

    Args:
        df (pd.DataFrame): 피처가 생성된 데이터프레임
        look_ahead_start (int): 예측 시작 시점 (개월)
        look_ahead_end (int): 예측 종료 시점 (개월)

    Returns:
        pd.DataFrame: 'is_at_risk' 컬럼이 추가된 데이터프레임
    """
    print("타겟 레이블링 시작...")
    # 날짜 타입이 변환되지 않았다면 변환
    if not pd.api.types.is_datetime64_any_dtype(df['TA_YM']):
        df['TA_YM'] = pd.to_datetime(df['TA_YM'], format='%Y%m')
    if not pd.api.types.is_datetime64_any_dtype(df['CLOSE_DT']):
        df['CLOSE_DT'] = pd.to_datetime(df['CLOSE_DT'], format='%Y%m%d', errors='coerce')

    df = df.sort_values(by=['ENCODED_MCT', 'TA_YM']).reset_index(drop=True)
    
    # 성과 하락을 판단할 핵심 지표 설정
    METRIC_COL = 'SLS_AMT'

    # --- 위기 정의 1: 폐업 (CLOSE_DT가 해당 월에 속하는 경우) ---
    df['close_month'] = df['CLOSE_DT'].dt.to_period('M')
    df['current_month'] = df['TA_YM'].dt.to_period('M')
    df['crisis_event_closure'] = (df['close_month'] == df['current_month'])

    # --- 위기 정의 2: 임계 성과 하락 ---
    # 12개월 이동 평균 및 표준편차 계산
    rolling_12m = df.groupby('ENCODED_MCT')[METRIC_COL].rolling(window=12, min_periods=6)
    rolling_mean = rolling_12m.mean().reset_index(0, drop=True)
    rolling_std = rolling_12m.std().reset_index(0, drop=True)
    
    # 임계값 정의: 평균 - 2 * 표준편차
    threshold = rolling_mean - 2 * rolling_std
    
    # 임계값 이하로 하락했는지 여부
    df['perf_decline'] = df[METRIC_COL] < threshold
    
    # 2개월 연속 하락 여부 확인 (shift(1)은 바로 전 달을 의미)
    df['crisis_event_perf'] = df['perf_decline'] & df.groupby('ENCODED_MCT')['perf_decline'].shift(1).fillna(False)

    # --- 최종 위기 이벤트 정의 (폐업 또는 성과 하락) ---
    df['crisis_event'] = df['crisis_event_closure'] | df['crisis_event_perf']

    # --- 레이블링 ---
    # 각 가맹점별 '첫' 위기 발생 시점 찾기
    crisis_dates = df[df['crisis_event']].groupby('ENCODED_MCT')['TA_YM'].min().reset_index()
    crisis_dates.rename(columns={'TA_YM': 'crisis_month'}, inplace=True)

    # 원본 데이터에 위기 발생 월 정보 병합
    df = pd.merge(df, crisis_dates[['ENCODED_MCT', 'crisis_month']], on='ENCODED_MCT', how='left')

    # is_at_risk 레이블 생성 (기본값은 0)
    df['is_at_risk'] = 0
    
    # crisis_month가 있는 경우 (위기 가맹점)에만 레이블링 수행
    mask_crisis_merchant = df['crisis_month'].notna()
    
    # 위기 발생 월과 현재 월의 차이 계산 (개월 단위)
    month_diff = (df.loc[mask_crisis_merchant, 'crisis_month'].dt.year - df.loc[mask_crisis_merchant, 'TA_YM'].dt.year) * 12 + \
                 (df.loc[mask_crisis_merchant, 'crisis_month'].dt.month - df.loc[mask_crisis_merchant, 'TA_YM'].dt.month)

    # 예측 기간(3~6개월 전)에 해당하는 데이터에 1로 레이블링
    label_mask = (month_diff >= look_ahead_start) & (month_diff <= look_ahead_end)
    df.loc[mask_crisis_merchant & label_mask, 'is_at_risk'] = 1
    
    print("타겟 레이블링 완료.")
    if 1 in df['is_at_risk'].value_counts(normalize=True):
        print(f"위험 클래스(1) 비율: {df['is_at_risk'].value_counts(normalize=True)[1]:.4f}")
    else:
        print("위험 클래스(1)가 데이터에 존재하지 않습니다. 위기 정의나 기간 설정을 확인해보세요.")
        
    # 중간 계산에 사용된 컬럼들 제거
    df = df.drop(columns=[
        'close_month', 'current_month', 'crisis_event_closure', 'perf_decline', 
        'crisis_event_perf', 'crisis_event', 'crisis_month'
    ])
    
    return df

if __name__ == '__main__':
    try:
        from data_loader import load_and_merge_data
        from feature_engineering import create_features
        
        PATH_INFO = '../data/dataset1_info.csv'
        PATH_SALES = '../data/dataset2_sales.csv'
        PATH_CUSTOMER = '../data/dataset3_customer.csv'
        
        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
        featured_df = create_features(abt)
        labeled_df = create_target_label(featured_df)

        print("\n레이블링 결과 샘플 (is_at_risk == 1):")
        # is_at_risk가 1인 샘플이 있는 경우에만 출력
        if not labeled_df[labeled_df['is_at_risk'] == 1].empty:
            print(labeled_df[labeled_df['is_at_risk'] == 1][['ENCODED_MCT', 'TA_YM', 'SLS_AMT', 'is_at_risk']].head())
        else:
            print("위험으로 레이블링된 샘플이 없습니다.")

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    except (ImportError, ModuleNotFoundError):
        print("오류: 'data_loader.py' 또는 'feature_engineering.py' 파일을 찾을 수 없습니다.")
