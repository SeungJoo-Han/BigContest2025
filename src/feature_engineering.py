import pandas as pd
import numpy as np
import re
import os

def extract_score_from_category(text):
    """'3_25-50%'와 같은 문자열에서 맨 앞 숫자만 추출합니다."""
    if not isinstance(text, str):
        return 0
    match = re.match(r'^\d+', text)
    return int(match.group(0)) if match else 0

def create_features(df):
    """
    파생 변수(피처)들을 생성합니다.

    Args:
        df (pd.DataFrame): 병합된 원본 데이터프레임

    Returns:
        pd.DataFrame: 피처가 추가된 데이터프레임
    """

    print("피처 엔지니어링 시작...")
    df = df.copy()
    
    # 1. 날짜 관련 피처 처리
    # 'TA_YM'은 월별 데이터, 'OPEN_DT'는 개별 날짜 데이터입니다.
    df['TA_YM'] = pd.to_datetime(df['TA_YM'], format='%Y%m')
    df['ARE_D'] = pd.to_datetime(df['ARE_D'], format='%Y%m%d', errors='coerce')
    df['MCT_ME_D'] = pd.to_datetime(df['MCT_ME_D'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(by=['ENCODED_MCT', 'TA_YM']).reset_index(drop=True)

    # 2. 업력(Business Age) 계산
    df['business_age'] = (df['TA_YM'].dt.year - df['ARE_D'].dt.year) * 12 + (df['TA_YM'].dt.month - df['ARE_D'].dt.month)

    # 3. 구간 변수 수치화
    score_cols_map = {
        'MCT_OPE_MS_CN': 'MCT_OPE_MS_CN_score', # 가맹점 운영개월수 구간
        'RC_M1_SAA': 'RC_M1_SAA_score', # 1개월 매출액 구간
        'RC_M1_TO_UE_CT': 'RC_M1_TO_UE_CT_score', # 1개월 매출건수 구간
        'RC_M1_UE_CUS_CN': 'RC_M1_UE_CUS_CN_score', # 1개월 유니크 고객수 구간
        'RC_M1_AV_NP_AT': 'RC_M1_AV_NP_AT_score', # 1개월 객단가 구간
        'APV_CE_RAT': 'APV_CE_RAT_score' # 취소율 구간
    }
    for original_col, new_col in score_cols_map.items():
        if original_col in df.columns:
            df[new_col] = df[original_col].apply(extract_score_from_category)

    '''
    # 4. 동적 위기 지표 (모멘텀 및 변동성)
    # 데이터셋 구성에 따라 실제 분석할 컬럼명을 명시합니다.
    key_metrics = list(score_cols_map.values()) + ['DLV_SAA_RAT']
    periods = [1, 3] # 성장률 및 시차를 계산할 기간 (월)
    windows = [3, 6] # 이동 통계를 계산할 기간 (월)

    for col in key_metrics:
        if col not in df.columns:
            continue # 데이터에 해당 컬럼이 없으면 건너뛰기
            
        # 성장률 (변화율)
        for p in periods:
            df[f'{col}_growth_{p}m'] = df.groupby('ENCODED_MCT')[col].pct_change(periods=p)
        
        # 이동 통계 (평균, 표준편차) - 현재를 제외한 과거 데이터로 계산하기 위해 shift(1) 사용
        for w in windows:
            rolling_window = df.groupby('ENCODED_MCT')[col].shift(1).rolling(window=w)
            df[f'{col}_rolling_mean_{w}m'] = rolling_window.mean().reset_index(0,drop=True)
            df[f'{col}_rolling_std_{w}m'] = rolling_window.std().reset_index(0,drop=True)
            

    
    # 6. 동종 그룹 벤치마킹 (상대적 성과)
    # 먼저 각 가맹점의 1개월 매출 성장률을 계산합니다.
    score_col = 'RC_M1_SAA_score'
    if score_col in df.columns:
        # 지역(시군구) 내 동종 그룹의 평균 매출 성장률 대비
        df['sigungu_avg_score'] = df.groupby(['MCT_SIGUNGU_NM', 'TA_YM'])[score_col].transform('mean')
        df['peer_score_perf_sigungu'] = df[score_col] - df['sigungu_avg_score']

        # 업종 내 동종 그룹의 평균 매출 성장률 대비
        df['industry_avg_score'] = df.groupby(['HPSN_MCT_ZCD_NM', 'TA_YM'])[score_col].transform('mean')
        df['peer_score_perf_industry'] = df[score_col] - df['industry_avg_score']
    '''
    
    # 5. 고객 행동 변화 지표
    customer_mix_cols = [
        'MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT',
        'RC_M1_SHC_RSD_UE_CLN_RAT', 'RC_M1_SHC_WP_UE_CLN_RAT', 'RC_M1_SHC_FLP_UE_CLN_RAT'
    ]

    for col in customer_mix_cols:
        if col not in df.columns: continue
        # 각 구성 비율의 n개월 전 대비 변화량
        for n in [3, 6, 12]:
            df[f'{col}_diff_{n}m'] = df.groupby('ENCODED_MCT')[col].diff(periods=n)
        # 각 구성 비율의 최근 3개월간 변동성 (표준편차)
        # df[f'{col}_rolling_std_3m'] = df.groupby('ENCODED_MCT')[col].shift(1).rolling(3).std().reset_index(0,drop=True)

    # 7. 무한대 값 처리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"피처 엔지니어링 완료. 총 {df.shape[1]}개의 피처 생성.")
    return df

if __name__ == '__main__':
    # 실제 환경에 맞게 파일 경로를 수정해야 합니다.
    try:
        from data_loader import load_and_merge_data

        PATH_INFO = "../BigContest2025-main/data/big_data_set1_f.csv"
        PATH_SALES = "../BigContest2025-main/data/big_data_set2_f.csv"
        PATH_CUSTOMER = "../BigContest2025-main/data/big_data_set3_f.csv"

        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
        featured_df = create_features(abt)
        
        print("\n피처 생성 후 데이터프레임 정보:")
        featured_df.info()
        
        print("\n생성된 피처 샘플 (성장률):")
        print(featured_df.filter(like='_growth_').head())

        # 생성된 데이터프레임을 CSV 파일로 저장
        output_dir = '../BigContest2025-main/data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'featured_data.csv')
        featured_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    except ImportError:
        print("오류: 'data_loader.py' 파일을 찾을 수 없습니다.")

