import pandas as pd
import numpy as np

def create_features(df):
    """
    보고서에서 제안된 파생 변수(피처)들을 생성합니다.
    (기존 코드의 오류를 모두 수정한 최종 버전)

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
<<<<<<< HEAD
    df['ARE_D'] = pd.to_datetime(df['ARE_D'], format='%Y%m%d', errors='coerce')
=======
    df['OPEN_DT'] = pd.to_datetime(df['OPEN_DT'], format='%Y%m%d', errors='coerce')
>>>>>>> 4e3b2bbf8d08961cd16e4c299e70e74d052af49b

    # 데이터 정렬 (시계열 분석을 위해 필수)
    df = df.sort_values(by=['ENCODED_MCT', 'TA_YM']).reset_index(drop=True)

    # 2. 업력(Business Age) 계산 (월 단위)
    # (현재 년월 - 개업 년월)
<<<<<<< HEAD
    df['business_age'] = (df['TA_YM'].dt.year - df['ARE_D'].dt.year) * 12 + (df['TA_YM'].dt.month - df['ARE_D'].dt.month)

    # 3. 동적 위기 지표 (모멘텀 및 변동성)
    # 데이터셋 구성에 따라 실제 분석할 컬럼명을 명시합니다.
    key_metrics = ['SLS_AMT', 'CUS_CNT', 'MCT_UE_CLN_REU_RAT'] 
=======
    df['business_age'] = (df['TA_YM'].dt.year - df['OPEN_DT'].dt.year) * 12 + (df['TA_YM'].dt.month - df['OPEN_DT'].dt.month)

    # 3. 동적 위기 지표 (모멘텀 및 변동성)
    # 데이터셋 구성에 따라 실제 분석할 컬럼명을 명시합니다.
    key_metrics = ['SLS_AMT', 'CUS_CNT', 'MCT_UE_CUN_REU_RAT'] 
>>>>>>> 4e3b2bbf8d08961cd16e4c299e70e74d052af49b
    periods = [1, 3, 6] # 성장률 및 시차를 계산할 기간 (월)
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
            
        # 시차(Lag) 피처
        for p in periods:
            df[f'{col}_lag_{p}m'] = df.groupby('ENCODED_MCT')[col].shift(p)

    # 4. 고객 행동 변화
    # 재방문율(loyalty_ratio) 지표: MCT_UE_CUN_REU_RAT 컬럼을 직접 사용
    if 'MCT_UE_CUN_REU_RAT' in df.columns:
        df.rename(columns={'MCT_UE_CUN_REU_RAT': 'loyalty_ratio'}, inplace=True)
        df['loyalty_ratio_change_1m'] = df.groupby('ENCODED_MCT')['loyalty_ratio'].pct_change(periods=1)

    # 고객 1인당 매출액(객단가) 및 성장률 (이 부분은 기존 코드 유지)
    df['sales_per_customer'] = df['SLS_AMT'] / (df['CUS_CNT'] + 1e-6)
    df['sales_per_customer_growth_3m'] = df.groupby('ENCODED_MCT')['sales_per_customer'].pct_change(periods=3)


    # 5. 동종 그룹 벤치마킹 (상대적 성과)
    # 먼저 각 가맹점의 1개월 매출 성장률을 계산합니다.
    sales_growth_col = 'SLS_AMT_growth_1m'
    if sales_growth_col not in df.columns:
         df[sales_growth_col] = df.groupby('ENCODED_MCT')['SLS_AMT'].pct_change(periods=1)

    # 지역(시군구) 내 동종 그룹의 평균 매출 성장률 대비
    df['sigungu_avg_sales_growth_1m'] = df.groupby(['SIGUNGU_CD', 'TA_YM'])[sales_growth_col].transform('mean')
    df['peer_sales_perf_sigungu'] = df[sales_growth_col] - df['sigungu_avg_sales_growth_1m']

    # 업종 내 동종 그룹의 평균 매출 성장률 대비
    df['industry_avg_sales_growth_1m'] = df.groupby(['IND_CD', 'TA_YM'])[sales_growth_col].transform('mean')
    df['peer_sales_perf_industry'] = df[sales_growth_col] - df['industry_avg_sales_growth_1m']

    # 6. 무한대 값(inf) 및 결측값(NaN) 처리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 모델링 단계에서 결측값을 채우거나(fillna), 해당 행을 제거할 수 있습니다.

    print(f"피처 엔지니어링 완료. 총 {df.shape[1]}개의 피처 생성.")
    return df

if __name__ == '__main__':
    # 예시 실행 코드
    # 실제 환경에 맞게 파일 경로를 수정해야 합니다.
    try:
        from data_loader import load_and_merge_data # data_loader.py가 같은 폴더에 있다고 가정
        PATH_INFO = '../data/dataset1_info.csv'
        PATH_SALES = '../data/dataset2_sales.csv'
        PATH_CUSTOMER = '../data/dataset3_customer.csv'
        
        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
        featured_df = create_features(abt)
        
        print("\n피처 생성 후 데이터프레임 정보:")
        featured_df.info()
        
        print("\n생성된 피처 샘플 (성장률):")
        print(featured_df.filter(like='_growth_').head())

        print("\n생성된 피처 샘플 (동종그룹 비교):")
        print(featured_df[['ENCODED_MCT', 'TA_YM', 'peer_sales_perf_sigungu', 'peer_sales_perf_industry']].head())

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    except ImportError:
        print("오류: 'data_loader.py' 파일을 찾을 수 없습니다. 같은 폴더에 위치시켜 주세요.")

