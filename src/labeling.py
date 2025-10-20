import pandas as pd
import numpy as np
import os

def create_target_label(df, look_ahead_start=1, look_ahead_end=12):
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
    df['is_at_risk'] = 0

    # 폐업일(MCT_ME_D)이 존재하는 모든 데이터에 대해 직접 레이블링을 시도합니다.
    # 1. 폐업일이 기록된 (NaT가 아닌) 행들만 대상으로 마스크를 생성합니다.
    mask_has_closure_date = df['MCT_ME_D'].notna()

    # 2. 각 행의 기준년월(TA_YM)이 폐업일(MCT_ME_D)의 범위에 속하는지 확인합니다.
    start_date_check = df['TA_YM'] >= (df['MCT_ME_D'] - pd.DateOffset(months=look_ahead_end))
    end_date_check = df['TA_YM'] < (df['MCT_ME_D'] - pd.DateOffset(months=look_ahead_start - 1))
    
    # 3. 폐업일이 존재하고, 날짜 범위가 맞는 모든 행을 최종 대상(final_mask)으로 선정합니다.
    final_mask = mask_has_closure_date & start_date_check & end_date_check

    # 4. 최종 대상의 'is_at_risk'를 1로 설정합니다.
    df.loc[final_mask, 'is_at_risk'] = 1
    
    print("타겟 레이블링 완료.")
    if 1 in df['is_at_risk'].value_counts(normalize=True):
        print(f"위험 클래스(1) 비율: {df['is_at_risk'].value_counts(normalize=True)[1]:.4f}")
    else:
        print("위험 클래스(1)가 데이터에 존재하지 않습니다. 위기 정의나 기간 설정을 확인해보세요.")

    return df

if __name__ == '__main__':
    try:
        from data_loader import load_and_merge_data
        from feature_engineering import create_features
        
        PATH_INFO = '../BigContest2025-main/data/big_data_set1_f.csv'
        PATH_SALES = '../BigContest2025-main/data/big_data_set2_f.csv'
        PATH_CUSTOMER = '../BigContest2025-main/data/big_data_set3_f.csv'

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
