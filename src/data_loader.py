import pandas as pd
import numpy as np

def load_and_merge_data(path_info, path_sales, path_customer):
    """
    3개의 데이터셋을 로드하고 병합하여 기본 분석 테이블(ABT)을 생성합니다.

    Args:
        path_info (str): 가맹점 개요 정보 파일 경로
        path_sales (str): 월별 매출 정보 파일 경로
        path_customer (str): 월별 이용 고객 정보 파일 경로

    Returns:
        pd.DataFrame: 병합된 데이터프레임
    """
    print("데이터 로딩 시작...")
    # 데이터 로드
    info_df = pd.read_csv(path_info)
    sales_df = pd.read_csv(path_sales)
    customer_df = pd.read_csv(path_customer)
    print("데이터 로딩 완료.")

    # 결측치를 의미하는 특이값(-999999.9)을 NaN으로 변환
    sales_df.replace(-999999.9, np.nan, inplace=True)
    customer_df.replace(-999999.9, np.nan, inplace=True)

    print("데이터 병합 시작...")
    # 매출 정보와 고객 정보를 'ENCODED_MCT', 'TA_YM' 기준으로 병합
    merged_df = pd.merge(sales_df, customer_df, on=['ENCODED_MCT', 'TA_YM'], how='left')

    # 위 결과에 가맹점 개요 정보를 'ENCODED_MCT' 기준으로 병합
    final_df = pd.merge(merged_df, info_df, on='ENCODED_MCT', how='left')
    print("데이터 병합 완료.")

    return final_df

if __name__ == '__main__':
    # 예시 실행 코드
    # 실제 파일 경로로 수정해야 합니다.
    PATH_INFO = '../data/dataset1_info.csv'
    PATH_SALES = '../data/dataset2_sales.csv'
    PATH_CUSTOMER = '../data/dataset3_customer.csv'

    abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
    print("\n병합된 데이터프레임 정보:")
    print(abt.info())
    print("\n병합된 데이터프레임 샘플:")
    print(abt.head())
