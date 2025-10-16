import pandas as pd
import numpy as np

# 지금까지 작성한 모든 모듈을 임포트합니다.
from data_loader import load_and_merge_data
from feature_engineering import create_features
from labeling import create_target_label
from model_trainer import train_and_evaluate, plot_feature_importance
from explainer import explain_model_globally, explain_single_prediction

def main():
    """
    데이터 로딩부터 모델 학습, 해석까지 전체 파이프라인을 실행하는 메인 함수
    (기존 코드의 미완성 부분을 모두 채운 최종 버전)
    """
    try:
        # --- 1. 데이터 로드 및 병합 ---
        # 실제 데이터셋이 저장된 경로를 지정해야 합니다.
        PATH_INFO = '../data/dataset1_info.csv'
        PATH_SALES = '../data/dataset2_sales.csv'
        PATH_CUSTOMER = '../data/dataset3_customer.csv'
        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)

        # --- 2. 피처 엔지니어링 ---
        featured_df = create_features(abt)

        # --- 3. 타겟 레이블링 ---
        # 위기 발생 3~6개월 전을 '위험'으로 정의합니다.
        labeled_df = create_target_label(featured_df, look_ahead_start=3, look_ahead_end=6)
        
        # --- 4. 모델링을 위한 데이터 준비 ---
        
        # 타겟 변수(is_at_risk)에 결측값이 있는 행은 레이블링이 불가능한 기간이므로 제거합니다.
        final_df = labeled_df.dropna(subset=['is_at_risk'])
        
        # is_at_risk 컬럼을 정수형으로 변환
        final_df['is_at_risk'] = final_df['is_at_risk'].astype(int)
        
            
        # 결측값 처리: 0으로 채우는 대신, 그룹별 Forward Fill 적용
        # 각 가맹점(ENCODED_MCT)별로 시간순(TA_YM)으로 정렬 후 이전 값으로 채웁니다.
        final_df = final_df.sort_values(by=['ENCODED_MCT', 'TA_YM'])
        final_df = final_df.groupby('ENCODED_MCT').ffill()
        
        # Forward Fill 후에도 남은 NaN은 0으로 채웁니다 (주로 각 가맹점의 첫 데이터)
        final_df.fillna(0, inplace=True)
    

        # LightGBM이 직접 처리할 수 있는 범주형 변수를 지정합니다.
        categorical_features = ['SIGUNGU_CD', 'IND_CD'] 
        for col in categorical_features:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype('category')
        
        # 학습 데이터가 너무 적은 경우를 방지
        if final_df.shape[0] < 1000:
            print("경고: 레이블링 후 유효 데이터가 1000개 미만입니다. 파이프라인을 종료합니다.")
            return

        # --- 5. 학습/테스트 데이터 분리 (시계열 기준) ---
        # 데이터를 시간순으로 정렬하여, 과거 데이터로 미래 데이터를 예측하도록 분리합니다.
        final_df = final_df.sort_values(by='TA_YM').reset_index(drop=True)
        
        TARGET = 'is_at_risk'
        X = final_df.drop(columns=[TARGET])
        y = final_df[TARGET]
        
        # 80%는 학습 및 검증, 마지막 20%는 최종 테스트용으로 사용
        train_val_size = int(len(X) * 0.8)
        X_train_val, X_test = X.iloc[:train_val_size], X.iloc[train_val_size:]
        y_train_val, y_test = y.iloc[:train_val_size], y.iloc[train_val_size:]

        # 모델에 사용될 피처만 선택 (ID, 날짜 등 불필요한 컬럼 제외)
        features_to_exclude = [
            'ENCODED_MCT', 'TA_YM', 'OPEN_DT', 'CLOSE_DT'
        ]
        train_features = [col for col in X_train_val.columns if col not in features_to_exclude and X_train_val[col].dtype != 'object']
        
        X_train_val_final = X_train_val[train_features]
        X_test_final = X_test[train_features]

        print(f"학습/검증 데이터 크기: {X_train_val_final.shape}")
        print(f"테스트 데이터 크기: {X_test_final.shape}")

        # --- 6. 모델 학습 및 평가 ---
        # TimeSeriesSplit 교차검증으로 모델의 일반화 성능을 평가합니다.
        model, scores = train_and_evaluate(X_train_val_final, y_train_val)

        # --- 7. 모델 해석 ---
        # 7-1. 전역적 설명: 모델이 어떤 피처를 중요하게 생각하는가?
        print("\n--- 모델 해석 단계 시작 ---")
        plot_feature_importance(model, train_features) # LightGBM 기본 피처 중요도
        explainer, _ = explain_model_globally(model, X_test_final, train_features) # SHAP 전역 설명

        # 7-2. 국소적 설명: 테스트셋에서 가장 위험하다고 예측된 가맹점은 '왜' 위험한가?
        y_pred_proba_test = model.predict_proba(X_test_final)[:, 1]
        
        if len(y_pred_proba_test) > 0:
            high_risk_idx_in_test = y_pred_proba_test.argmax()
            high_risk_sample = X_test_final.iloc[high_risk_idx_in_test]
            
            explain_single_prediction(explainer, high_risk_sample)
        else:
            print("테스트 데이터가 없어 국소적 설명을 생성할 수 없습니다.")

    except FileNotFoundError as e:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    except Exception as e:
        print(f"파이프라인 실행 중 오류가 발생했습니다: {e}")


if __name__ == '__main__':
    main()

