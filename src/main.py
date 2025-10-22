import pandas as pd
import numpy as np
import os
from data_loader import load_and_merge_data
from feature_engineering import create_features
from labeling import create_target_label
from model_trainer import train_and_evaluate, plot_feature_importance
from explainer import explain_model_globally, explain_single_prediction

def main():
    """
    데이터 로딩부터 모델 학습, 해석까지 전체 파이프라인을 실행하는 메인 함수
    """
    try:
        # 실제 데이터셋이 저장된 경로를 지정해야 합니다.
        DATA_DIR = '../BigContest2025-main/data'
        PATH_INFO = os.path.join(DATA_DIR, 'big_data_set1_f.csv')
        PATH_SALES = os.path.join(DATA_DIR, 'big_data_set2_f.csv')
        PATH_CUSTOMER = os.path.join(DATA_DIR, 'big_data_set3_f.csv')
        PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'labeled_data.csv')

        print("\n데이터 전처리 및 라벨링을 새로 시작합니다.")
        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
        featured_df = create_features(abt)
        labeled_df = create_target_label(featured_df)
        final_df = labeled_df.copy()
        final_df.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"전처리 및 라벨링 완료된 데이터를 '{PROCESSED_DATA_PATH}'에 저장했습니다.")
        
        final_df.sort_values(by=['ENCODED_MCT', 'TA_YM'], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        TARGET = 'is_at_risk'
        features_to_exclude = [
            TARGET, 'ENCODED_MCT', 'MCT_BSE_AR', 'MCT_NM', 'MCT_BRD_NUM', 'MCT_SIGUNGU_NM', 
            'HPSN_MCT_ZCD_NM', 'HPSN_MCT_BZN_CD_NM', 'ARE_D', 'MCT_ME_D', 'TA_YM', 
            'MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT', 
            'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT', 'APV_CE_RAT',

            'business_age',
            'M12_MAL_1020_RAT', 'M12_MAL_30_RAT', 'M12_MAL_40_RAT', 'M12_MAL_50_RAT', 'M12_MAL_60_RAT',
            'M12_FME_1020_RAT', 'M12_FME_30_RAT', 'M12_FME_40_RAT', 'M12_FME_50_RAT', 'M12_FME_60_RAT'
        ]
        features = [col for col in final_df.columns if col not in features_to_exclude]
        
        X = final_df[features]
        y = final_df[TARGET]
        print(f"최종 데이터 크기: {X.shape}")
        print(f"사용할 피처 개수: {len(features)}개")  

        model, scores = train_and_evaluate(X, y)

        print(f"평균 AUPRC: {np.mean(scores['auprc']):.4f} ± {np.std(scores['auprc']):.4f}")
        print(f"평균 F1 Score: {np.mean(scores['f1']):.4f} ± {np.std(scores['f1']):.4f}")
        print(f"평균 RECALL: {np.mean(scores['recall']):.4f} ± {np.std(scores['recall']):.4f}")
        print(f"평균 PRECISION: {np.mean(scores['precision']):.4f} ± {np.std(scores['precision']):.4f}")

        print("\n모델 해석을 시작합니다.")
        plot_feature_importance(model, features) # LightGBM 기본 피처 중요도
        explainer, _ = explain_model_globally(model, X, features) # SHAP 전역 설명

        # 국소적 설명: 테스트셋에서 가장 위험하다고 예측된 가맹점은 '왜' 위험한가?
        most_recent_month = final_df['TA_YM'].max()
        recent_df = final_df[final_df['TA_YM'] == most_recent_month].copy()
        
        X_recent = recent_df[features]
        recent_pred_proba = model.predict_proba(X_recent)[:, 1]
        
        highest_risk_idx = np.argmax(recent_pred_proba)
        risky_sample = recent_df.iloc[[highest_risk_idx]]
        risky_X = X_recent.iloc[[highest_risk_idx]]
        
        print(f"가장 최근 월({most_recent_month.strftime('%Y-%m')}) 데이터 중,")
        print(f"가장 위험도가 높은 가맹점(ID: {risky_sample['ENCODED_MCT'].values[0]})을 분석합니다.")
        print(f"예측된 폐업 위험 확률: {recent_pred_proba[highest_risk_idx]:.2%}")

        # 해당 샘플에 대한 SHAP 분석을 수행합니다.
        explain_single_prediction(model, risky_X, features)

    except FileNotFoundError as e:
        print(f"오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    except Exception as e:
        print(f"파이프라인 실행 중 오류가 발생했습니다: {e}")


if __name__ == '__main__':
    main()

