import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def set_korean_font():
    """Matplotlib에서 한글 폰트를 설정합니다."""
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("Malgun Gothic font not found. Please install it for Korean characters.")

def explain_model_globally(model, X, feature_names, sample_size=1000):
    """
    SHAP을 사용하여 모델의 전역적 특성 중요도를 시각화합니다.
    대용량 데이터의 경우 샘플링하여 연산 속도를 높입니다.

    Args:
        model: 학습된 LightGBM 모델
        X (pd.DataFrame): 설명에 사용할 데이터
        feature_names (list): 피처 이름 목록
        sample_size (int): SHAP 계산에 사용할 데이터 샘플 크기
    """
    print("\n모델 전역 설명 생성 중 (SHAP)...")
    set_korean_font()
    
    # 데이터가 클 경우 샘플링하여 사용
    if len(X) > sample_size:
        print(f"데이터가 크므로 {sample_size}개의 샘플로 SHAP 값을 계산합니다.")
        X_sample = shap.sample(X, sample_size, random_state=42)
    else:
        X_sample = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # SHAP 요약 플롯 (Beeswarm): 각 피처가 예측에 미치는 영향의 분포를 보여줍니다.
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='dot', show=False)
    plt.title('SHAP 요약 플롯 (전역 설명)')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    plt.close()
    print("SHAP Summary Plot 저장 완료: shap_summary_plot.png")

    # SHAP 중요도 플롯 (Bar): 피처의 평균적인 영향력 크기를 보여줍니다.
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP 피처 중요도 (전역)')
    plt.tight_layout()
    plt.savefig('shap_importance_bar_plot.png')
    plt.close()
    print("SHAP Importance Bar Plot 저장 완료: shap_importance_bar_plot.png")
    
    return explainer, shap_values

def explain_single_prediction(model, X_instance, feature_names):
    """
    단일 예측에 대한 국소적 설명을 SHAP Waterfall 플롯으로 시각화합니다.
    '왜' 특정 가맹점이 위험하다고 예측되었는지 설명합니다.

    Args:
        explainer: SHAP TreeExplainer 객체
        X_instance (pd.Series): 설명할 단일 데이터 샘플 (가맹점의 특정 월 데이터)
    """
    print(f"\n단일 예측에 대한 국소 설명 생성 중...")
    set_korean_font()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)

    ev = explainer.expected_value
    expected_value_for_plot = ev
    if isinstance(ev, (list, tuple, np.ndarray)) and len(ev) > 1:
        expected_value_for_plot = ev[1]
    elif isinstance(ev, (list, tuple, np.ndarray)) and len(ev) == 1:
        expected_value_for_plot = ev[0] 
    
    shap_values_2d_array = None
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values_2d_array = shap_values[1]
        elif len(shap_values) == 1:
            shap_values_2d_array = shap_values[0]
        else:
            raise ValueError("SHAP이 빈 리스트를 반환했습니다.")
    else:
        shap_values_2d_array = shap_values

    shap_values_sample = shap_values_2d_array[0]
    
    plt.figure()
    # SHAP Waterfall 플롯: 기본 확률에서 시작하여 각 피처가 확률을 어떻게 올리고 내렸는지 보여줍니다.
    shap.waterfall_plot(shap.Explanation(
        values=shap_values_sample,
        base_values=expected_value_for_plot,
        data=X_instance.values[0],
        feature_names=feature_names),
        show=False)
    plt.title("단일 예측 분석 (무엇이 위험도를 높였는가?)")
    plt.tight_layout()
    instance_index = X_instance.index[0]
    plt.savefig(f'shap_waterfall_plot_idx_{instance_index}.png')
    plt.close()
    print(f"SHAP Waterfall Plot 저장 완료: shap_waterfall_plot_idx_{instance_index}.png")


if __name__ == '__main__':
    try:
        from data_loader import load_and_merge_data
        from feature_engineering import create_features
        from labeling import create_target_label
        from model_trainer import train_and_evaluate
        
        # --- 전체 파이프라인 실행 ---
        PATH_INFO = '../BigContest2025-main/data/big_data_set1_f.csv'
        PATH_SALES = '../BigContest2025-main/data/big_data_set2_f.csv'
        PATH_CUSTOMER = '../BigContest2025-main/data/big_data_set3_f.csv'
        
        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
        featured_df = create_features(abt)
        labeled_df = create_target_label(featured_df)

        final_df = labeled_df.dropna(subset=['is_at_risk'])

        TARGET = 'is_at_risk'
        features_to_exclude = [
            TARGET, 'ENCODED_MCT', 'TA_YM', 'OPEN_DT', 'CLOSE_DT',
            'SIGUNGU_CD', 'IND_CD'
        ]   
        features = [col for col in final_df.columns if col not in features_to_exclude]
        
        X = final_df[features]
        y = final_df[TARGET]

        if y.nunique() > 1:
            final_model, _ = train_and_evaluate(X, y)
            
            # --- 모델 설명 로직 ---
            # 1. 전역 설명: 모델 전체적으로 어떤 피처가 중요한가?
            explainer, _ = explain_model_globally(final_model, X, features)

            # 2. 국소 설명: 가장 위험하다고 예측된 샘플은 '왜' 위험한가?
            y_pred_proba = final_model.predict_proba(X)[:, 1]
            # 위험 확률이 가장 높은 샘플의 인덱스를 찾습니다.
            high_risk_idx = np.argmax(y_pred_proba)
            instance_to_explain = X.iloc[high_risk_idx]
            
            explain_single_prediction(explainer, instance_to_explain)
        else:
            print("타겟 변수에 클래스가 하나뿐이어서 모델 설명 로직을 실행할 수 없습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
