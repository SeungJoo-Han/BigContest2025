import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, top_n=20):
    """
    학습된 모델의 피처 중요도를 시각화합니다.

    Args:
        model: 학습된 LightGBM 모델
        feature_names (list): 피처 이름 목록
        top_n (int): 시각화할 상위 피처 개수
    """
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=feature_names)
    ftr_top = ftr_importances.sort_values(ascending=False)[:top_n]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=ftr_top.values, y=ftr_top.index)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    # 한글 폰트가 깨질 경우를 대비하여 폰트 설정 (환경에 맞는 폰트 경로 필요)
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except:
        print("Malgun Gothic font not found. Please install it for Korean characters.")
    plt.tight_layout()
    plt.show()

def train_and_evaluate(X, y, n_splits=5):
    """
    LightGBM 모델을 학습하고 TimeSeriesSplit을 사용하여 평가합니다.
    (기존 코드의 오류를 모두 수정한 최종 버전)

    Args:
        X (pd.DataFrame): 피처 데이터
        y (pd.Series): 타겟 데이터
        n_splits (int): 교차 검증 폴드 수

    Returns:
        tuple: (학습된 최종 모델, 교차 검증 점수 딕셔너리)
    """
    print("모델 학습 및 평가 시작...")
    
    # 시계열 교차 검증 설정
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 클래스 불균형 처리를 위한 가중치 계산
    if (y == 1).sum() == 0:
        print("경고: 타겟 데이터에 양성 클래스(1)가 없습니다. scale_pos_weight를 1로 설정합니다.")
        scale_pos_weight = 1
    else:
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"클래스 가중치 (scale_pos_weight): {scale_pos_weight:.2f}")

    # LightGBM 모델 파라미터
    params = {
        'objective': 'binary',
        'metric': 'aucpr', # AUPRC (Area Under Precision-Recall Curve)
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos_weight
    }

    # 점수를 저장할 딕셔너리 초기화
    scores = {
        'auprc': [],
        'f1': [],
        'recall': [],
        'precision': []
    }
    
    # 교차 검증 루프
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(**params)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='aucpr',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        scores['auprc'].append(average_precision_score(y_val, y_pred_proba))
        scores['f1'].append(f1_score(y_val, y_pred_binary))
        scores['recall'].append(recall_score(y_val, y_pred_binary))
        scores['precision'].append(precision_score(y_val, y_pred_binary, zero_division=0))
        
        print(f"Fold {fold+1} AUPRC: {scores['auprc'][-1]:.4f}")

    print("\n교차 검증 결과 (평균):")
    for metric, values in scores.items():
        print(f"- {metric.upper()}: {np.mean(values):.4f}")

    # 전체 데이터로 최종 모델 학습
    print("\n전체 데이터로 최종 모델 학습 중...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)
    print("최종 모델 학습 완료.")

    return final_model, scores

if __name__ == '__main__':
    try:
        from data_loader import load_and_merge_data
        from feature_engineering import create_features
        from labeling import create_target_label
        
        PATH_INFO = '../data/dataset1_info.csv'
        PATH_SALES = '../data/dataset2_sales.csv'
        PATH_CUSTOMER = '../data/dataset3_customer.csv'
        
        # 1. 데이터 로드 및 전처리
        abt = load_and_merge_data(PATH_INFO, PATH_SALES, PATH_CUSTOMER)
        featured_df = create_features(abt)
        labeled_df = create_target_label(featured_df)

        # 2. 모델 학습을 위한 데이터 준비
        # 결측값이 있는 행 제거 (모델 학습 전 처리 필요)
        final_df = labeled_df.dropna(subset=['is_at_risk'])
        final_df = final_df.fillna(0) # 간단하게 0으로 채움 (전략 수정 가능)

        # 타겟 변수 및 피처 분리
        TARGET = 'is_at_risk'
        # 모델이 학습할 수 없는 ID, 날짜, 문자열 컬럼 등 제외
        features_to_exclude = [
            TARGET, 'ENCODED_MCT', 'TA_YM', 'OPEN_DT', 'CLOSE_DT',
            'SIGUNGU_CD', 'IND_CD'
        ]
        # object 타입 컬럼 자동 제외
        features_to_exclude.extend(final_df.select_dtypes(include='object').columns.tolist())
        
        features = [col for col in final_df.columns if col not in features_to_exclude]
        
        X = final_df[features]
        y = final_df[TARGET]

        # 3. 모델 학습 및 평가
        if y.nunique() > 1: # 타겟에 0과 1이 모두 있는지 확인
            final_model, cv_scores = train_and_evaluate(X, y)

            # 4. 피처 중요도 시각화
            plot_feature_importance(final_model, features)
        else:
            print("타겟 변수에 클래스가 하나뿐이어서 모델을 학습할 수 없습니다.")

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"오류: 필요한 모듈을 찾을 수 없습니다. ({e})")

