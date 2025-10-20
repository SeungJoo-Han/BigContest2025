import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def set_korean_font():
    """Matplotlib에서 한글 폰트를 설정합니다."""
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
    except:
        print("Malgun Gothic font not found. Please install it for Korean characters.")

def plot_feature_importance(model, feature_names, top_n=20):
    """
    학습된 모델의 피처 중요도를 시각화합니다.

    Args:
        model: 학습된 LightGBM 모델
        feature_names (list): 피처 이름 목록
        top_n (int): 시각화할 상위 피처 개수
    """
    set_korean_font()
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index=feature_names)
    ftr_top = ftr_importances.sort_values(ascending=False)[:top_n]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=ftr_top.values, y=ftr_top.index)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
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

    # LightGBM이 인식할 수 있도록 범주형 피처를 'category' 타입으로 변환
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        X[col] = X[col].astype('category')
    print(f"범주형 피처로 처리될 컬럼: {categorical_features}")

    # 시계열 교차 검증 설정
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # 점수를 저장할 딕셔너리 초기화
    scores = {'auprc': [], 'f1': [], 'recall': [], 'precision': []}

    # LightGBM 모델 파라미터
    params = {
        'objective': 'binary',
        'metric': 'average_precision', # aucpr?
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'is_unbalance': True,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,        
        }
    
    """        
        'verbose': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'scale_pos_weight': scale_pos_weight
        """
    
    # 교차 검증 루프
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(**params)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  categorical_feature=categorical_features
                  )

        preds = model.predict(X_val)
        pred_proba = model.predict_proba(X_val)[:, 1]

        scores['auprc'].append(average_precision_score(y_val, pred_proba))
        scores['f1'].append(f1_score(y_val, preds))
        scores['recall'].append(recall_score(y_val, preds))
        scores['precision'].append(precision_score(y_val, preds))
        
        print(f"Fold {fold+1} AUPRC: {scores['auprc'][-1]:.4f}")

    print("\n--- 교차 검증 결과 ---")
    print(f"평균 AUPRC: {np.mean(scores['auprc']):.4f} ± {np.std(scores['auprc']):.4f}")
    print(f"평균 F1 Score: {np.mean(scores['f1']):.4f} ± {np.std(scores['f1']):.4f}")
    
    print("\n===== 전체 데이터로 최종 모델 학습 시작 =====")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y, categorical_feature=categorical_features)
    print("===== 최종 모델 학습 완료 =====")
    
    # 모델 저장
    MODEL_OUTPUT_DIR = '../BigContest2025-main/model'
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'lgbm_final_model.pkl')
    joblib.dump(final_model, model_path)
    print(f"학습된 최종 모델이 '{model_path}'에 저장되었습니다.")

    return final_model, scores

if __name__ == '__main__':
    try:
        DATA_PATH = '../BigContest2025-main/data/labeled_data.csv'
        print(f"데이터 로딩 중... ({DATA_PATH})")
        final_df = pd.read_csv(DATA_PATH)

        # 결측값이 있는 행 제거 (모델 학습 전 처리 필요)
        final_df.reset_index(drop=True, inplace=True)
        final_df = final_df.dropna(subset=['is_at_risk'])
        final_df = final_df.fillna(0)

        # 타겟 변수 및 피처 분리
        TARGET = 'is_at_risk'
        # 모델이 학습할 수 없는 ID, 날짜, 문자열 컬럼 등 제외
        features_to_exclude = [
            TARGET, 'ENCODED_MCT', 'MCT_NM', 'MCT_BSE_AR', 'MCT_BRD_NUM',
            'TA_YM', 'ARE_D', 'MCT_ME_D',
            'MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT'
        ]
        # object 타입 컬럼 자동 제외
        features_to_exclude.extend(final_df.select_dtypes(include='object').columns.tolist())
        
        features = [col for col in final_df.columns if col not in features_to_exclude]
        
        X = final_df[features]
        y = final_df[TARGET]

        print(f"학습 데이터 크기: {X.shape}")
        print(f"사용할 피처 개수: {len(features)}개")

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