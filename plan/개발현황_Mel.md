# 개발현황_Mel

## 현재 기능
- 멜+OTO 커플링 모델(coupled_nn_v1) 학습/추론 파이프라인 유지.
- 멜 클래스/blank 신호를 patch 입력에 직접 포함.
- patch 피처에 onset/tail 주변 에너지/voicing 및 syllable_mel_* / syllable_blank_confidence 포함.
- 멀티태스크 헤드 추가로 보조 타깃 학습.
- aux 타깃: aux_vowel_start_rel, aux_vowel_end_rel, aux_next_onset_rel.
- 유효 마스크 기반 보조 손실로 신뢰 가능한 구간만 학습.
- 파일 단위 컨텍스트 피처 추가.
- 동일 wav 내 CV/VC/VV/VCV/BR/MONO 분포(카운트/비율/밸런스) 피처 포함.
- VC↔CV 페어링 손실(양방향) 추가.
- 같은 wav 내 가까운 VC/CV 간 offset gap 보존을 위한 추가 loss.
- 양방향 적용으로 페어링 가중치는 내부적으로 0.5 스케일.
- 홀드아웃 기본화: group-column 기본값을 wav_norm으로 변경.
- 같은 파일의 VC/CV가 train/valid에 섞이지 않도록 기본 분리.

## 구현 의도
- 오매핑(점프/역행) 감소를 위해 멜 클래스/blank 신호를 모델 입력에 직접 반영.
- 음절 경계 안정성 확보를 위해 모음 시작/끝 및 다음 onset 예측을 함께 학습.
- 동일 녹음 내 VC/CV 관계를 모델이 학습하도록 파일 단위 컨텍스트 및 페어링 손실 추가.
- 파일 단위 분리를 기본으로 하여 관계 학습의 누수/과적합 리스크를 낮춤.

## 학습 손실 구성(요약)
- 기본 타깃 손실: SmoothL1, 타깃 가중치 적용.
- 구조 제약 손실: pre/cons/cutoff/offset 관계 위반 패널티.
- 멜 경계 정렬 손실: offset/cutoff_abs를 mel 후보 경계에 정렬.
- confidence 손실: 예측 오차 기반 신뢰도 학습.
- aux 손실: 보조 타깃에 대한 SmoothL1, 유효 마스크 반영.
- VC↔CV 페어 손실: 같은 wav 내 페어의 offset gap 보존.
- 페어링 가중치: UTOA_ML_VC_CV_PAIR_WEIGHT 값을 0.5로 스케일 적용.

## 데이터/캐시 구조
- FEATURE_VERSION / TRAIN_ROW_MATCH_VERSION 변경 시 캐시 무효화.
- dataset CSV에 aux 타깃 컬럼이 포함됨.
- mel/feature 캐시 경로: .cache/oto_ml_features, .cache/oto_ml_training_rows.
- 파일 단위 컨텍스트 피처는 feature row 생성 시 wav 단위로 계산됨.

## 재학습/운용 절차(권장)
1. dataset CSV 재생성(FEATURE_VERSION 업데이트 반영).
2. 학습 실행(기본 group-column = wav_norm).
3. eval_summary.json 및 model_meta.json 확인.
4. 모델 번들 export/install 후 샘플 평가.

## 앞으로 보완해야 하는 부분
- 페어링 기준을 offset 외에 pre_abs/anchor 기반으로 확장 검토.
- 페어링 max_gap, weight 튜닝 가이드 문서화 및 자동 스윕 옵션 추가 검토.
- aux loss/weight, pair loss/weight에 대한 하이퍼파라미터 실험 기록 자동화.
- 장모음/삽입모음/youon 하드 네거티브 마이닝 확장(필요시 contrastive 추가).
- 페어링 손실에 alias_type 가중치(VC stop/sonorant 등) 반영 옵션 추가 검토.
- 품질 낮은 pseudo 라벨의 사용 비중 자동 조절 로직 개선.

## 나중에 수정할 때 필요한 사항
- PATCH_FEATURES 변경 시 모델 저장/로딩에 반영 필요.
- AUX_TARGET_NAMES 변경 시 dataset_fieldnames, 학습 로직, 메타 기록 동기화 필요.
- 홀드아웃 정책 변경은 학습 스크립트 기본값을 함께 업데이트해야 함.
- 환경 변수: UTOA_ML_VC_CV_PAIR_WEIGHT (기본 0.06, 양방향 적용으로 내부 0.5 스케일)
- 환경 변수: UTOA_ML_VC_CV_MAX_GAP (기본 5)
- 모델 메타 기록 필드(model_meta.json, eval_summary.json) 변경 시 호환성 확인 필요.
- 캐시 무효화가 필요한 경우 버전 상승 또는 캐시 삭제 필요.

## 실험 기록 템플릿
- 실험 ID:
- 날짜:
- 데이터셋 버전(FEATURE_VERSION):
- 학습 설정: epochs, batch, lr, group-column, device
- 페어링 설정: UTOA_ML_VC_CV_PAIR_WEIGHT, UTOA_ML_VC_CV_MAX_GAP
- aux/pair 가중치: aux_weight, pair_weight
- 결과 요약: holdout_metrics, aux_holdout_metrics, confidence_mean
- 관찰: 개선점/퇴행점/특이 케이스
- 다음 액션:

## 릴리스 체크리스트
- 데이터셋 재생성 완료(FEATURE_VERSION 최신 반영)
- 학습 완료 및 eval_summary.json 확인
- model_meta.json 기록 필드 확인
- 샘플 음원 기준 수동 검증(대표 케이스 포함)
- fallback(lightgbm/base) 동작 확인
- 설치/번들 경로 검증(models_installed/exports)
- 변경사항 문서화(개발현황_Mel 갱신)
