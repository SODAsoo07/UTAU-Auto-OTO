# 개발현황_Mel

## 현재 기능
- 멜+OTO 커플링 모델(`coupled_nn_v1`) 학습/추론 파이프라인 유지.
- 멜 클래스/blank 신호를 patch 입력에 직접 포함.
- patch 피처에 onset/tail 주변 에너지, voicing, `syllable_mel_*`, `syllable_blank_confidence` 포함.
- 멀티태스크 헤드 추가.
- aux 타깃: `aux_vowel_start_rel`, `aux_vowel_end_rel`, `aux_next_onset_rel`.
- 유효 마스크 기반 보조 손실로 신뢰 가능한 구간만 학습.
- 파일 단위 컨텍스트 피처 추가.
- 동일 wav 내 `CV/VC/VV/VCV/BR/MONO` 분포(카운트/비율/밸런스) 피처 포함.
- VC↔CV 페어링 손실(양방향) 추가.
- 같은 wav 내 가까운 VC/CV 간 offset gap 보존을 위한 추가 loss.
- 양방향 적용 기준으로 페어링 가중치는 내부적으로 0.5 스케일.
- 홀드아웃 기본화: `group-column` 기본값을 `wav_norm`으로 변경.
- 같은 파일의 VC/CV가 train/valid에 섞이지 않도록 기본 분리.
- alias 정규화 시 숨소리 변형(`吸こ吐`, `吸き吐`, `吸き吐_A3`)을 `br`로 처리.
- alias 뒤의 주석성 접미사(한자/한글/기타 메모)를 제거한 뒤 alias 분류/정규화 수행.
- 데이터 폴더에서 `prefix.map`뿐 아니라 affix 규칙으로 파싱 가능한 다른 `*.map`도 읽어서 접두/접미를 제거.
- `prefix.map`과 다른 `*.map`이 같이 있으면 `prefix.map`을 우선 사용.

## 구현 의도
- 오매핑(점프/역행) 감소를 위해 멜 클래스/blank 신호를 모델 입력에 직접 반영.
- 음절 경계 안정성 확보를 위해 모음 시작/끝 및 다음 onset 예측을 함께 학습.
- 동일 녹음 내 VC/CV 관계를 모델이 학습하도록 파일 단위 컨텍스트 및 페어링 손실 추가.
- 파일 단위 분리를 기본으로 하여 관계 학습의 누수/과적합 리스크를 낮춤.
- 실사용 voicebank에서 자주 보이는 숨소리 변형, 접미 메모, pitch affix, `prefix.map` 계열 affix 때문에 alias 매칭이 흔들리지 않도록 전처리를 강건하게 유지.

## 학습 손실 구성(요약)
- 기본 타깃 손실: `SmoothL1`, 타깃 가중치 적용.
- 구조 제약 손실: `pre/cons/cutoff/offset` 관계 위반 패널티.
- 멜 경계 정렬 손실: `offset/cutoff_abs`를 mel 후보 경계에 정렬.
- confidence 손실: 예측 오차 기반 신뢰도 학습.
- aux 손실: 보조 타깃에 대한 `SmoothL1`, 유효 마스크 반영.
- VC↔CV 페어 손실: 같은 wav 내 페어의 offset gap 보존.
- 페어링 가중치: `UTOA_ML_VC_CV_PAIR_WEIGHT` 값을 내부적으로 0.5 스케일 적용.

## 데이터/캐시 구조
- `FEATURE_VERSION` / `TRAIN_ROW_MATCH_VERSION` 변경 시 캐시 무효화.
- dataset CSV에 aux 타깃 컬럼이 포함됨.
- mel/feature 캐시 경로: `.cache/oto_ml_features`, `.cache/oto_ml_training_rows`.
- 파일 단위 컨텍스트 피처는 feature row 생성 시 wav 단위로 계산됨.
- prefix affix 제거에 사용된 map 파일 경로도 feature/training row 캐시 키에 반영됨.

## alias 전처리 규칙
- `_A3`, `-C4`, ` F4` 같은 음계 affix는 alias 본체에서 제거 후 매칭.
- alias 뒤에 붙은 한자/한글/메모성 접미사는 제거 후 분류.
- 숨소리 alias는 본문이 조금 달라도 최종적으로 `br`로 정규화.
- `prefix.map` 또는 규칙형 `*.map`이 있으면 note별 prefix/suffix를 먼저 제거.
- 위 과정을 거친 뒤 `alias_norm`, `alias_type`, `occurrence_index`가 결정됨.

## 재학습/운용 절차(권장)
1. dataset CSV 재생성.
2. 학습 실행(`group-column = wav_norm` 기본).
3. `eval_summary.json`, `model_meta.json` 확인.
4. 모델 번들 export/install 후 샘플 평가.

## 전처리를 다시 해야 하는 경우
- `FEATURE_VERSION` 또는 `TRAIN_ROW_MATCH_VERSION`이 올라간 경우.
- alias 정규화 규칙이 바뀐 경우.
- 숨소리 분류 규칙이 바뀐 경우.
- 접미 메모 제거 규칙이 바뀐 경우.
- `prefix.map` / `*.map` affix 처리 규칙이 바뀐 경우.
- 위 경우에는 기존 CSV와 캐시를 재사용하지 말고 `build_dataset`부터 다시 수행하는 것이 안전.

## 앞으로 보완해야 하는 부분
- 페어링 기준을 offset 외에 `pre_abs`/anchor 기반으로 확장 검토.
- 페어링 `max_gap`, `weight` 튜닝 가이드 문서화 및 자동 스윕 옵션 추가 검토.
- aux loss/weight, pair loss/weight에 대한 하이퍼파라미터 실험 기록 자동화.
- 장모음/삽입모음/youon 하드 네거티브 마이닝 확장 검토.
- 페어링 손실에 alias_type 가중치(`VC stop/sonorant` 등) 반영 옵션 검토.
- 품질 낮은 pseudo 라벨의 사용 비중 자동 조절 로직 개선.
- `.map` 후보 선택 기준을 더 엄격하게 할 필요가 있는지 실데이터 기준 검토.

## 나중에 수정할 때 필요한 사항
- `PATCH_FEATURES` 변경 시 모델 저장/로딩에 반영 필요.
- `AUX_TARGET_NAMES` 변경 시 `dataset_fieldnames`, 학습 로직, 메타 기록 동기화 필요.
- 홀드아웃 정책 변경은 학습 스크립트 기본값도 함께 업데이트해야 함.
- 환경 변수: `UTOA_ML_VC_CV_PAIR_WEIGHT` (기본 0.06, 양방향 적용으로 내부 0.5 스케일)
- 환경 변수: `UTOA_ML_VC_CV_MAX_GAP` (기본 5)
- 모델 메타 기록 필드(`model_meta.json`, `eval_summary.json`) 변경 시 호환성 확인 필요.
- 캐시 무효화가 필요한 경우 버전 상승 또는 캐시 삭제 필요.
- `.map` 처리 로직을 수정하면 실데이터에 어떤 map 파일이 선택되는지 샘플 bank로 반드시 검증할 것.

## 실험 기록 템플릿
- 실험 ID:
- 날짜:
- 데이터셋 버전(`FEATURE_VERSION`):
- 학습 설정: `epochs`, `batch`, `lr`, `group-column`, `device`
- 페어링 설정: `UTOA_ML_VC_CV_PAIR_WEIGHT`, `UTOA_ML_VC_CV_MAX_GAP`
- aux/pair 가중치: `aux_weight`, `pair_weight`
- alias 전처리 규칙 변경 여부:
- 사용된 map affix 처리 여부:
- 결과 요약: `holdout_metrics`, `aux_holdout_metrics`, `confidence_mean`
- 관찰: 개선점/퇴행점/특이 케이스
- 다음 액션:

## 릴리스 체크리스트
- 데이터셋 재생성 완료(`FEATURE_VERSION` 최신 반영)
- 학습 완료 및 `eval_summary.json` 확인
- `model_meta.json` 기록 필드 확인
- 샘플 음원 기준 수동 검증(대표 케이스 포함)
- 숨소리/접미 메모/affix alias 케이스 수동 검증
- fallback(lightgbm/base) 동작 확인
- 설치/번들 경로 검증(`models_installed`, `exports`)
- 변경사항 문서화(`개발현황_Mel` 갱신)
