# Coupled Matrix 학습 가이드

`ml/scripts/coupled/train_matrix.py`로 언어/형식 조합을 한 번에 학습할 수 있습니다.

## 1) 전체 일괄 학습 (KR+JA, 자동 형식 탐색)
```powershell
python -X utf8 .\ml\scripts\coupled\train_matrix.py `
  --dataset-root ".\dataset_staged" `
  --workspace-root ".\ml_workspace" `
  --model-root ".\ML_models" `
  --languages "korean,japanese" `
  --formats auto `
  --auto-oto-policy require `
  --backend coupled_nn_v2_rawmel `
  --device cuda
```

## 2) 언어별 형식 지정 학습
```powershell
python -X utf8 .\ml\scripts\coupled\train_matrix.py `
  --dataset-root ".\dataset_staged" `
  --languages "korean,japanese" `
  --formats "korean=cv,cvc,cvvc,vcv;japanese=cv,cvvc,vcv" `
  --auto-oto-policy require `
  --backend coupled_nn_v2_rawmel `
  --device cuda
```

## 3) 학습 없이 CSV만 생성
```powershell
python -X utf8 .\ml\scripts\coupled\train_matrix.py `
  --dataset-root ".\dataset_staged" `
  --languages "korean,japanese" `
  --formats auto `
  --skip-train `
  --auto-oto-policy require
```

## 4) rawmel 캐시 생성 (CSV 생성 후 권장)
단일 CSV 예시:
```powershell
python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvc.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing
```

생성된 모든 CSV에 대해 일괄 캐시 생성:
```powershell
Get-ChildItem ".\ml_workspace\datasets" -Recurse -Filter "dataset_*.csv" | ForEach-Object {
  $csv = $_.FullName
  python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
    --dataset $csv `
    --dataset-root ".\dataset_staged" `
    --allow-missing
}
```

## 5) CSV 재사용(빌드 스킵) 학습
```powershell
python -X utf8 .\ml\scripts\coupled\train_matrix.py `
  --dataset-root ".\dataset_staged" `
  --languages "korean,japanese" `
  --formats auto `
  --skip-build `
  --backend coupled_nn_v2_rawmel `
  --device cuda
```

## 자주 쓰는 옵션
- `--backend`: `coupled_nn_v1` 또는 `coupled_nn_v2_rawmel`
- `--rawmel-cache`: v2 전용 캐시 경로(미지정 시 자동 탐색)
- `--min-mapping-confidence`: `<0`이면 언어/형식 기본값 사용
- `--require-all`: 선택한 조합 중 하나라도 실패하면 전체 실패 처리
- `--dry-run`: 실제 실행 없이 대상 조합만 확인
