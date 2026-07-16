# Lamination Peel Comparator

스마트폰 패널 하면의 보호필름을 풀테이프로 벗길 때, 진공척에 붙어 있는 상면 필름이 반대로 박리될 위험을 두 공정 조건(A/B) 사이에서 비교하는 Windows 데스크톱 시뮬레이터입니다.

> [!IMPORTANT]
> 이 프로그램은 아직 실측 데이터로 보정되지 않은 **상대 비교용 공정 스크리닝 도구**입니다. 결과의 위험도와 힘은 절대 불량률, 합격 판정 또는 보증값이 아닙니다. A와 B에 같은 가정을 적용했을 때 어느 궤적이 더 유리한지를 탐색하는 데 사용하세요.

## 입력 규칙

조건 A와 B에는 같은 형식의 값을 입력합니다.

| 항목 | 단위/규칙 |
|---|---|
| 패널 폭·길이·두께 | `mm` |
| PET 두께, PSA 두께 | `µm` |
| 상·하면 점착력 | `gf` (측정 폭·각도를 모르는 상대값) |
| 궤적 X, Y, Z | `mm` |
| 궤적 속도 | `mm/s` |
| 궤적 포인트 | 시작점을 포함해 정확히 6개 |

좌표계는 패널에 고정됩니다.

- 원점: 패널 좌하단
- `+X`: 패널 폭 방향
- `+Y`: 패널 길이 방향
- `+Z`: 패널 면에서 풀테이프가 들리는 방향

기본 속도 의미는 `Pi`에 입력한 속도가 `Pi → Pi+1` 직선 구간에 적용되는 **구간 속도**라는 뜻입니다. 따라서 P1~P5의 속도가 실제 이동 시간 계산에 사용되고, P6 속도는 파일 호환성을 위해 저장되지만 기본 계산에서는 사용하지 않습니다. 각 구간 시간은 `두 점 사이의 3D 거리 / 구간 속도`로 계산합니다.

`gf`는 힘 단위지만 시험 폭, 박리각, 속도를 모르면 계면 파괴에너지나 응력으로 유일하게 환산할 수 없습니다. 앱은 공통 시험 가정과 `1 gf = 0.00980665 N` 환산을 사용해 정규화된 위험도를 계산합니다. 비교 신뢰도를 높이려면 패널·필름·파지 조건을 A/B에서 잠그고 궤적만 변경하세요.

## 결과 해석

주요 비교값은 상면 최대 역박리 위험도, 임계 초과 면적, 패널 들림·비틀림, 하면 박리 진행률입니다. 다음 원칙으로 읽습니다.

- 두 조건 모두 하면 박리가 완료되는지 먼저 확인합니다.
- 그다음 상면 최대 위험도와 위험 면적이 더 낮은 조건을 찾습니다.
- 민감도 범위에서 A/B 순위가 자주 뒤집히면 `판정 보류`로 취급합니다.
- 위험도 `1.0`은 현재 가정에서의 임계값일 뿐, 실제 불량이 반드시 발생한다는 뜻이 아닙니다.

실제 합격 한계나 불량률을 예측하려면 최소한 동일 시험조건의 점착력(시험 폭·각도·속도), 공정 로드셀 곡선, 실제 역박리 시작 위치로 모델을 보정해야 합니다.

## 개발 환경에서 실행

Python 3.11 이상과 Windows PowerShell을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m lamination_sim
```

설치 후에는 다음 명령도 사용할 수 있습니다.

```powershell
lamination-peel
```

테스트 실행:

```powershell
python -m pytest
```

## Windows 패키징과 배포 점검

PyInstaller 설정은 Code ZIP으로 받아 바로 실행하기 쉬운 `onedir` 배포를 생성합니다. PyVista, VTK, PyTorch는 포함하지 않습니다.

```powershell
python -m PyInstaller --clean --noconfirm LaminationPeelComparator.spec
```

결과 실행 파일:

```text
dist\LaminationPeelComparator\LaminationPeelComparator.exe
```

이 저장소에는 `dist/LaminationPeelComparator` onedir 폴더가 함께 포함됩니다. GitHub에서 `Code → Download ZIP`으로 받은 뒤 압축을 풀고, 위 경로의 EXE를 실행하세요. EXE만 따로 이동하면 Qt·NumPy·SciPy 런타임 파일을 찾지 못하므로 폴더 전체를 유지해야 합니다.

배포 전에는 모든 파일의 개별 크기를 출력하고 GitHub 일반 Git 제한(100 MiB)을 넘는 파일이 없는지 검사합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\audit_dist.ps1
```

GUI가 시작 직후 종료되지 않는지 간단히 확인합니다. 스크립트는 앱을 숨김 상태로 실행해 지정 시간 동안 살아 있는지 확인한 뒤 종료합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_app.ps1
```

단일 파일이 100 MiB 이상이거나 긴 압축 해제 경로에서 Qt 런타임 문제가 발생하면 대형 파일을 그대로 커밋하거나 Git LFS에 의존하지 않습니다. 전체 onedir을 최대 90 MiB 조각으로 나누고, 조각 검증·재조립·짧은 로컬 경로 해제·실행을 담당하는 별도 런처 배포로 전환해야 합니다.

## 현재 모델의 한계

- 균일한 전면 진공척과 등가 재료물성을 가정합니다.
- 하면 필름 형상은 경량 기구학 근사이며 정밀 셸 FEA 결과가 아닙니다.
- 온도, 점탄성, 진공홀 국부 함몰, 주름, stick-slip은 보정 전에는 정량적으로 예측하지 않습니다.
- 알 수 없는 물성은 A/B에 동일하게 적용하므로 궤적의 상대 순위 분석에 초점을 둡니다.
