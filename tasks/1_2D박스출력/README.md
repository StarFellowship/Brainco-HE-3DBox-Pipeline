# ① 2D Bounding Box 출력 — 완료

**요구**: 3D 박스만 내던 파이프라인 출력에 2D 박스도 포함.

**방법**: 2D 박스는 1단계(GroundingDINO)에서 이미 프레임별로 계산되므로,
GPU 재실행 없이 후처리로 표준 산출물 2종을 생성.
- `boxes2d.json` — 프레임별 label·좌표(xyxy)·신뢰도·대상 여부 (이 폴더의 예시 참조)
- `BOX2D.mp4` — 2D 박스만 그린 오버레이 영상 (드라이브 공유본)

**코드**: [`pipeline/postprocess.py`](../../pipeline/postprocess.py) (`box2d` 명령)
**결과**: 39유닛 전체 — 박스 10,080개, 영상 39편. `results/최종결과/brainco/*/*/boxes2d.json` · `results/최종결과/he/*/boxes2d.json`
