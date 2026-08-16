"""⑤ Ghost — 검출 오류형 해결. 로봇 마스크 포함률로 '손 위에 생긴 박스'를 찾는다.

배경: 꼬리 전파형은 오프라인 트리밍으로 해결했다. 남은 것은 검출기가 로봇 손을
물체로 '확신을 갖고' 잡는 경우라 크기·신뢰도로는 구분되지 않는다.

접근: task ②에서 로봇 팔·손이 분할된다는 것을 실측으로 확인했다. 그래서
대상 마스크 픽셀 중 몇 %가 로봇 영역에 속하는지(포함률)를 재고, 이 값이 높으면
'물체가 아니라 로봇을 잡은 것'으로 본다.

주의: 물체를 쥐고 있으면 당연히 겹친다. 그래서 단순 겹침(IoU)이 아니라
포함률(대상 ∩ 로봇 / 대상)을 쓴다. 쥐고 있는 물체는 그리퍼 사이로 표면이 보이므로
포함률이 1.0에 가깝지 않다. 임계값은 이 스크립트의 실측 분포를 보고 정한다.

**이 스크립트는 아무것도 제거하지 않는다.** 프레임별 포함률만 기록한다.
제거 여부는 집계 결과로 손익을 확인한 뒤 별도로 결정한다(소탐대실 방지).

출력: ~/task3/ghost2/<유닛키>.json + 요약 summary.json
"""
import os, sys, json, glob, shutil, time
import numpy as np
import cv2

ROOT = os.path.expanduser("~/task3")
sys.path.insert(0, f"{ROOT}/pipeline")
import run_review as RV

R6 = f"{ROOT}/review/r6"
OUT = f"{ROOT}/ghost2"
DEV = "cuda:0"

# 로봇 부위 프롬프트 — task ② 실측에서 검출률이 가장 높았던 조합
ROBOT_PROMPT = "robot hand . robot arm . gripper . robotic manipulator ."
ROBOT_THR = 0.30          # 로봇 부위 검출 임계 (낮게 잡아 놓치지 않게)
MAX_ROBOT_BOX = 6         # 프레임당 로봇 박스 상한


def log(m):
    print(m, flush=True)


def unit_key(u):
    return u.replace("/", "__")


def target_label(unit):
    s = json.load(open(f"{R6}/{unit}/stats.json"))
    for lb, v in s["stats"].items():
        if v.get("is_target"):
            return lb
    return None


def get_seq(unit):
    parts = unit.rstrip("/").split("/")
    if unit.startswith("brainco/"):
        task, ep = parts[1].split("_ep")
        files, tmp = RV.frames_brainco(task, int(ep), parts[2])
        return [(cv2.imread(f), None) for f in files], tmp
    return RV.frames_he(int(parts[1].split("_ep")[1])), None


def union_mask(masks, shape):
    m = np.zeros(shape, bool)
    for x in masks:
        m |= x
    return m


def main():
    from models_wrap import Detector, Segmenter
    os.makedirs(OUT, exist_ok=True)
    det, seg = Detector(DEV, box_thr=ROBOT_THR, text_thr=0.25), Segmenter(DEV)
    log("모델 로드 완료")

    units = sorted(glob.glob(f"{R6}/brainco/*/cam_*") + glob.glob(f"{R6}/he/*_ep*"))
    units = [u.replace(R6 + "/", "") for u in units if os.path.exists(f"{u}/frames.json")]
    log(f"대상 유닛 {len(units)}개")

    summary = {}
    for ui, unit in enumerate(units):
        key = unit_key(unit)
        outp = f"{OUT}/{key}.json"
        if os.path.exists(outp):
            log(f"[{ui+1}/{len(units)}] {unit} — 이미 있음, 건너뜀")
            continue
        t0 = time.time()
        lb = target_label(unit)
        if not lb:
            continue
        recs = json.load(open(f"{R6}/{unit}/frames.json"))
        try:
            seq, tmp = get_seq(unit)
        except Exception as e:
            log(f"  !! {unit}: 시퀀스 로드 실패 {e}")
            continue

        rows = []
        for i, r in enumerate(recs):
            tgt = [x for x in r["r"] if x["label"] == lb and x.get("accepted") and x.get("box2d")]
            if not tgt:
                continue
            x = tgt[0]
            if i >= len(seq) or seq[i][0] is None:
                continue
            img = seq[i][0]
            H, W = img.shape[:2]
            # 대상 마스크 (저장된 box2d를 SAM에 재투입 — 파이프라인과 동일)
            tm = seg(img, np.array([x["box2d"]], dtype=np.float32))
            if len(tm) == 0:
                continue
            tmask = tm[0]
            tpx = int(tmask.sum())
            if tpx < 20:
                continue
            # 로봇 부위 마스크 (합집합)
            boxes, phrases, scores = det(img, ROBOT_PROMPT)
            rmask = np.zeros((H, W), bool)
            n_rb = 0
            if len(boxes):
                order = np.argsort(-scores)[:MAX_ROBOT_BOX]
                rb = np.array([boxes[j] for j in order], dtype=np.float32)
                rms = seg(img, rb)
                if len(rms):
                    rmask = union_mask(list(rms), (H, W))
                    n_rb = len(rms)
            inter = int((tmask & rmask).sum())
            rows.append(dict(f=i, src=x.get("src"), det_score=round(float(x.get("det_score", 0)), 3),
                             tgt_px=tpx, robot_px=int(rmask.sum()), inter_px=inter,
                             contain=round(inter / max(tpx, 1), 4),
                             n_robot_box=n_rb,
                             size_cm=[round(v * 100, 1) for v in x.get("size", [])] if x.get("size") else None))
        json.dump(dict(unit=unit, label=lb, n=len(rows), rows=rows),
                  open(outp, "w"), ensure_ascii=False)
        if rows:
            cs = np.array([r["contain"] for r in rows])
            summary[unit] = dict(label=lb, n=len(rows),
                                 contain_median=round(float(np.median(cs)), 3),
                                 contain_p90=round(float(np.percentile(cs, 90)), 3),
                                 n_over_70=int((cs >= 0.70).sum()),
                                 n_over_85=int((cs >= 0.85).sum()),
                                 n_over_95=int((cs >= 0.95).sum()))
            log(f"[{ui+1}/{len(units)}] {unit} n={len(rows)} "
                f"포함률 중앙={summary[unit]['contain_median']} "
                f">=0.85 {summary[unit]['n_over_85']}건 ({time.time()-t0:.0f}s)")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        json.dump(summary, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)

    log(f"완료 -> {OUT}")


if __name__ == "__main__":
    main()
