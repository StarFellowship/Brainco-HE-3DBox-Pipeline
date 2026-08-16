"""⑤ Ghost — cold-start 가드 시뮬레이션 (CPU, 이미지 불필요).

남은 ghost 유형은 '검출기가 물체 아닌 것을 물체로 잡는' 경우다. 후처리 필터 두 가지
(크기 필터, 로봇마스크 포함률)는 완벽 유닛을 깎아 이미 기각했다. 여기서는 세 번째,
**시점 기반** 규칙을 시험한다. 이미지가 필요 없어 마운트 없이 검증 가능하다.

규칙: 에피소드 초반, 그 라벨이 아직 '확립'되지 않은 동안에는 전파(prop) 승인을 막는다.
      확립 = 실검출(det/redet)이 CONFIRM_N번 누적된 시점.
근거: 사수 육안 검증에서 지적된 '사과 2~5초 ghost'가 초반 구간이다. 물체를 아직 한 번도
      제대로 못 봤는데 추적 창만 떠다니는 상태라, 이 구간의 전파는 근거가 없다.

안전 원칙: 완벽 유닛(GraspOreo·GraspRubiksCube) 손실 0을 만족하는 설정만 채택한다.
이 스크립트는 제거하지 않고 '제거될 프레임 수'만 센다.
"""
import os, json, glob
import numpy as np

R6 = os.path.expanduser("~/task3/review/r6")
OUT = os.path.expanduser("~/task3/coldstart")
PERFECT = ("GraspOreo", "GraspRubiksCube")


def units():
    us = sorted(glob.glob(f"{R6}/brainco/*/cam_*") + glob.glob(f"{R6}/he/*_ep*"))
    return [u.replace(R6 + "/", "") for u in us if os.path.exists(f"{u}/frames.json")]


def is_perfect(u):
    return any(p in u for p in PERFECT)


def simulate(unit, confirm_n):
    """반환: (제거프레임수, 대상라벨수, 라벨별 상세)"""
    recs = json.load(open(f"{R6}/{unit}/frames.json"))
    stats = json.load(open(f"{R6}/{unit}/stats.json"))["stats"]
    detail = {}
    total_removed = 0
    for lb, v in stats.items():
        seen = 0
        established_at = None
        removed = []
        for i, r in enumerate(recs):
            for x in r["r"]:
                if x["label"] != lb or not x.get("accepted"):
                    continue
                src = x.get("src")
                if src in ("det", "redet"):
                    seen += 1
                    if established_at is None and seen >= confirm_n:
                        established_at = i
                elif src == "prop" and established_at is None:
                    removed.append(i)
        if removed:
            detail[lb] = dict(n_removed=len(removed), established_at=established_at,
                              first=removed[0], last=removed[-1],
                              is_target=bool(v.get("is_target")))
            total_removed += len(removed)
    return total_removed, detail


def main():
    os.makedirs(OUT, exist_ok=True)
    us = units()
    print(f"유닛 {len(us)}개 (완벽 {sum(is_perfect(u) for u in us)}개)\n")
    print(f"{'CONFIRM_N':>10}{'완벽유닛 손실':>14}{'그외 제거':>10}   판정")
    table = []
    for cn in (1, 2, 3, 5, 8):
        lossP = lossO = 0
        for u in us:
            n, _ = simulate(u, cn)
            if is_perfect(u):
                lossP += n
            else:
                lossO += n
        ok = lossP == 0
        table.append(dict(confirm_n=cn, perfect_loss=lossP, other_removed=lossO, safe=ok))
        print(f"{cn:>10}{lossP:>14}{lossO:>10}   {'채택 가능' if ok else '불가'}")

    safe = [t for t in table if t["safe"]]
    best = max(safe, key=lambda t: t["other_removed"]) if safe else None
    print()
    if not best or best["other_removed"] == 0:
        print("완벽 유닛을 건드리지 않으면서 제거되는 프레임이 없다 → 이 규칙은 효과 없음.")
        verdict = "효과 없음"
    else:
        print(f"채택: CONFIRM_N={best['confirm_n']} — 완벽 유닛 손실 0, 그 외 {best['other_removed']}프레임 제거")
        verdict = f"채택 CONFIRM_N={best['confirm_n']}"
        print("\n  유닛별 제거 내역")
        for u in us:
            n, d = simulate(u, best["confirm_n"])
            if n:
                for lb, dd in d.items():
                    print(f"    {u:<44} {lb:<14} {dd['n_removed']:>4}프레임 "
                          f"(f{dd['first']}~{dd['last']}, 확립 f{dd['established_at']}, "
                          f"{'대상' if dd['is_target'] else '참조'})")
    json.dump(dict(verdict=verdict, table=table), open(f"{OUT}/summary.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n판정: {verdict}")


if __name__ == "__main__":
    main()
