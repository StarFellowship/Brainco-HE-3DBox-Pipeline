"""⑤ 규칙 확정 — 육안 검증된 정답(19프레임)에 대해 규칙 후보를 채점한다.

정답은 에이전트 5명이 각 프레임을 실제로 열어 판정한 결과다.
규칙은 '정상 박스를 하나도 지우지 않는다'를 절대 조건으로 하고,
그 안에서 ghost 제거를 최대화하는 것을 고른다.
"""
import os, json, glob
R6 = os.path.expanduser("~/task3/review/r6")

GT = {  # (유닛, 라벨, 프레임): is_ghost
 ("brainco/PickApple_ep5/cam_left_wrist","plate",166):True,
 ("brainco/PickApple_ep5/cam_left_wrist","plate",193):True,
 ("brainco/PickApple_ep5/cam_left_wrist","plate",219):True,
 ("brainco/PickCharger_ep5/cam_left_high","charger",17):False,
 ("brainco/PickCharger_ep5/cam_left_high","charger",29):False,
 ("brainco/PickCharger_ep5/cam_left_high","charger",42):False,
 ("brainco/PickDrink_ep5/cam_left_wrist","plate",180):True,
 ("brainco/PickDrink_ep5/cam_left_wrist","plate",189):True,
 ("brainco/PickDrink_ep5/cam_left_wrist","plate",218):True,
 ("brainco/PickDrink_ep5/cam_right_wrist","bottle",32):True,
 ("brainco/PickTissues_ep5/cam_left_wrist","tissue pack",16):True,
 ("brainco/PickTissues_ep5/cam_left_wrist","tissue pack",25):True,
 ("brainco/PickTissues_ep5/cam_left_wrist","tissue pack",30):True,
 ("brainco/PickTissues_ep5/cam_right_wrist","plate",7):False,
 ("brainco/PickTissues_ep5/cam_right_wrist","plate",34):True,
 ("brainco/PickTissues_ep5/cam_right_wrist","plate",53):True,
 ("brainco/PickToothpaste_ep5/cam_right_wrist","toothpaste",27):False,
 ("brainco/PickToothpaste_ep5/cam_right_wrist","toothpaste",30):False,
 ("brainco/PickToothpaste_ep5/cam_right_wrist","toothpaste",32):False,
}
PERFECT=("GraspOreo","GraspRubiksCube")

def load(u):
    return (json.load(open(f"{R6}/{u}/frames.json")),
            json.load(open(f"{R6}/{u}/stats.json"))["stats"])

def removed_set(u, lb, n_confirm, only_never):
    """규칙이 제거할 프레임 집합."""
    recs,_ = load(u)
    seen, est, rem = 0, None, []
    for i,r in enumerate(recs):
        for x in r["r"]:
            if x["label"]!=lb or not x.get("accepted"): continue
            if x.get("src") in ("det","redet"):
                seen += 1
                if est is None and seen>=n_confirm: est=i
            elif x.get("src")=="prop" and est is None:
                rem.append(i)
    if only_never and est is not None:
        return set(), est          # 확립된 라벨은 건드리지 않는다
    return set(rem), est

units=[u.replace(R6+"/","") for u in sorted(glob.glob(f"{R6}/brainco/*/cam_*")+glob.glob(f"{R6}/he/*_ep*"))
       if os.path.exists(f"{u}/frames.json")]

print(f"{'규칙':<34}{'정상오삭제':>10}{'ghost제거':>10}{'완벽손실':>9}{'총삭제':>8}  판정")
best=None
for only_never in (True, False):
    for n in (3,4,5,6,7,8,10):
        wrong=hit=0
        for (u,lb,f),g in GT.items():
            rem,_ = removed_set(u,lb,n,only_never)
            if f in rem:
                if g: hit+=1
                else: wrong+=1
        lossP=tot=0
        for u in units:
            _,st = load(u)
            for lb in st:
                rem,_ = removed_set(u,lb,n,only_never)
                tot += len(rem)
                if any(p in u for p in PERFECT): lossP += len(rem)
        name = f"{'미확립라벨만' if only_never else '초반전파'} N={n}"
        ok = (wrong==0 and lossP==0)
        print(f"{name:<34}{wrong:>10}{hit:>10}{lossP:>9}{tot:>8}  {'채택 가능' if ok else '불가'}")
        if ok and (best is None or tot>best[3]): best=(name,only_never,n,tot,hit)
print()
if best:
    print(f"채택: {best[0]} — 총 {best[3]}프레임 제거, 검증 ghost {best[4]}건 적중, 오삭제·완벽손실 0")
    print("\n  제거 내역")
    for u in units:
        _,st = load(u)
        for lb in st:
            rem,est = removed_set(u,lb,best[2],best[1])
            if rem:
                print(f"    {u:<44} {lb:<13} {len(rem):>4}프레임 (확립 {est})")
else:
    print("조건을 만족하는 규칙 없음")
