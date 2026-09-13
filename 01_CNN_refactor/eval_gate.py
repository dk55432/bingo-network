"""Ship gate: refuse to deploy a CNN checkpoint that regresses.

The retrain loop fine-tunes cell_classifier_phone.pth on user-confirmed
cells and used to overwrite the runtime checkpoint no questions asked.  This
module evaluates a candidate checkpoint against the incumbent on the SAME
held-out valid split (the flatbed-scan phone_sheets 66..82 partition that
training itself scores on) and refuses to let train_learning.py save a
checkpoint whose valid-sample accuracy drops more than a small tolerance.

Both models are evaluated with one deterministic loader, so the comparison
is apples-to-apples; the training JIT (DataLoader order) is irrelevant
because we only count correct/total.

CLI:
    python eval_gate.py --candidate new.pth
    python eval_gate.py --candidate new.pth --incumbent baseline.pth --tol 0.01

Exit code: 0 = shipped, 1 = rejected (candidate regressed past tolerance).
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent))

from train_phone_cells import CellClassifier, DST, build, test_tf  # noqa: E402

CKPT = Path(__file__).parent / "cell_classifier_phone.pth"
# Regression budget: candidate valid acc may trail the incumbent by at most
# this much (absolute, 0.005 = 0.5pp).  Tuned to absorb float/repro noise
# (the valid split is finite, ~1.2k cells, so ±1 misclassification is
# ~0.1pp) while still rejecting a genuinely worse model.
DEFAULT_TOL = 0.005


def ensure_valid_split():
    if (DST / "valid").is_dir():
        return
    print("held-out valid split missing - building base dataset first ...")
    build()


def valid_loader():
    ensure_valid_split()
    return DataLoader(ImageFolder(DST / "valid", test_tf), batch_size=32,
                      shuffle=False, num_workers=0)


def evaluate_state(sd, loader, device) -> dict:
    """Validate one checkpoint's state_dict against the held-out split."""
    ncls = sd["classifier.4.weight"].shape[0]
    model = CellClassifier(ncls).to(device)
    model.load_state_dict(sd)
    model.eval()
    correct = total = 0
    per_class = {}
    with torch.no_grad():
        for imgs, labs in loader:
            pr = model(imgs.to(device)).argmax(1).cpu()
            correct += int((pr == labs).sum())
            total += labs.size(0)
            for lab, pred in zip(labs.tolist(), pr.tolist()):
                st = per_class.setdefault(lab, [0, 0])
                st[0] += int(pred == lab)
                st[1] += 1
    return {
        "acc": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "per_class": {str(k): (ok, tot) for k, (ok, tot) in per_class.items()},
    }


def load_state(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def gate(candidate_path, incumbent_path=CKPT, tol=DEFAULT_TOL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = valid_loader()
    inc = evaluate_state(load_state(incumbent_path, device), loader, device)
    cand = evaluate_state(load_state(candidate_path, device), loader, device)
    delta = cand["acc"] - inc["acc"]
    ok = delta >= -tol
    cls_deltas = {}
    for lab in sorted(
            set(inc["per_class"]) | set(cand["per_class"])):
        i = inc["per_class"].get(lab, (0, 0))
        c = cand["per_class"].get(lab, (0, 0))
        ia = i[0] / i[1] if i[1] else 0.0
        ca = c[0] / c[1] if c[1] else 0.0
        if ca < ia - 0.05:
            cls_deltas[lab] = (ia, ca)
    return {
        "ok": ok,
        "tol": tol,
        "delta": delta,
        "incumbent": inc,
        "candidate": cand,
        "worst_classes": cls_deltas,
    }


def print_report(res):
    inc, cand = res["incumbent"], res["candidate"]
    print("SHIP GATE report")
    print(f"  held-out valid split: {inc['total']} cells")
    print(f"  incumbent : acc={inc['acc']:.4f} ({inc['correct']}/{inc['total']})")
    print(f"  candidate : acc={cand['acc']:.4f} ({cand['correct']}/{cand['total']})")
    print(f"  delta     : {res['delta']:+.4f} (tol {res['tol']:+.4f})")
    if res["worst_classes"]:
        print("  classes where candidate lost >=5pp:")
        for lab, (ia, ca) in sorted(
                res["worst_classes"].items(),
                key=lambda kv: kv[1][0] - kv[1][1], reverse=True):
            print(f"    {int(lab):3d}  {ia:.3f} -> {ca:.3f} ({ca - ia:+.3f})")
    verdict = "PASS - safe to ship" if res["ok"] else "FAIL - do NOT deploy"
    print(f"  verdict: {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True,
                    help="checkpoint to gate (path to a .pth state_dict)")
    ap.add_argument("--incumbent", default=str(CKPT),
                    help="baseline checkpoint (default: current runtime model)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"max allowed valid-acc drop (default {DEFAULT_TOL})")
    args = ap.parse_args(argv)

    res = gate(args.candidate, args.incumbent, args.tol)
    print_report(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())