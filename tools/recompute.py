import argparse
import csv
import json
from pathlib import Path


def metrics(truth,predictions,labels):
    if len(truth)!=len(predictions):raise ValueError("Length mismatch")
    valid=[(gold,pred) for gold,pred in zip(truth,predictions) if gold in labels]
    confusion={gold:{pred:0 for pred in labels} for gold in labels}
    for gold,pred in valid:
        if pred in labels:confusion[gold][pred]+=1
    scores={}
    for label in labels:
        tp=sum(gold==label and pred==label for gold,pred in valid)
        fp=sum(gold!=label and pred==label for gold,pred in valid)
        fn=sum(gold==label and pred!=label for gold,pred in valid)
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        scores[label]={"tp":tp,"fp":fp,"fn":fn,"support":tp+fn,"precision":precision,"recall":recall,"f1":f1}
    return {"n":len(truth),"n_with_gold":len(valid),"missing_or_outside_label_gold":len(truth)-len(valid),
            "invalid_predictions":sum(pred not in labels for _,pred in valid),
            "correct":sum(gold==pred for gold,pred in valid),
            "accuracy":sum(gold==pred for gold,pred in valid)/len(valid) if valid else None,
            "macro_f1":sum(row["f1"] for row in scores.values())/len(labels),"per_class":scores,"confusion":confusion}


def selftest():
    labels=["a","b","c"]
    assert metrics(labels,labels,labels)["macro_f1"]==1.0
    assert metrics(labels,["b","c","a"],labels)["accuracy"]==0.0
    assert abs(metrics(labels,["b","b","b"],labels)["macro_f1"]-1/6)<1e-12
    assert metrics(["a",None],["invalid","a"],labels)["n_with_gold"]==1
    assert metrics(["a"],["invalid"],labels)["per_class"]["a"]["fn"]==1


def main():
    parser=argparse.ArgumentParser(description="Recompute row-level classification point metrics; not cluster bootstrap or root-weighted inference")
    parser.add_argument("--csv",type=Path)
    parser.add_argument("--jsonl",type=Path)
    parser.add_argument("--gold",default="gold")
    parser.add_argument("--prediction",required=False,default="prediction")
    parser.add_argument("--labels",default="equivalent,compatible,conflict")
    parser.add_argument("--domain")
    parser.add_argument("--selftest",action="store_true")
    args=parser.parse_args()
    selftest()
    if args.selftest:
        print(json.dumps({"selftest":"passed","controls":["perfect","cyclic-wrong","majority","missing-gold","invalid-prediction"]}))
        return
    if args.csv:
        with args.csv.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.DictReader(stream))
    elif args.jsonl:rows=[json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:raise ValueError("Pass --csv or --jsonl")
    if args.domain:rows=[row for row in rows if row.get("domain")==args.domain]
    labels=args.labels.split(",")
    result=metrics([row.get(args.gold) for row in rows],[row.get(args.prediction) for row in rows],labels)
    result["weighting"]="row-equal point metrics; use supplied historical analysis for cluster/root/bootstrap claims"
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
