import argparse
import json
from pathlib import Path

from l0_core import _L0Model


def main():
    parser=argparse.ArgumentParser(description="Pure Python inference with the released L0 vocabulary/IDF/logistic coefficients")
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--text",required=True)
    args=parser.parse_args()
    spec=json.loads(args.model.read_text(encoding="utf-8"))
    model=_L0Model(spec)
    values=model.probs(args.text)
    print(json.dumps({"classes":model.classes,"probabilities":values,"prediction":model.classes[max(range(len(values)),key=values.__getitem__)],
                      "operating_point":model.operating_point,"permission":"prediction is not write authorization"},ensure_ascii=False))


if __name__=="__main__":
    main()
