import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"source/src"))


def main():
    from transformers import AutoTokenizer
    from dataset_recommender.agent.semantic_relations import prepare_semantic_input

    parser=argparse.ArgumentParser(description="Reuse the released V3 tree formatter; no label/ID is inserted into model text")
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-length",type=int,default=384)
    parser.add_argument("--direction",choices=("forward","reverse"),default="forward")
    parser.add_argument("--role-independent-frame",action="store_true")
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.base,local_files_only=True,token=False,trust_remote_code=False)
    outputs=[]
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        row=json.loads(line)
        prepared=prepare_semantic_input(row.get("pair",row),tokenizer,max_length=args.max_length,
                                       role_independent_frame=args.role_independent_frame)
        view=prepared["directions"][args.direction]
        outputs.append({"id":row.get("id"),"visible_text":view["text"],**view["encoding"],
                        "complete":prepared["complete"],"recipe":prepared["recipe"],
                        "requires_llm_fallback":prepared["requires_llm_fallback"]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in outputs)+"\n",encoding="utf-8")
    print(json.dumps({"rows":len(outputs),"complete":sum(row["complete"] for row in outputs),
                      "note":"V3 formatter only; legacy seven-class/V2 use their frozen formatter or archived encodings"},ensure_ascii=False))


if __name__=="__main__":
    main()
