import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn


class Classifier(nn.Module):
    def __init__(self, backbone, hidden, labels, pooling):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden, labels)
        self.pooling = pooling

    def forward(self, **batch):
        hidden = self.backbone(**batch).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.pooling == "mean":
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            indices = batch["attention_mask"].long().sum(1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
        return self.classifier(pooled.to(self.classifier.weight.dtype))


def load_model(directory, base, device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    kind = meta["format"]
    if kind == "ltr":
        model = nn.Sequential(nn.Linear(9,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
        model.load_state_dict(load_file(str(directory/"weights.safetensors")),strict=True)
        return meta, model.to(device).eval(), None
    if kind == "finite":
        raise ValueError("Finite-set models use the supplied head-design study engine, not the text encoder interface")
    if kind == "sklearn":
        raise ValueError("Use the archived features.load_scorer pipeline and verified joblib artifact; this entrypoint is tensor-only")
    if base is None:
        raise ValueError("Pass --base pointing to the locally fetched official tokenizer/config (and base weights for adapters/head-only models)")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if kind == "routing-adapter":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        backbone = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False, dtype=torch.float32)
        model = PeftModel.from_pretrained(backbone, directory/"adapter", local_files_only=True)
    elif kind == "qwen-head":
        from peft import PeftModel
        backbone = AutoModel.from_pretrained(base, local_files_only=True, trust_remote_code=False, dtype=torch.float32)
        backbone.config.use_cache=False
        backbone = PeftModel.from_pretrained(backbone,directory/"adapter",local_files_only=True)
        state = load_file(str(directory/"head.safetensors"))
        model = Classifier(backbone,state["weight"].shape[1],state["weight"].shape[0],meta["pooling"])
        model.classifier.load_state_dict(state,strict=True)
    elif kind == "encoder":
        state=load_file(str(directory/"weights.safetensors"))
        config=AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        kwargs={"add_pooling_layer":meta["add_pooling_layer"]} if config.model_type in {"bert","roberta","xlm-roberta"} else {}
        if meta["weights_scope"]=="head_only":
            backbone=AutoModel.from_pretrained(base,local_files_only=True,trust_remote_code=False,**kwargs)
        else:
            backbone=AutoModel.from_config(config,trust_remote_code=False,**kwargs)
        model=Classifier(backbone,int(config.hidden_size),len(meta["classes"]),meta["pooling"])
        missing,unexpected=model.load_state_dict(state,strict=False)
        allowed_missing=meta["weights_scope"]=="head_only" and all(key.startswith("backbone.") for key in missing)
        if unexpected or missing and not allowed_missing:
            raise ValueError("Released tensor architecture mismatch")
    else:
        raise ValueError("Unsupported released model format")
    return meta,model.to(device).eval(),tokenizer


def predict(directory,base,rows,*,device="cpu",task="binary",temperature=1.0):
    meta,model,tokenizer=load_model(directory,base,device)
    results=[]
    for row in rows:
        with torch.inference_mode():
            if meta["format"]=="ltr":
                spec=json.loads((directory/"ltr-spec.json").read_text(encoding="utf-8"))
                values=row["features"]
                x=torch.tensor([values],dtype=torch.float32,device=device)
                if list(x.shape)!=[1,9]:raise ValueError("LTR requires nine features in ltr-spec order")
                mean=torch.tensor(spec["feature_mean"],device=device)
                std=torch.tensor(spec["feature_std"],device=device)
                result={"id":row.get("id"),"score":float(model((x-mean)/std).reshape(-1)[0])}
            elif meta["format"]=="routing-adapter":
                spec=json.loads((directory/"routing-spec.json").read_text(encoding="utf-8"))
                config=spec["tasks"][task]
                messages=[{"role":"system","content":spec["system"]},{"role":"user","content":config["prompt"].format(utt=row["text"])}]
                options={"tokenize":False,"add_generation_prompt":True}
                if "Qwen3" in meta["base_model"]:options["enable_thinking"]=False
                text=tokenizer.apply_chat_template(messages,**options)
                batch=tokenizer(text,return_tensors="pt",truncation=False).to(device)
                ids=[config["option_token_ids"][name] for name in config["options"]]
                logits=model(**batch).logits[0,-1,ids].float()
                probs=torch.softmax(logits/float(config.get("temperature",1.0)),-1).cpu().tolist()
                result={"id":row.get("id"),"classes":config["options"],"probabilities":probs,"prediction":config["options"][max(range(len(probs)),key=probs.__getitem__)]}
            else:
                if "input_ids" in row:
                    batch={name:torch.tensor([row[name]],dtype=torch.long,device=device) for name in ("input_ids","attention_mask","token_type_ids") if name in row}
                    if "attention_mask" not in batch:batch["attention_mask"]=torch.ones_like(batch["input_ids"])
                else:
                    text=row.get("visible_text",row.get("text"))
                    if not isinstance(text,str):raise ValueError("Provide frozen visible_text or encoded input_ids, not an unformatted old/new object")
                    batch=tokenizer(text,return_tensors="pt",truncation=False).to(device)
                logits=model(**batch).float()[0]
                probabilities=torch.softmax(logits/temperature,-1).cpu().tolist()
                result={"id":row.get("id"),"classes":meta["classes"],"probabilities":probabilities,"prediction":meta["classes"][max(range(len(probabilities)),key=probabilities.__getitem__)],"score_kind":"softmax_uncalibrated_unless_explicit_temperature"}
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser(description="Local inference on released tensor artifacts; never downloads models or performs edits")
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--base",type=Path)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--task",default="binary")
    parser.add_argument("--temperature",type=float,default=1.0)
    args=parser.parse_args()
    if args.temperature<=0:raise ValueError("Temperature must be positive")
    rows=[json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in predict(args.model_dir,args.base,rows,device=args.device,task=args.task,temperature=args.temperature):
        print(json.dumps(row,ensure_ascii=False))


if __name__=="__main__":
    main()
