import argparse
import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block:=stream.read(4*1024**2):value.update(block)
    return value.hexdigest()


def extract(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name=PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename or ":" in info.filename or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError("Unsafe release archive member")
            if info.is_dir():continue
            target=destination/Path(*name.parts)
            if target.exists():
                with bundle.open(info) as stream:
                    expected=hashlib.file_digest(stream,"sha256").hexdigest()
                if sha(target)!=expected:raise ValueError("Existing file differs; use a fresh destination: "+str(target))
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.open(info) as source,target.open("wb") as output:
                shutil.copyfileobj(source,output,4*1024**2)


def obtain(asset,destination):
    release=json.loads((ROOT/"assets.json").read_text(encoding="utf-8"))
    entry=release["assets"][asset]
    download=ROOT/".downloads"/asset
    download.parent.mkdir(exist_ok=True)
    if not download.is_file():
        temporary=download.with_suffix(download.suffix+".part")
        request=urllib.request.Request(entry["url"],headers={"User-Agent":"BioData-Model-Lab/1.0"})
        with urllib.request.urlopen(request,timeout=90) as response,temporary.open("wb") as output:
            shutil.copyfileobj(response,output,4*1024**2)
        temporary.replace(download)
    if sha(download)!=entry["sha256"]:raise ValueError("Release asset SHA256 mismatch")
    extract(download,destination)
    download.unlink()
    print(json.dumps({"asset":asset,"destination":str(destination),"sha256_verified":True},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description="Fetch one explicit research asset; no all-model download by default")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument("--base")
    group.add_argument("--source-data",action="store_true")
    group.add_argument("--list",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--upstream",action="store_true")
    args=parser.parse_args()
    catalog=json.loads((ROOT/"models/catalog.json").read_text(encoding="utf-8"))
    if args.list:
        print(json.dumps({"aliases":catalog["aliases"],"artifacts":[{"id":row["id"],"family":row["family"],"bytes":row.get("asset_bytes")} for row in catalog["artifacts"]]},ensure_ascii=False,indent=2))
    elif args.source_data:
        obtain("source-data.zip",args.output or ROOT)
    elif args.model:
        identifier=catalog["aliases"].get(args.model,args.model)
        row=next(row for row in catalog["artifacts"] if row["id"]==identifier)
        if row["format"]=="l0-json":
            print(ROOT/row["path"])
        else:obtain(row["asset"],args.output or ROOT/".cache/models"/identifier)
    else:
        row=next((row for row in catalog["bases"] if row["id"]==args.base),None)
        destination=args.output or ROOT/".cache/bases"/args.base.replace("/","--")
        if row is not None:obtain(row["asset"],destination)
        else:
            if not args.upstream:raise ValueError("This base is official-source-only; add --upstream for an explicit network download")
            from huggingface_hub import snapshot_download
            models=json.loads((ROOT/"provenance/base-models.json").read_text(encoding="utf-8"))["models"]
            base=next(row for row in models if row["id"]==args.base)
            revision=base["historical_revision"] or base["upstream_revision_observed_at_publication"]
            snapshot_download(args.base,revision=revision,local_dir=destination,token=False,
                              allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.bpe"])
            print(json.dumps({"base":args.base,"revision":revision,"historical_pin_recorded":base["historical_revision"] is not None,"destination":str(destination)},ensure_ascii=False))


if __name__=="__main__":
    main()
