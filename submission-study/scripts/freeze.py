"""Hash planned inputs and existing checkpoints without modifying them."""
from common import *
EXPERIMENT_FILES=['common.py','prepare.py','engine.py','run.py','batch_run.py','analyze_core.py','test_mathematics.py','external.py','heads.py']
def main():
    if (ROOT/'data/freeze.json').exists():raise RuntimeError('A freeze already exists; do not overwrite it')
    models={}
    for name,path in MODEL_PATHS.items():
        files={}
        for p in sorted(path.iterdir()):
            if not p.is_file() or not (p.suffix in ['.json','.safetensors','.txt','.model']):continue
            meta=path/'.cache/huggingface/download'/(p.name+'.metadata')
            revision=meta.read_text().splitlines()[0] if meta.exists() else None
            files[p.name]={'bytes':p.stat().st_size,'sha256':sha(p),'revision':revision}
        models[name]={'location_relative_to_workspace':str(path.relative_to(WORK)),'files':files}
        print('Hashed',name,flush=True)
    dump(ROOT/'data/checkpoints.json',models)
    paths=[ROOT/'scripts'/n for n in EXPERIMENT_FILES]+[ROOT/'docs/PROTOCOL.md']+list((ROOT/'data').glob('*.json'))
    paths += [ROOT/'vendor/blimp-relational-noun.jsonl']
    dump(ROOT/'data/freeze.json',{'created_utc':utc(),'scope':'Extension primary and planned supplementary scientific inputs; before test inference. Development outcomes were available.','files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}})
    print('Frozen experimental and analysis inputs',flush=True)
if __name__=='__main__':main()
