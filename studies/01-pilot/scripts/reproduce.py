"""Create an isolated replay directory; never overwrite the published measurements."""
import argparse, datetime, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['analysis','data','full']);args=ap.parse_args()
if args.mode=='analysis':
    subprocess.run([sys.executable,str(ROOT/'scripts/audit.py'),'statistics'],check=True)
    raise SystemExit(0)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
parent=ROOT/'tmp/reproductions'/stamp;clone=parent/ROOT.name
clone.mkdir(parents=True)
for directory in ['scripts','docs']:
    shutil.copytree(ROOT/directory,clone/directory,ignore=shutil.ignore_patterns('__pycache__'))
import os; os.environ.setdefault('MODELS_DIR', str(ROOT.parents[1]/'models'))  # release: the copy finds the models through MODELS_DIR
def run(script,*arguments):subprocess.run([sys.executable,str(clone/'scripts'/script),*arguments],check=True)
if args.mode=='data':
    run('experiment.py','prepare')
    checks={p:hashlib.sha256((clone/'data'/p).read_bytes()).hexdigest()==hashlib.sha256((ROOT/'data'/p).read_bytes()).hexdigest() for p in ['calibration.json','evaluation.json','pools.json','model-manifest.json']}
    (clone/'data-regeneration.json').write_text(json.dumps(checks,indent=2)+'\n');assert all(checks.values())
else:
    shutil.copytree(ROOT/'data',clone/'data')
    for model in ['gpt2-small','gpt2-medium']:
        for stage in ['calibrate','evaluate','replay']:run('experiment.py',stage,'--model',model)
        run('audit.py','readout','--model',model)
    run('analyze.py')
print('Replay artifacts:',clone)
