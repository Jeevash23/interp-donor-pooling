"""Portable verification/statistics and isolated fresh-model replay launcher.

No frozen source is edited. Replay creates a complete local copy, preserves the
archived comparison measurements and writes only into that new copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def command(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True)


def clone(destination):
    if destination.exists():raise RuntimeError('Choose a fresh, nonexistent replay directory')
    if destination.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('Replay directory must be outside the source artifact tree')
    def ignore(directory,names):
        return [n for n in names if n in ['__pycache__','output','tmp'] or n.endswith(('.zip','.png','.aux','.log','.out','.fdb_latexmk','.fls'))]
    shutil.copytree(ROOT,destination,ignore=ignore)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['verify','analyze','token-replay','circuit-replay','_token-internal'])
    parser.add_argument('--model',choices=['gpt2-small','qwen3-0.6b'],default='gpt2-small')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.stage=='verify':
        command(ROOT/'scripts/verify_revision.py');return
    if args.stage=='analyze':
        command(ROOT/'token/analyze.py','--split','test')
        command(ROOT/'circuit/experiment.py','analyze')
        command(ROOT/'scripts/verify_revision.py');return
    if args.checkpoint is None:raise ValueError('--checkpoint is required for local inference')
    checkpoint=args.checkpoint.resolve()
    name='gpt2-small' if args.stage=='circuit-replay' else args.model
    expected=json.loads((ROOT/'results/environment-and-inputs.json').read_text())['models'][name]
    for filename,record in expected.items():
        with (checkpoint/filename).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=record['sha256']:raise ValueError(f'Checkpoint mismatch: {filename}')
    if args.stage=='_token-internal':
        sys.path.insert(0,str(ROOT/'token'))
        import common
        common.MODEL_PATHS={args.model:checkpoint}
        import run
        sys.argv=[str(ROOT/'token/run.py'),'replay','--model',args.model]
        run.main();return
    if args.output is None:raise ValueError('--output must name a fresh directory')
    target=clone(args.output.resolve())
    if args.stage=='token-replay':
        folder=target/'token/results'/args.model
        replay=folder/'replay'
        if replay.exists():shutil.rmtree(replay)
        for name in ['replay-verification.json','replay-completion.json','environment-replay.json']:
            (folder/name).unlink(missing_ok=True)
        command(target/'scripts/reproduce.py','_token-internal','--model',args.model,'--checkpoint',checkpoint)
    else:
        folder=target/'circuit/results'
        replay=folder/'replay'
        if replay.exists():shutil.rmtree(replay)
        for name in ['replay-verification.json','replay-completion.json']:
            (folder/name).unlink(missing_ok=True)
        command(target/'circuit/experiment.py','replay','--model-path',checkpoint)
    print(f'Replay and verification saved in {target}')


if __name__=='__main__':
    main()
