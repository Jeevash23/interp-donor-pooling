"""Portable-copy CPU checks; intentionally stop before any model inference."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_launcher(path):
    spec=importlib.util.spec_from_file_location('portable_launcher_check',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def check_frozen(root):
    counts={}
    for panel in ['token','circuit']:
        entries=read(root/panel/'data/freeze.json')['files']
        for path,h in entries.items():assert sha(root/panel/path)==h,(panel,path)
        counts[panel]=len(entries)
    return counts

def main():
    checkpoint=Path(os.environ.get('MODELS_DIR', ROOT.parents[1]/'models'))/'gpt2-small'  # release: see models/README.md
    report={"created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            "no_model_inference_performed":True,"launcher_sha256":sha(ROOT/'scripts/reproduce.py')}
    with tempfile.TemporaryDirectory(prefix='pooling-arbitrary-copy-') as tmp:
        target=Path(tmp)/'unrelated-project-name'
        def ignore(directory,names):
            return [n for n in names if n in ['output','tmp','__pycache__'] or n.endswith(('.zip','.aux','.log','.out','.fdb_latexmk','.fls'))]
        shutil.copytree(ROOT,target,ignore=ignore)
        report['copied_frozen_files']=check_frozen(target)
        launcher=load_launcher(target/'scripts/reproduce.py')
        try:launcher.clone(target/'would-recurse')
        except RuntimeError:report['recursive_destination_rejected']=True
        else:raise AssertionError('recursive clone destination accepted')
        assert not (target/'would-recurse').exists()
        # Full checkpoint-independent statistics are tested for the completed
        # GPT-2 token panel and circuit panel. Qwen is still running while this
        # check is prepared; the final ZIP integration must cover both models.
        commands=[['token/analyze.py','--model','gpt2-small','--split','test'],
                  ['circuit/experiment.py','analyze']]
        for command in commands:
            subprocess.run([sys.executable,str(target/command[0]),*command[1:]],check=True,capture_output=True,text=True)
        before=read(ROOT/'token/results/analysis-test/gpt2-small.json')
        after=read(target/'token/results/analysis-test/gpt2-small.json')
        assert before==after
        before=read(ROOT/'circuit/results/analysis.json');after=read(target/'circuit/results/analysis.json')
        before.pop('created_utc');after.pop('created_utc');assert before==after
        report['gpt2_token_statistics_exact']=True
        report['circuit_statistics_exact_except_analysis_timestamp']=True
        # Exercise the real internal launcher and its frozen run/common/engine
        # imports, but stop at Engine construction before loading any model.
        sys.path.insert(0,str(target/'token'))
        import common
        import run
        import native_engine
        class BeforeInference(Exception):pass
        class StubEngine:
            def __init__(self,name,*args,**kwargs):
                assert name=='gpt2-small'
                assert common.MODEL_PATHS=={'gpt2-small':checkpoint.resolve()}
                # run/native_engine were deliberately imported above before
                # the launcher changes the dict; remove those modules below
                # and import them afresh to mimic the real launch sequence.
                raise BeforeInference
        # The launcher imports run only after replacing the configuration.
        # A minimal import hook wraps that fresh module's Engine, retaining
        # all real frozen imports and pre-Engine freeze/checkpoint checks.
        del sys.modules['run'];del sys.modules['native_engine']
        import builtins
        original_import=builtins.__import__
        def intercept(name,*args,**kwargs):
            mod=original_import(name,*args,**kwargs)
            if name=='run':
                actual_engine=sys.modules['native_engine']
                assert actual_engine.MODEL_PATHS=={'gpt2-small':checkpoint.resolve()}
                mod.Engine=StubEngine
            return mod
        builtins.__import__=intercept
        saved_argv=sys.argv[:]
        try:
            sys.argv=[str(target/'scripts/reproduce.py'),'_token-internal','--model','gpt2-small','--checkpoint',str(checkpoint)]
            try:launcher.main()
            except BeforeInference:report['selected_checkpoint_mapping_and_frozen_preflight_passed']=True
            else:raise AssertionError('Inference barrier was not reached')
        finally:
            builtins.__import__=original_import;sys.argv=saved_argv
        # Verify circuit replay dispatch and stale-record removal without
        # executing its GPU command. This is a dispatch test, not a replay.
        replay_target=Path(tmp)/'new-circuit-replay'
        def mocked_command(*args):
            args=list(map(str,args))
            assert args[1]=='replay' and args[2]=='--model-path'
            assert Path(args[3])==checkpoint.resolve()
            folder=replay_target/'circuit/results'
            assert not (folder/'replay').exists()
            assert not (folder/'replay-verification.json').exists()
            assert not (folder/'replay-completion.json').exists()
            report['circuit_replay_dispatch_and_stale_result_removal_passed']=True
        launcher.command=mocked_command
        try:
            sys.argv=[str(target/'scripts/reproduce.py'),'circuit-replay','--checkpoint',str(checkpoint),'--output',str(replay_target)]
            launcher.main()
        finally:sys.argv=saved_argv
        assert check_frozen(target)==report['copied_frozen_files']
        report['frozen_sources_unchanged_after_mapping_and_statistics']=True
    report['passed']=True
    report['remaining_integration_scope']='Final ZIP extraction must run complete two-model verify/analyze/verify without checkpoint paths or GPU inference.'
    (HERE/'results/portable-launcher-check.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
