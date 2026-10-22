"""Outcome-independent task generation and tokenizer/overlap checks."""
from __future__ import annotations
import random
from common import *
from transformers import AutoTokenizer

NAMES="Alice Bob Carol David Emma Frank Grace Henry Jack James Jane John Julia Kate Laura Lucy Luke Mark Mary Michael Mike Nick Oliver Paul Peter Rachel Richard Robert Rose Sam Sarah Simon Sophie Steve Susan Thomas Tom Victor William Anna Ben Daniel Edward George Helen Joe Kevin Linda Lisa Maria Martin Nancy Patrick Rebecca Roger Steven Tony Walter Wendy".split()
PLACES="store park garden school office station market hotel museum library beach church hospital theater restaurant".split()
OBJECTS="book ring ball cup bag box hat map toy key letter flower ticket bottle apple".split()
ADJS="young old happy quiet calm clever tired proud kind brave polite friendly busy careful patient curious tall short new clever eager gentle".split()
NOUNS="pilot guard teacher doctor farmer lawyer driver nurse singer artist baker dancer painter sailor worker actor student soldier chef player".split()
COLORS="red blue green yellow white black orange purple pink brown".split()
IOI=["When {first} and {second} went to the {place}, {sub} gave the {obj} to", "After {first} and {second} arrived at the {place}, {sub} handed the {obj} to", "At the {place}, {first} met with {second}. Later, {sub} passed the {obj} to", "During a trip to the {place}, {first} spoke with {second} before {sub} offered the {obj} to"]
SVA=["The {adj} {head} near the {distr}", "The {adj} {head} beside the {distr}", "The {adj} {head} behind the {distr}", "The {adj} {head} in front of the {distr}"]
KV=["At the {place}, the list reads: {pairs}. The color for {query} is", "In the {place}, the notes say: {pairs}. The color assigned to {query} is", "From the {place}, a record states: {pairs}. According to this record, {query} has the color", "A list from the {place} says: {pairs}. Looking at this list, the color of {query} is"]

def prompt(text, anchor, occurrence=-1):
    start=text.rfind(anchor) if occurrence==-1 else text.find(anchor)
    assert start>=0
    return {"text":text,"anchor_span":[start,start+len(anchor)]}

def make_group(task,rng,template,cls):
    if task=="ioi":
        io,sub,alt=rng.sample(NAMES,3); first,second=(io,sub) if cls==0 else (sub,io)
        def make(f,place,obj,corrupt=False):
            rep=alt if corrupt else sub
            return prompt(IOI[f].format(first=first,second=second,place=place,sub=rep,obj=obj),rep)
        contexts=rng.sample([(p,o) for p in PLACES for o in OBJECTS],17)
        pair=lambda i,broad: [make(rng.randrange(4) if broad else template,*contexts[i],corrupt=c) for c in [False,True]]
        # A paired donor uses exactly the same template and nuisance context.
        def paired(i,broad):
            f=rng.randrange(4) if broad else template
            return [make(f,*contexts[i],corrupt=c) for c in [False,True]]
        rec=[make(template,*contexts[0],corrupt=c) for c in [False,True]]
        narrow=[paired(i,False) for i in range(1,9)]; broad=[paired(i,True) for i in range(9,17)]
        good,bad=io,sub
    elif task=="sva":
        head=rng.choice(NOUNS); distractors=rng.sample([n for n in NOUNS if n!=head],17)
        contexts=list(zip(rng.sample(list(dict.fromkeys(ADJS)),17),distractors))
        def make(f,h,ctx,corrupt=False):
            adj,dist=ctx; plural=bool(cls)^bool(corrupt)
            word=h+('s' if plural else '')
            dw=dist+('s' if cls==0 else '')
            return prompt(SVA[f].format(adj=adj,head=word,distr=dw),word,0)
        rec=[make(template,head,contexts[0],c) for c in [False,True]]
        narrow=[[make(template,head,contexts[i],c) for c in [False,True]] for i in range(1,9)]
        broad=[]
        for i in range(9,17):
            f=rng.randrange(4); h=rng.choice([n for n in NOUNS if n!=contexts[i][1]])
            broad.append([make(f,h,contexts[i],c) for c in [False,True]])
        good,bad=("is","are") if cls==0 else ("are","is")
    else:
        keys=rng.sample(NAMES,4); cols=rng.sample(COLORS,4); query=keys[cls]
        good=cols[cls]; other=(cls+1)%4; bad=cols[other]
        # All banks retain query/answer/contrast, while irrelevant values vary.
        def pair(f,place,order,values):
            ans=[]
            for corrupt in [False,True]:
                vv=values.copy()
                if corrupt:vv[cls],vv[other]=vv[other],vv[cls]
                ps="; ".join(f"{keys[k]}: {vv[k]}" for k in order)
                text=KV[f].format(place=place,pairs=ps,query=query)
                loc=text.index(f"{query}: ")+len(query)+2
                ans.append({"text":text,"anchor_span":[loc,loc+len(vv[cls])]})
            return ans
        order=list(range(4));rng.shuffle(order)
        rec=pair(template,rng.choice(PLACES),order,cols)
        narrow=[];broad=[]
        used={rec[0]["text"]}
        for kind in [narrow,broad]:
            while len(kind)<8:
                f=template if kind is narrow else rng.randrange(4)
                ord2=order.copy()
                if kind is broad:rng.shuffle(ord2)
                vals=cols.copy(); rest=rng.sample([c for c in COLORS if c not in [good,bad]],2)
                for k,v in zip([k for k in range(4) if k not in [cls,other]],rest):vals[k]=v
                p=pair(f,rng.choice(PLACES),ord2,vals)
                if p[0]["text"] in used:continue
                used.add(p[0]["text"]);kind.append(p)
    return {"task":task,"template":template,"class":cls,"good":good,"bad":bad,"recipient":rec,"narrow":narrow,"broad":broad}

def all_prompts(row):return [*row["recipient"],*[p for k in ["narrow","broad"] for pair in row[k] for p in pair]]

def encode_prompt(tok,config,p,good,bad):
    text=p["text"]
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc["input_ids"]; offsets=enc["offset_mapping"]
    target_ids=[]
    for word in [good,bad]:
        joined=tok.encode(text+" "+word,add_special_tokens=False)
        if len(joined)!=len(ids)+1 or joined[:-1]!=ids:raise ValueError("continuation not one appended token")
        target_ids.append(joined[-1])
    s,e=p["anchor_span"]
    candidates=[i for i,(a,b) in enumerate(offsets) if b> s and a<e]
    if not candidates:raise ValueError("anchor has no token")
    anchor=candidates[-1]
    if not (offsets[anchor][0]<e<=offsets[anchor][1]):raise ValueError("anchor endpoint mismatch")
    bos=config.get("bos_token_id")
    if bos is None:bos=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return {**p,"ids":[bos]+ids,"anchor":anchor+1,"end":len(ids),"good_id":target_ids[0],"bad_id":target_ids[1]}

def main():
    tokenizers={m:AutoTokenizer.from_pretrained(p,local_files_only=True) for m,p in MODEL_PATHS.items()}
    configs={m:read(p/"config.json") for m,p in MODEL_PATHS.items()}
    previous=set()
    for f in [ROOT.parent/"data/calibration.json",ROOT.parent/"data/evaluation.json"]:
        for row in read(f):previous.update(p["text"] for p in [row["clean"],row["corrupt"],*row["donors"]])
    used=set(previous); all_rows={}; rejected={}; pool_by_split={}
    for split,n,seed in [("calibration",4,26091201),("development",16,26091202),("test",96,26091203)]:
        rows=[]; local=set(); rej={"overlap":0,"tokenization":0}
        for ti,task in enumerate(TASKS):
            for g in range(n):
                seedblock=g//32 if split=="test" else 0
                rng=random.Random(seed+10000*ti+100*seedblock+g)
                template=g%(4 if split=="test" else 2); cls=(g//(4 if split=="test" else 2))%2
                for attempt in range(5000):
                    row=make_group(task,rng,template,cls)
                    ps=all_prompts(row); texts={p["text"] for p in ps}
                    if len(texts)!=len(ps) or texts&used or texts&local:
                        rej["overlap"]+=1;continue
                    try:
                        encoded={m:[encode_prompt(tok,configs[m],p,row["good"],row["bad"]) for p in ps] for m,tok in tokenizers.items()}
                        for m,ep in encoded.items():
                            if len({(p["good_id"],p["bad_id"]) for p in ep})!=1:raise ValueError("bank target token changes")
                    except ValueError:
                        rej["tokenization"]+=1;continue
                    row.update(id=f"{split}-{task}-{g:03d}",split=split,group=g,seed_block=seedblock,heldout_template=template>=2,encoded=encoded)
                    rows.append(row);local.update(texts);break
                else:raise RuntimeError(f"Failed generation for {split} {task} {g}")
        all_rows[split]=rows;rejected[split]=rej;pool_by_split[split]=local;used.update(local)
        dump(ROOT/f"data/{split}.json",rows)
        print(f"Prepared {split}: {len(rows)} groups; rejected {rej}",flush=True)
    dump(ROOT/"data/generation-audit.json",{"created_utc":utc(),"counts":{s:len(r) for s,r in all_rows.items()},"rejections":rejected,"unique_strings":{s:len(v) for s,v in pool_by_split.items()},"cross_split_overlap":{a+"/"+b:len(pool_by_split[a]&pool_by_split[b]) for a in pool_by_split for b in pool_by_split if a<b},"pilot_overlap":{s:len(v&previous) for s,v in pool_by_split.items()},"tokenizers":list(tokenizers),"all_targets_single_appended_token":True})

if __name__=="__main__":main()
