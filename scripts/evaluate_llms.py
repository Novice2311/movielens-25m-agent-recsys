"""Identical, deterministic genre extraction tasks for three local LLMs."""
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
LOCAL_HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

MODELS=['qwen3:0.6b','qwen3.5:0.8b','gemma3:270m']
CASES=[('I want a comedy','Comedy'),('Хочу фантастику','Sci-Fi'),
       ('Покажи ужасы','Horror'),('Recommend a drama','Drama'),
       ('Хочу мультфильм','Animation'),('Action movie please','Action'),
       ("I'd like romance",'Romance'),('Документальное кино','Documentary')]
SCHEMA={'type':'object','properties':{'genre':{'type':'string','enum':[y for _,y in CASES]}},
        'required':['genre'],'additionalProperties':False}

def main():
    result={'protocol':{'tasks':len(CASES),'temperature':0,'seed':20261111,
                        'num_ctx':4096,'num_predict':40,
                        'description':'Authored bilingual genre extraction; same schema and tasks for all models.'},'models':{}}
    for model in MODELS:
        rows=[]
        for query,expected in CASES:
            payload={'model':model,'stream':False,'format':SCHEMA,
                     'messages':[{'role':'system','content':'Extract the requested film genre. Return only JSON following the schema.'},
                                 {'role':'user','content':query}],
                     'options':{'temperature':0,'seed':20261111,'num_ctx':4096,'num_predict':40},'keep_alive':'10m'}
            if model.startswith('qwen3'):payload['think']=False
            started=time.perf_counter();answer=None;valid=False;error=None;meta={}
            try:
                req=urllib.request.Request('http://127.0.0.1:11435/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with LOCAL_HTTP.open(req,timeout=120) as r:meta=json.load(r)
                raw=meta['message']['content'];d=json.loads(raw)
                valid=(isinstance(d,dict) and set(d)=={'genre'}
                       and d.get('genre') in SCHEMA['properties']['genre']['enum']
                       and meta.get('done_reason')!='length')
                answer=d.get('genre')
            except Exception as e:error=type(e).__name__
            row={'query':query,'expected':expected,'answer':answer,'valid_json':valid,
                 'correct':bool(valid and answer==expected),'seconds':time.perf_counter()-started,
                 'load_seconds':meta.get('load_duration',0)/1e9,
                 'input_tokens':meta.get('prompt_eval_count',0),'output_tokens':meta.get('eval_count',0),'error':error}
            rows.append(row);print(model,json.dumps(row,ensure_ascii=False),flush=True)
        times=[r['seconds'] for r in rows]
        result['models'][model]={'n':len(rows),'correct':sum(r['correct'] for r in rows),
              'valid_json':sum(r['valid_json'] for r in rows),'median_seconds':float(np.median(times)),
              'p95_seconds':float(np.quantile(times,.95)),
              'warm_median_seconds':float(np.median([r['seconds'] for r in rows if r['load_seconds']<.5])) if any(r['load_seconds']<.5 for r in rows) else None,
              'cases':rows}
        Path('results/llm_eval.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with LOCAL_HTTP.open('http://127.0.0.1:11435/api/tags',timeout=10) as r:tags=json.load(r)
    Path('results/llm_models.json').write_text(json.dumps(tags,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
