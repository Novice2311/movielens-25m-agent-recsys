"""Fixed allowlist gateway: Ollama-compatible JSON protocol over llama.cpp.

The engine is llama.cpp, not an Ollama server. No arbitrary backend URLs accepted.
Separate resident model servers avoid reload latency on every family switch.
"""
import json
import time
import urllib.request
from pathlib import Path
from fastapi import FastAPI,HTTPException

app=FastAPI(title='Fixed local LLM router')
LOCAL_HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))
TARGETS={'qwen3:0.6b':'http://qwen-small:8080',
         'qwen3.5:0.8b':'http://qwen-modern:8080',
         'gemma3:270m':'http://gemma-validator:8080'}

@app.get('/api/tags')
def tags():return json.loads(Path('/models/models.json').read_text())

@app.get('/api/version')
def version():return {'engine':'llama.cpp','protocol':'ollama-compatible','resident_models':3}

@app.post('/api/chat')
def chat(request:dict):
    model=request.get('model')
    if model not in TARGETS:raise HTTPException(422,'Model is not allowlisted')
    options=request.get('options',{})
    payload={'model':model,'messages':request.get('messages',[]),'stream':False,
             'temperature':options.get('temperature',0),'seed':options.get('seed',20261111),
             'max_tokens':max(1,min(int(options.get('num_predict',220)),400)),
             'chat_template_kwargs':{'enable_thinking':False}}
    schema=request.get('format')
    if isinstance(schema,dict):
        payload['response_format']={'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':schema}}
    elif schema=='json':payload['response_format']={'type':'json_object'}
    req=urllib.request.Request(TARGETS[model]+'/v1/chat/completions',data=json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    started=time.perf_counter()
    try:
        with LOCAL_HTTP.open(req,timeout=100) as r:d=json.load(r)
        choice=d['choices'][0]
        return {'model':model,'message':choice['message'],'done':True,
                'done_reason':'length' if choice.get('finish_reason')=='length' else 'stop',
                'prompt_eval_count':d.get('usage',{}).get('prompt_tokens',0),
                'eval_count':d.get('usage',{}).get('completion_tokens',0),
                'total_duration':int((time.perf_counter()-started)*1e9),'load_duration':0}
    except Exception as e:raise HTTPException(503,'Local model unavailable: '+type(e).__name__)
