import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI,HTTPException,Response,Query as QueryParam
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field
from prometheus_client import generate_latest,CONTENT_TYPE_LATEST

from agent.decision import DecisionGate
from agent.graph import MovieAgents
from agent.memory import Memory
from agent.observability import Observability,RETRIEVAL_TIME,LLM_CALLS,REQUESTS
from agent.tools import Catalog

ROOT=Path(__file__).parent
STATE=Path(os.getenv('STATE_DIR',str(ROOT/'state')))
catalog=Catalog(os.getenv('ARTIFACT_DIR',str(ROOT/'artifacts')))
memory=Memory(STATE/'memory.sqlite')
obs=Observability(STATE)
agents=MovieAgents(catalog,memory,DecisionGate(ROOT/'agent/decision_weights.json'),obs,ROOT/'prompts')
for model in (agents.small,agents.strong,agents.validator_model):
    for status in ('ok','error'):LLM_CALLS.labels(model,status)
for outcome in ('verified','fallback','clarify'):REQUESTS.labels(outcome)
app=FastAPI(title='MovieLens 25M recommendation agents',version='1.0.0')
lock=threading.Lock()

class Query(BaseModel):
    query:str=Field(min_length=1,max_length=500)
    user_id:int=Field(default=1,ge=1,le=1000000)
    k:int=Field(default=3,ge=1,le=5)
    session_id:str=Field(default='demo',pattern=r'^[A-Za-z0-9_-]{1,40}$')

class Preference(BaseModel):
    user_id:int=Field(ge=1,le=1000000)
    key:str=Field(pattern=r'^(exclude_genres|prefer_genres)$')
    genres:list[str]=Field(max_length=5)

@app.get('/',response_class=HTMLResponse)
def index():return (ROOT/'web/index.html').read_text(encoding='utf-8')

@app.get('/health')
def health():return {'status':'ok','catalog':'MovieLens 25M','model':'TwoTower32','catalog_items':int(catalog.a['catalog'].sum())}

@app.post('/api/chat')
def chat(q:Query):
    # Bounded concurrency prevents simultaneous resident LLM contexts exhausting VRAM.
    if not lock.acquire(blocking=False):raise HTTPException(429,'One request is already running')
    try:return agents.invoke(q.query,q.user_id,q.k,q.session_id)
    finally:lock.release()

@app.get('/api/recommendations')
def direct_recommendations(user_id:int=QueryParam(default=1,ge=1,le=1000000),
                           k:int=QueryParam(default=10,ge=1,le=20),
                           genre:str|None=None,exclude:str|None=None):
    """Fast retrieval path. LLM interpretation is optional, outside this path."""
    started=time.perf_counter()
    saved=memory.recall(user_id)['facts']
    wanted=[genre] if genre else saved.get('prefer_genres',{}).get('value',[])
    if genre and genre==exclude:raise HTTPException(422,'Conflicting genre filters')
    excluded=[g for g in saved.get('exclude_genres',{}).get('value',[]) if g not in wanted]
    if exclude:excluded.append(exclude)
    try:
        items=catalog.retrieve(user_id,wanted,excluded,k)
        return {'items':items,'method':items[0]['basis'] if items else 'empty'}
    except ValueError as e:raise HTTPException(422,str(e))
    finally:RETRIEVAL_TIME.observe(time.perf_counter()-started)

@app.post('/api/preferences')
def preferences(p:Preference):
    if not set(p.genres).issubset(catalog.genre_names):raise HTTPException(422,'Unknown genre')
    memory.explicit_preference(p.user_id,p.key,p.genres)
    return memory.recall(p.user_id)

@app.delete('/api/memory/{user_id}')
def forget(user_id:int):
    memory.forget(user_id);return {'deleted':True}

@app.get('/api/observability')
def observability():
    return {'trajectories':obs.recent(10),'alerts':memory.recent_alerts(),
            'models':{'router':agents.small,'writer':agents.strong,'validator':agents.validator_model},
            'metrics':read_result('recsys_metrics.json'),'llm_eval':read_result('llm_eval.json'),
            'agent_eval':read_result('agent_eval.json'),'decision_eval':read_result('decision_eval.json')}

def read_result(name):
    p=ROOT/'results'/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

@app.get('/metrics')
def metrics():return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)

@app.post('/alerts')
def alerts(payload:dict):
    memory.alert(payload);return {'stored':True}
