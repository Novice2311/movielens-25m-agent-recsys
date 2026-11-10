from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from .llm import JsonModelClient
from .observability import REQUESTS,REQUEST_TIME,REJECTS


class Plan(BaseModel):
    genres:list[str]=Field(default_factory=list,max_length=3)
    exclude:list[str]=Field(default_factory=list,max_length=5)
    needs_clarification:bool=False


class Pick(BaseModel):
    movieId:int
    evidence_genre:str


class Selection(BaseModel):
    items:list[Pick]=Field(max_length=5)


class Verdict(BaseModel):
    accepted:bool
    issue:str


class State(TypedDict,total=False):
    query:str
    user_id:int
    session_id:str
    k:int
    memory:dict
    decision:dict
    plan:dict
    candidates:list
    selected:list
    verification:dict
    attempt:int
    trajectory:list
    trace_id:str
    outcome:str
    error:str


ALIASES={
    'Sci-Fi':r'фантаст\w*|sci[ -]?fi|science fiction',
    'Comedy':r'комеди\w*|комедий\w*|comedy',
    'Horror':r'ужас\w*|хоррор\w*|horror',
    'Drama':r'драм\w*|drama',
    'Animation':r'мульт\w*|анимаци\w*|animation',
    'Action':r'боевик\w*|action',
    'Romance':r'романт\w*|romance',
    'Thriller':r'триллер\w*|thriller',
    'Documentary':r'документаль\w*|documentary',
}


def explicit_constraints(text):
    wanted=[];excluded=[];text=text.lower()
    for genre,pattern in ALIASES.items():
        for m in re.finditer(r'\b(?:'+pattern+r')\b',text):
            prefix=text[max(0,m.start()-20):m.start()]
            if re.search(r'\b(?:без|кроме|не хочу|исключи|no|without|exclude)\s*$',prefix):
                excluded.append(genre)
            else:wanted.append(genre)
    return sorted(set(wanted)-set(excluded)),sorted(set(excluded))


class MovieAgents:
    def __init__(self,catalog,memory,gate,obs,prompts):
        self.catalog,self.memory,self.gate,self.obs=catalog,memory,gate,obs
        self.prompts=Path(prompts)
        self.llm=JsonModelClient(obs.llm)
        self.small=os.getenv('ROUTER_MODEL','qwen3:0.6b')
        self.strong=os.getenv('WRITER_MODEL','qwen3.5:0.8b')
        self.validator_model=os.getenv('VALIDATOR_MODEL','gemma3:270m')
        if self.validator_model.split(':')[0].startswith('qwen'):
            raise ValueError('Validator must use a different model family')
        graph=StateGraph(State)
        for name in ['recall','decision','planner','retriever','selector','validator','fallback','finish']:
            graph.add_node(name,self.wrap(name,getattr(self,name)))
        graph.add_edge(START,'recall');graph.add_edge('recall','decision')
        graph.add_edge('decision','planner')
        graph.add_conditional_edges('planner',lambda s:'finish' if s['plan'].get('needs_clarification') else 'retriever')
        graph.add_edge('retriever','selector');graph.add_edge('selector','validator')
        graph.add_conditional_edges('validator',self.after_validation)
        graph.add_edge('fallback','finish');graph.add_edge('finish',END)
        self.graph=graph.compile()

    def wrap(self,name,function):
        def run(state):
            with self.obs.span(name,state):
                result=function(state)
            # Return fields mutated for trajectory plus explicit node updates.
            return {**state,**(result or {})}
        return run

    def prompt(self,name):
        root=self.prompts.parent
        role=(self.prompts/(name+'.md')).read_text(encoding='utf-8')
        if name=='validator':return role
        soul=(root/'agent/SOUL.md').read_text(encoding='utf-8')
        skill=(root/'skills/recommend-movies/SKILL.md').read_text(encoding='utf-8')
        return soul+'\n'+skill+'\n'+role

    def recall(self,s):return {'memory':self.memory.recall(s['user_id'])}
    def decision(self,s):return {'decision':self.gate.decide(s['query'])}

    def planner(self,s):
        requested,excluded=explicit_constraints(s['query'])
        remembered=s['memory']['facts']
        saved=[g for g in remembered.get('exclude_genres',{}).get('value',[]) if g not in requested]
        excluded=sorted(set(excluded+saved))
        chosen=self.small if s['decision']['choice']=='recommend' and not s['decision']['abstain'] else self.strong
        try:
            schema=Plan.model_json_schema()
            for field in ('genres','exclude'):
                schema['properties'][field]['items']['enum']=self.catalog.genre_names
            result=self.llm.json(chosen,self.prompt('planner'),
                   {'query':s['query'],'allowed_genres':self.catalog.genre_names,
                    'explicit_requested':requested,'explicit_excluded':excluded},schema)
            plan=Plan.model_validate(result).model_dump()
            plan['genres']=[g for g in plan['genres'] if g in self.catalog.genre_names]
            plan['exclude']=[g for g in plan['exclude'] if g in self.catalog.genre_names]
        except Exception as e:
            plan={'genres':requested,'exclude':excluded,'needs_clarification':len(s['query'].strip())<3}
            self.obs.log('planner_fallback',error=type(e).__name__)
        # Explicit constraints always win over model suggestions.
        if requested:plan['genres']=requested
        elif not plan['genres']:
            plan['genres']=remembered.get('prefer_genres',{}).get('value',[])
        plan['exclude']=sorted(set([x for x in plan['exclude'] if x not in requested]+excluded))
        plan['genres']=[x for x in plan['genres'] if x not in plan['exclude']]
        if requested:plan['needs_clarification']=False
        plan['planner_model']=chosen
        return {'plan':plan,'attempt':0}

    def retriever(self,s):
        episodes=s['memory']['episodes']
        recent_ids={i for ep in episodes if ep.get('catalog_version')=='ml25m-2017-train'
                    for i in ep.get('movie_ids',[])}
        candidates=self.catalog.retrieve(s['user_id'],s['plan']['genres'],s['plan']['exclude'],
                                         k=20,exclude_ids=recent_ids)
        if len(candidates)<s['k'] and recent_ids:
            # Episodic diversity is soft; hard genre constraints and seen history stay intact.
            candidates=self.catalog.retrieve(s['user_id'],s['plan']['genres'],s['plan']['exclude'],k=20)
        return {'candidates':candidates}

    def selector(self,s):
        if not s['candidates']:return {'selected':[],'error':'empty_catalog'}
        try:
            candidates=s['candidates'][:12]
            k=min(s['k'],len(candidates))
            schema=Selection.model_json_schema()
            props=schema['$defs']['Pick']['properties']
            props['movieId']['enum']=[m['movieId'] for m in candidates]
            props['evidence_genre']['enum']=sorted({g for m in candidates for g in m['genres']})
            schema['properties']['items'].update(minItems=k,maxItems=k)
            d=self.llm.json(self.strong,self.prompt('selector'),
                {'request':s['query'],'plan':s['plan'],'candidates':candidates,
                 'k':k,
                 'previous_errors':s.get('verification',{}).get('errors',[])},
                schema,max_tokens=300)
            selected=[x.model_dump() for x in Selection.model_validate(d).items]
            return {'selected':selected,'error':''}
        except Exception as e:return {'selected':[],'error':type(e).__name__}

    def validator(self,s):
        errors=self.catalog.verify(s['selected'],s['candidates'],s['plan'],s['k'])
        if s.get('error'):errors.append(s['error'])
        accepted=False;issue='deterministic_contract_failed';judge_called=False
        if not errors:
            try:
                by_id={m['movieId']:m for m in s['candidates']}
                chosen=[by_id[x['movieId']] for x in s['selected']]
                judge_called=True
                d=self.llm.json(self.validator_model,self.prompt('validator'),
                   {'plan':s['plan'],'selected':s['selected'],'candidates':chosen,
                    'k':min(s['k'],len(s['candidates']))},Verdict.model_json_schema(),max_tokens=100)
                verdict=Verdict.model_validate(d)
                accepted=verdict.accepted;issue=verdict.issue
                if not accepted:errors.append('llm_validator_rejected')
            except Exception as e:
                errors.append('validator_unavailable');issue=type(e).__name__
        for e in set(errors):REJECTS.labels(e).inc()
        return {'verification':{'accepted':bool(accepted and not errors),'errors':sorted(set(errors)),
                                'issue':issue,'model':self.validator_model if judge_called else None},
                'attempt':s['attempt']+1}

    def after_validation(self,s):
        if s['verification']['accepted']:return 'finish'
        if s['attempt']<2 and not s.get('error') and s['candidates']:return 'selector'
        return 'fallback'

    def fallback(self,s):
        selected=[{'movieId':m['movieId'],'evidence_genre':next(
                    (g for g in m['genres'] if g in s['plan']['genres']),m['genres'][0])}
                  for m in s.get('candidates',[])[:s['k']]]
        # Deterministic catalog output remains useful when the LLM layer is unavailable.
        errors=self.catalog.verify(selected,s.get('candidates',[]),s['plan'],s['k'])
        if errors:raise ValueError('Fallback violates catalog contract')
        return {'selected':selected,'outcome':'fallback'}

    def finish(self,s):
        if s['plan'].get('needs_clarification'):return {'outcome':'clarify','selected':[]}
        return {'outcome':s.get('outcome','verified')}

    def invoke(self,query,user_id,k=3,session_id='demo'):
        started=time.perf_counter()
        with self.obs.tracer.start_as_current_span('recommendation_request'):
            result=self.graph.invoke({'query':query,'user_id':user_id,'k':k,
                       'session_id':session_id,'trajectory':[]},config={'recursion_limit':15})
            if result['outcome']!='clarify':
                self.memory.remember_episode(user_id,session_id,[m['movieId'] for m in result['selected']],result['trace_id'])
            REQUESTS.labels(result['outcome']).inc()
            REQUEST_TIME.observe(time.perf_counter()-started)
            self.obs.save_trajectory(result)
        candidates={m['movieId']:m for m in result.get('candidates',[])}
        result['items']=[{**candidates[x['movieId']], 'evidence_genre':x['evidence_genre']}
                         for x in result.get('selected',[])]
        result['elapsed_seconds']=round(time.perf_counter()-started,3)
        return {k:result.get(k) for k in ['items','outcome','trace_id','trajectory','plan',
                                        'decision','verification','elapsed_seconds']}
