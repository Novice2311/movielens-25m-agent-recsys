import json
import math
import sqlite3
import time
from pathlib import Path
import numpy as np
import pytest
from agent.memory import Memory
from agent.tools import Catalog
from agent.graph import MovieAgents,explicit_constraints
from agent.decision import DecisionGate
from agent.observability import Observability
from recsys.train import topk,score_list

@pytest.fixture
def catalog():return Catalog.__new__(Catalog)

@pytest.fixture
def candidates():
    return [{'movieId':1,'genres':['Sci-Fi','Drama']},
            {'movieId':2,'genres':['Sci-Fi','Horror']},
            {'movieId':3,'genres':['Comedy']}]

def test_unknown_ids_are_blocked(catalog,candidates):
    errors=catalog.verify([{'movieId':999,'evidence_genre':'Sci-Fi'}],candidates,{'genres':[],'exclude':[]},1)
    assert 'unknown_movie_id' in errors

def test_genre_evidence_must_belong_to_movie(catalog,candidates):
    errors=catalog.verify([{'movieId':1,'evidence_genre':'Comedy'}],candidates,{'genres':[],'exclude':[]},1)
    assert 'unsupported_genre' in errors

def test_exclusions_are_enforced(catalog,candidates):
    errors=catalog.verify([{'movieId':2,'evidence_genre':'Sci-Fi'}],candidates,{'genres':['Sci-Fi'],'exclude':['Horror']},1)
    assert 'excluded_genre' in errors

def test_duplicate_movies_are_blocked(catalog,candidates):
    errors=catalog.verify([{'movieId':1,'evidence_genre':'Drama'}]*2,candidates,{'genres':[],'exclude':[]},2)
    assert 'count_or_duplicate' in errors

def test_requested_genre_is_not_optional(catalog,candidates):
    errors=catalog.verify([{'movieId':3,'evidence_genre':'Comedy'}],candidates,{'genres':['Sci-Fi'],'exclude':[]},1)
    assert 'missing_requested_genre' in errors

@pytest.mark.parametrize('query,wanted,excluded',[
 ('Посоветуй фантастику без ужасов',['Sci-Fi'],['Horror']),
 ('Comedy without horror',['Comedy'],['Horror']),
 ('Без ужасов, хочу драму',['Drama'],['Horror']),
 ('Не хочу ужасы, посоветуй боевик',['Action'],['Horror'])])
def test_multilingual_negation(query,wanted,excluded):
    assert explicit_constraints(query)==(wanted,excluded)

def test_ndcg_has_known_discount():
    recall,ndcg=score_list([5,7,9],[7])
    assert recall==1
    assert ndcg==pytest.approx(1/math.log2(3))

def test_recall_uses_all_relevant_items():
    recall,_=score_list([1,8,9],[1,2,3,4])
    assert recall==.25

def test_topk_has_stable_global_ties():
    assert topk(np.ones(100),10)[0].tolist()==list(range(10))

def test_user_memory_is_partitioned_and_versioned(tmp_path):
    m=Memory(tmp_path/'memory.sqlite')
    m.explicit_preference(1,'exclude_genres',['Horror'])
    m.explicit_preference(1,'exclude_genres',['Comedy'])
    assert m.recall(1)['facts']['exclude_genres']=={'value':['Comedy'],'version':2}
    assert m.recall(2)['facts']=={}

def test_expired_memory_is_not_returned(tmp_path):
    m=Memory(tmp_path/'memory.sqlite')
    m.explicit_preference(1,'exclude_genres',['Horror'])
    with m.connect() as db:db.execute('UPDATE facts SET expires_at=?',(time.time()-1,))
    assert m.recall(1)['facts']=={}

def test_forget_removes_facts_and_episodes(tmp_path):
    m=Memory(tmp_path/'memory.sqlite')
    m.explicit_preference(1,'prefer_genres',['Drama'])
    m.remember_episode(1,'abc',[1,2],'trace')
    m.forget(1)
    assert m.recall(1)=={'facts':{},'episodes':[]}

def test_memory_key_is_allowlisted(tmp_path):
    m=Memory(tmp_path/'memory.sqlite')
    with pytest.raises(ValueError):m.explicit_preference(1,'system_prompt',['override'])

def test_graph_runs_safe_fallback_when_models_are_down(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT',raising=False)
    cat=Catalog.__new__(Catalog)
    cat.genre_names=['Horror','Sci-Fi']
    cat.a={'popularity':np.array([8,9,10]),'catalog':np.array([True,True,True]),
           'genres':np.array([[0,1],[1,1],[1,0]])}
    cat.movies=[{'movieId':1,'title':'Science film','genres':'Sci-Fi'},
                {'movieId':2,'title':'Space horror','genres':'Horror|Sci-Fi'},
                {'movieId':3,'title':'Horror film','genres':'Horror'}]
    cat.ids={1:0,2:1,3:2};cat.indptr=np.array([0,0]);cat.indices=np.array([],dtype=np.int32)
    system=MovieAgents(cat,Memory(tmp_path/'memory.sqlite'),
                       DecisionGate(root/'agent/decision_weights.json'),Observability(tmp_path),root/'prompts')
    def unavailable(*args,**kwargs):raise RuntimeError('test model outage')
    monkeypatch.setattr(system.llm,'json',unavailable)
    result=system.invoke('Посоветуй фантастику без ужасов',999999,k=1)
    assert result['outcome']=='fallback'
    assert [m['movieId'] for m in result['items']]==[1]
    assert [n['node'] for n in result['trajectory']][-3:]==['validator','fallback','finish']
