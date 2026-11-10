"""Exercise memory, telemetry, and a real temporary failure of the model gateway."""
import json,subprocess,time,urllib.request,urllib.parse,xml.etree.ElementTree as ET
from pathlib import Path

HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))
APP='http://127.0.0.1:18080';PROM='http://127.0.0.1:18090';TEMPO='http://127.0.0.1:13200'
GATEWAY='http://127.0.0.1:11435'

def call(url,body=None,method=None,timeout=60):
    req=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None,
                               headers={'Content-Type':'application/json'},method=method)
    with HTTP.open(req,timeout=timeout) as r:return json.load(r)

def ask(query,user):return call(APP+'/api/chat',{'query':query,'user_id':user,'k':3,'session_id':'verification'},timeout=240)

def main():
    out={}
    unit=ET.parse('results/unit_tests.xml').getroot()
    suite=unit.find('testsuite') if unit.tag=='testsuites' else unit
    out['unit_passed']=int(suite.attrib['tests'])-sum(int(suite.attrib.get(k,0)) for k in ('errors','failures','skipped'))
    assert call(APP+'/health')['status']=='ok'
    assert call(GATEWAY+'/api/version')['engine']=='llama.cpp'
    user=999998
    call(APP+f'/api/memory/{user}',method='DELETE')
    try:
        pref=call(APP+'/api/preferences',{'user_id':user,'key':'exclude_genres','genres':['Horror']})
        first=ask('Посоветуй фантастику',user);second=ask('Посоветуй фантастику',user)
        ids1={m['movieId'] for m in first['items']};ids2={m['movieId'] for m in second['items']}
        out['explicit_memory_applied']=(pref['facts']['exclude_genres']['version']==1 and
            'Horror' in first['plan']['exclude'] and all('Horror' not in m['genres'] for m in first['items']))
        out['episodic_repeats_filtered']=len(ids1)==len(ids2)==3 and not bool(ids1&ids2)
    finally:
        out['test_memory_deleted']=call(APP+f'/api/memory/{user}',method='DELETE')['deleted']
    query=PROM+'/api/v1/query?'+urllib.parse.urlencode({'query':'up{job="movie-agent"}'})
    status=call(query)['data']['result'];out['prometheus_target_up']=bool(status and status[0]['value'][1]=='1')
    when=time.time()
    subprocess.run(['docker','compose','stop','llm-gateway'],check=True)
    try:
        fault=ask('Посоветуй фантастику без ужасов',999997)
        out['fallback_on_model_outage']=(fault['outcome']=='fallback' and len(fault['items'])==3 and
            all('Sci-Fi' in m['genres'] and 'Horror' not in m['genres'] for m in fault['items']))
        out['fault_trace_id']=fault['trace_id']
        out['fault_trajectory']=fault['trajectory']
        end=time.monotonic()+50
        out['alert_received']=False;out['trace_available_in_tempo']=False
        while time.monotonic()<end and not (out['alert_received'] and out['trace_available_in_tempo']):
            obs=call(APP+'/api/observability',timeout=5)
            out['alert_received']=any(a['received_at']>=when and a['payload']['status']=='firing' and
                 a['payload']['commonLabels'].get('alertname')=='MovieAgentLLMFailure' for a in obs['alerts'])
            try:
                trace=call(TEMPO+'/api/traces/'+fault['trace_id'],timeout=3)
                out['trace_available_in_tempo']=bool(trace.get('batches') or trace.get('resourceSpans'))
            except (OSError,ValueError):pass
            if not (out['alert_received'] and out['trace_available_in_tempo']):time.sleep(2)
    finally:
        subprocess.run(['docker','compose','start','llm-gateway'],check=True)
    out['models_restored']=False
    for _ in range(8):
        try:
            d=call(GATEWAY+'/api/chat',{'model':'qwen3:0.6b','stream':False,'format':{'type':'object',
               'properties':{'ok':{'type':'boolean'}},'required':['ok']},
               'messages':[{'role':'user','content':'Return JSON with ok true.'}],'options':{'num_predict':30}},timeout=20)
            out['models_restored']=json.loads(d['message']['content']).get('ok') is True
            if out['models_restored']:break
        except (OSError,ValueError,KeyError):time.sleep(2)
    Path('results/verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
    assert all(out[k] for k in ['explicit_memory_applied','episodic_repeats_filtered','prometheus_target_up',
           'fallback_on_model_outage','alert_received','trace_available_in_tempo','models_restored'])

if __name__=='__main__':main()
