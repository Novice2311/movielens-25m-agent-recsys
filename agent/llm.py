import json
import os
import time
import urllib.error
import urllib.request

LOCAL_HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))


class JsonModelClient:
    def __init__(self,observe):
        self.base=os.getenv('LLM_BASE_URL','http://127.0.0.1:11435')
        self.observe=observe

    def json(self,model,system,payload,schema,max_tokens=220):
        started=time.perf_counter()
        request={"model":model,"stream":False,"format":schema,
                 "messages":[{"role":"system","content":system},
                             {"role":"user","content":json.dumps(payload,ensure_ascii=False)}],
                 "options":{"temperature":0,"seed":20261111,"num_ctx":4096,"num_predict":max_tokens},
                 "keep_alive":"10m"}
        if model.startswith('qwen3'):request['think']=False
        req=urllib.request.Request(self.base+'/api/chat',data=json.dumps(request).encode(),
                                   headers={'Content-Type':'application/json'})
        try:
            with LOCAL_HTTP.open(req,timeout=45) as r:d=json.load(r)
            content=d['message']['content']
            result=json.loads(content)
            if not d.get('done',False) or d.get('done_reason')=='length':
                raise ValueError('Incomplete model response')
            self.observe(model,time.perf_counter()-started,d.get('prompt_eval_count',0),
                         d.get('eval_count',0),None,d)
            return result
        except (urllib.error.URLError,TimeoutError,ValueError,KeyError) as e:
            self.observe(model,time.perf_counter()-started,0,0,type(e).__name__)
            raise RuntimeError('LLM call failed: '+type(e).__name__) from e
