import contextlib
import json
import os
import threading
import time
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from prometheus_client import Counter, Histogram

REQUESTS=Counter('movie_agent_requests_total','Completed requests',['outcome'])
LLM_CALLS=Counter('movie_agent_llm_calls_total','LLM calls',['model','status'])
LLM_LATENCY=Histogram('movie_agent_llm_seconds','LLM wall latency',['model'],
                      buckets=(.1,.25,.5,1,2,5,10,20,40,60))
TOKENS=Counter('movie_agent_tokens_total','LLM tokens',['model','direction'])
REJECTS=Counter('movie_agent_verifier_reject_total','Rejected selections',['reason'])
REQUEST_TIME=Histogram('movie_agent_request_seconds','End to end latency',
                       buckets=(.1,.5,1,2,5,10,20,40,60,120))
RETRIEVAL_TIME=Histogram('movie_retrieval_seconds','Direct retrieval HTTP handler latency',
                         buckets=(.001,.005,.01,.025,.05,.1,.2,.5,1))


class Observability:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
        provider=TracerProvider(resource=Resource.create({'service.name':'movie-agent'}))
        endpoint=os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint+'/v1/traces')))
        trace.set_tracer_provider(provider)
        self.tracer=trace.get_tracer('movie-agent','1.0.0')

    def log(self,event,**fields):
        row={'timestamp':time.time(),'event':event,**fields}
        text=json.dumps(row,ensure_ascii=False)
        with self.lock:
            with (self.root/'events.jsonl').open('a',encoding='utf-8') as f:f.write(text+'\n')
        print(text,flush=True)

    @contextlib.contextmanager
    def span(self,name,state):
        started=time.perf_counter()
        with self.tracer.start_as_current_span(name) as span:
            tid=f'{span.get_span_context().trace_id:032x}'
            state['trace_id']=tid
            try:yield
            finally:
                entry={'node':name,'ms':round((time.perf_counter()-started)*1000,2)}
                state.setdefault('trajectory',[]).append(entry)
                self.log('node_finished',trace_id=tid,**entry)

    def llm(self,model,seconds,input_tokens,output_tokens,error,meta=None):
        LLM_CALLS.labels(model,'error' if error else 'ok').inc()
        LLM_LATENCY.labels(model).observe(seconds)
        TOKENS.labels(model,'input').inc(input_tokens)
        TOKENS.labels(model,'output').inc(output_tokens)
        span=trace.get_current_span()
        span.set_attribute('gen_ai.request.model',model)
        span.set_attribute('gen_ai.usage.input_tokens',input_tokens)
        span.set_attribute('gen_ai.usage.output_tokens',output_tokens)
        if error:span.set_status(trace.Status(trace.StatusCode.ERROR,error))
        meta=meta or {}
        self.log('llm_call',model=model,seconds=round(seconds,4),input_tokens=input_tokens,
                 output_tokens=output_tokens,error=error,
                 trace_id=f'{span.get_span_context().trace_id:032x}',
                 load_seconds=round(meta.get('load_duration',0)/1e9,4))

    def save_trajectory(self,state):
        record={k:state.get(k) for k in ['trace_id','trajectory','decision','plan','verification','outcome']}
        with self.lock:
            with (self.root/'trajectories.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')

    def recent(self,n=10):
        p=self.root/'trajectories.jsonl'
        if not p.exists():return []
        return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()[-n:]][::-1]
