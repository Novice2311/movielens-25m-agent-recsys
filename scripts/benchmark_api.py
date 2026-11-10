"""Small HTTP throughput/latency probe for the direct retrieval path."""
import concurrent.futures,json,time,urllib.request
from pathlib import Path
import numpy as np
LOCAL_HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(i):
    started=time.perf_counter()
    with LOCAL_HTTP.open('http://127.0.0.1:18080/api/recommendations?user_id=1&k=10',timeout=20) as r:d=json.load(r)
    assert len(d['items'])==10
    assert len({m['movieId'] for m in d['items']})==10
    return time.perf_counter()-started

def main():
    request(0)
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:times=list(pool.map(request,range(100)))
    elapsed=time.perf_counter()-started
    result={'endpoint':'GET /api/recommendations','requests':len(times),'concurrency':8,
            'elapsed_seconds':elapsed,'requests_per_second':len(times)/elapsed,
            'p50_ms':float(np.median(times)*1000),'p95_ms':float(np.quantile(times,.95)*1000),
            'scope':'Local loopback, warmed artifacts, same user, no LLM, 100 requests; not a production load test.'}
    Path('results/api_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
