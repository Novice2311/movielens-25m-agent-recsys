"""Download exact GGUF model blobs from the public Ollama registry; verify SHA256.

Ollama is used as a distribution registry. Runtime inference uses llama.cpp.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

NAMES=[('qwen3','0.6b'),('qwen3.5','0.8b'),('gemma3','270m')]

def main():
    root=Path('models');root.mkdir(exist_ok=True)
    # Once an experiment is published, replay the recorded immutable digests.
    recorded=Path('results/llm_models.json')
    pinned={r['name']:r for r in json.loads(recorded.read_text())['models']} if recorded.exists() else {}
    records=[]
    for family,tag in NAMES:
        base=f'https://registry.ollama.ai/v2/library/{family}'
        layer=pinned.get(family+':'+tag)
        if layer is None:
            with urllib.request.urlopen(base+'/manifests/'+tag,timeout=60) as r:m=json.load(r)
            layer=next(x for x in m['layers'] if x['mediaType']=='application/vnd.ollama.image.model')
        digest=layer['digest'].split(':')[1]
        out=root/(family+'-'+tag+'.gguf')
        def matches(path):
            if not path.exists() or path.stat().st_size!=layer['size']:return False
            with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==digest
        if not matches(out):
            print('Downloading',family,tag,layer['size'],flush=True)
            part=out.with_suffix('.part')
            urllib.request.urlretrieve(base+'/blobs/'+layer['digest'],part)
            assert matches(part), 'GGUF SHA256 verification failed'
            part.replace(out)
        assert matches(out)
        records.append({'name':family+':'+tag,'family':family,'tag':tag,
                        'digest':layer['digest'],'size':layer['size']})
    (root/'models.json').write_text(json.dumps({'models':records},indent=2))
    Path('results').mkdir(exist_ok=True)
    Path('results/llm_models.json').write_text(json.dumps({'models':records},indent=2))

if __name__=='__main__':main()
