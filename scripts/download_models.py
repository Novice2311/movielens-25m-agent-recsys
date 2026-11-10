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
    records=[]
    for family,tag in NAMES:
        base=f'https://registry.ollama.ai/v2/library/{family}'
        with urllib.request.urlopen(base+'/manifests/'+tag,timeout=60) as r:m=json.load(r)
        layer=next(x for x in m['layers'] if x['mediaType']=='application/vnd.ollama.image.model')
        digest=layer['digest'].split(':')[1]
        out=root/(family+'-'+tag+'.gguf')
        if not out.exists() or hashlib.sha256(out.read_bytes()).hexdigest()!=digest:
            print('Downloading',family,tag,layer['size'],flush=True)
            urllib.request.urlretrieve(base+'/blobs/'+layer['digest'],out)
        assert hashlib.sha256(out.read_bytes()).hexdigest()==digest
        records.append({'name':family+':'+tag,'family':family,'tag':tag,
                        'digest':layer['digest'],'size':layer['size']})
    (root/'models.json').write_text(json.dumps({'models':records},indent=2))
    Path('results').mkdir(exist_ok=True)
    Path('results/llm_models.json').write_text(json.dumps({'models':records},indent=2))

if __name__=='__main__':main()
