"""Local Jev-inspired bounded decision model, not the proprietary Jev model.

Character trigram features + trained softmax classifier + held-out temperature.
Confidence is reported, never presented as a guarantee of correctness.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = ["recommend", "clarify", "memory"]
DIM = 1024


def features(text):
    text = ' '+text.lower().strip()+' '
    v = np.zeros(DIM,dtype=np.float32)
    for n in (3,4):
        for i in range(max(0,len(text)-n+1)):
            h = int.from_bytes(hashlib.blake2s(text[i:i+n].encode(),digest_size=4).digest(),'little')
            v[h%DIM]+=1
    return v/max(np.linalg.norm(v),1)


class DecisionGate:
    def __init__(self,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        self.w=np.array(d['weights'],dtype=np.float32)
        self.b=np.array(d['bias'],dtype=np.float32)
        self.temperature=d['temperature']

    def decide(self,text):
        z=(features(text)@self.w+self.b)/self.temperature
        p=np.exp(z-z.max());p/=p.sum()
        i=int(p.argmax())
        return {"choice":LABELS[i],"confidence":float(p[i]),
                "probabilities":dict(zip(LABELS,map(float,p))),
                "abstain":bool(p[i]<.80),"model":"local-char-softmax-v1"}
