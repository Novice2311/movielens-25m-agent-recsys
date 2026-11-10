"""Explore the cold or weak-profile cohort without tuning the warm-start model."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from recsys.train import SEED,TRAIN_END,VAL_END,topk,score_list

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default='data/ml-25m.zip');args=p.parse_args()
    with zipfile.ZipFile(args.zip) as z:
        df=pd.read_csv(z.open('ml-25m/ratings.csv'),dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'})
    arrays=np.load('artifacts/serving.npz')
    ids=arrays['movie_ids'];catalog=arrays['catalog'];u=df.userId.to_numpy();i=np.searchsorted(ids,df.movieId.to_numpy())
    t=df.timestamp.to_numpy();r=df.rating.to_numpy();n_users=int(u.max())+1
    train=t<TRAIN_END;positive=train&(r>=4)
    counts=np.bincount(u[positive],minlength=n_users)
    eligible=t>=VAL_END;targets=eligible&(r>=4)&(counts[u]<5)
    warm_items=targets&catalog[i]
    truth={int(uid):g['i'].to_numpy() for uid,g in pd.DataFrame({'u':u[warm_items],'i':i[warm_items]}).groupby('u')}
    hist=t<VAL_END
    seen=sp.csr_matrix((np.ones(hist.sum(),dtype=np.int8),(u[hist],i[hist])),shape=(n_users,len(ids)))
    rng=np.random.default_rng(SEED);users=np.sort(rng.choice(sorted(truth),min(1000,len(truth)),replace=False))
    out={'definition':'fewer than five positive ratings in train; test 2019; same catalog and seen filters',
         'positive_test_ratings_cold':int(targets.sum()),'positive_cold_in_train_catalog':int(warm_items.sum()),
         'eligible_users':len(truth),'evaluated_users':len(users),'models':{}}
    for name,field in [('Popularity','popularity'),('Recent90d','recent')]:
        base=arrays[field].astype(np.float32);rows=[];recommendations=set()
        for uid in users:
            s=base.copy();s[~catalog]=-np.inf;s[seen[uid].indices]=-np.inf
            pred=topk(s)[0];rows.append(score_list(pred,truth[uid]));recommendations.update(pred.tolist())
        means=np.mean(rows,axis=0)
        out['models'][name]={'recall_at_10':float(means[0]),'ndcg_at_10':float(means[1]),
                             'catalog_coverage_at_10':len(recommendations)/int(catalog.sum())}
    audit=np.load('artifacts/test_audit.npz');a=audit['TwoTower'][:,1]
    comparisons={}
    for name in ['UserKNN40','Recent90d']:
        delta=a-audit[name][:,1];samples=np.random.default_rng(SEED).integers(len(delta),size=(2000,len(delta)))
        comparisons[name]={'mean_delta':float(delta.mean()),'ci95':np.quantile(delta[samples].mean(axis=1),[.025,.975]).tolist()}
    out['warm_exploratory_paired_comparisons_two_tower_minus']=comparisons
    Path('results/cold_eval.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
