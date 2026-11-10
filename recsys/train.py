"""Temporal, full-catalog experiment. No sampled-negative evaluation.

Run: python -m recsys.train --zip ../ml-25m.zip --epochs 3 --eval-users 1000
Training uses only ratings strictly before 2018-01-01 UTC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20261111
TRAIN_END = 1514764800
VAL_END = 1546300800


class TwoTower(nn.Module):
    """Independent nonlinear user and item encoders, joined by dot product."""
    def __init__(self, n_users: int, genres: np.ndarray, profiles: np.ndarray):
        super().__init__()
        self.user_id = nn.Embedding(n_users, 64)
        self.item_id = nn.Embedding(len(genres), 64)
        self.register_buffer("genres", torch.from_numpy(genres))
        self.register_buffer("profiles", torch.from_numpy(profiles))
        self.user_mlp = nn.Sequential(nn.Linear(64 + genres.shape[1], 64),
                                      nn.ReLU(), nn.Linear(64, 32))
        self.item_mlp = nn.Sequential(nn.Linear(64 + genres.shape[1], 64),
                                      nn.ReLU(), nn.Linear(64, 32))
        nn.init.normal_(self.user_id.weight, std=.05)
        nn.init.normal_(self.item_id.weight, std=.05)

    def users(self, u):
        x = torch.cat((self.user_id(u), self.profiles[u]), dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def items(self, i):
        x = torch.cat((self.item_id(i), self.genres[i]), dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)


def topk(scores, k=10):
    """Stable ties by dense item index; works with 1D or 2D arrays."""
    a = np.atleast_2d(scores)
    out = []
    for row in a:
        threshold=np.partition(row,len(row)-k)[len(row)-k]
        better=np.flatnonzero(row>threshold)
        tied=np.flatnonzero(row==threshold)[:k-len(better)]
        ids=np.concatenate((better,tied))
        out.append(ids[np.lexsort((ids, -row[ids]))])
    return np.array(out)


def score_list(prediction, truth):
    relevant = set(int(x) for x in truth)
    hit = np.array([int(x in relevant) for x in prediction], dtype=float)
    recall = hit.sum() / len(relevant)
    dcg = (hit / np.log2(np.arange(2, len(hit)+2))).sum()
    ideal = (1 / np.log2(np.arange(2, min(len(relevant), len(hit))+2))).sum()
    return float(recall), float(dcg / ideal)


def encode_model(model, n_users, n_items):
    model.eval()
    with torch.no_grad():
        users = torch.cat([model.users(torch.arange(i, min(i+8192, n_users)))
                           for i in range(0, n_users, 8192)]).numpy()
        items = torch.cat([model.items(torch.arange(i, min(i+8192, n_items)))
                           for i in range(0, n_items, 8192)]).numpy()
    return users, items


def evaluate(split, users, truths, seen, positive, user_norm, genres,
             profiles, popularity, recent, vectors, catalog):
    """All methods see exactly the same eligible training catalog and history."""
    user_vec, item_vec = vectors
    models = ["Popularity", "Recent90d", "ContentGenres", "UserKNN40", "TwoTower"]
    records = {m: [] for m in models}
    recommendations = {m: set() for m in models}
    elapsed = {m: [] for m in models}
    actual = []
    for number, uid in enumerate(users):
        uid = int(uid)
        mask_ids = seen[uid].indices
        y = truths[uid]
        for name in models:
            started = time.perf_counter()
            if name == "Popularity":
                scores = popularity.astype(np.float32).copy()
            elif name == "Recent90d":
                scores = recent.astype(np.float32).copy()
            elif name == "ContentGenres":
                scores = genres @ profiles[uid]
            elif name == "UserKNN40":
                similarity = (user_norm[uid] @ user_norm.T).toarray().ravel()
                similarity[uid] = 0
                near = np.argpartition(-similarity, 40)[:40]
                weights = similarity[near]
                scores = np.asarray(weights @ positive[near]).ravel()
            else:
                scores = item_vec @ user_vec[uid]
            scores[~catalog] = -np.inf
            scores[mask_ids] = -np.inf
            pred = topk(scores, 10)[0]
            assert len(pred) == len(set(pred)) == 10
            assert not np.isin(pred, mask_ids).any()
            recall, ndcg = score_list(pred, y)
            elapsed[name].append((time.perf_counter()-started)*1000)
            records[name].append([recall, ndcg])
            recommendations[name].update(int(x) for x in pred)
        actual.append(uid)
        if (number+1) % 250 == 0:
            print(f"{split}: {number+1}/{len(users)} users", flush=True)
    summary = {}
    for name in models:
        arr = np.array(records[name])
        rng = np.random.default_rng(SEED)
        means = np.array([arr[rng.integers(len(arr), size=len(arr)), 1].mean()
                          for _ in range(1000)])
        summary[name] = {
            "recall_at_10": float(arr[:, 0].mean()),
            "ndcg_at_10": float(arr[:, 1].mean()),
            "ndcg_95ci": np.quantile(means, [.025, .975]).tolist(),
            "catalog_coverage_at_10": len(recommendations[name])/int(catalog.sum()),
            "latency_ms_p50": float(np.median(elapsed[name])),
            "latency_ms_p95": float(np.quantile(elapsed[name], .95)),
        }
    a = np.array(records["TwoTower"])[:, 1]
    b = np.array(records["Popularity"])[:, 1]
    rng = np.random.default_rng(SEED)
    delta = np.array([(a-b)[rng.integers(len(a), size=len(a))].mean()
                      for _ in range(2000)])
    summary["paired_two_tower_minus_popularity_ndcg_ci95"] = np.quantile(delta, [.025,.975]).tolist()
    # Per-user records are kept locally for audit, not redistributed with MovieLens IDs.
    Path("artifacts").mkdir(exist_ok=True)
    np.savez_compressed(f"artifacts/{split}_audit.npz", users=actual,
                        **{k: np.array(v) for k,v in records.items()})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="../ml-25m.zip")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--eval-users", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8192)
    args = p.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(6)
    started = time.perf_counter()
    Path("artifacts").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    archive_hash = hashlib.md5(Path(args.zip).read_bytes()).hexdigest()
    with zipfile.ZipFile(args.zip) as z:
        movies = pd.read_csv(z.open("ml-25m/movies.csv"))
        ratings = pd.read_csv(z.open("ml-25m/ratings.csv"), dtype={
            "userId":"int32", "movieId":"int32", "rating":"float32", "timestamp":"int64"})
    assert len(ratings) == 25000095 and len(movies) == 62423
    assert ratings.rating.between(.5, 5).all()
    assert (ratings.rating*2 % 1 == 0).all()
    assert not ratings.duplicated(["userId","movieId"]).any()
    assert ratings.movieId.isin(movies.movieId).all()
    movies = movies.sort_values("movieId").reset_index(drop=True)
    mid = movies.movieId.to_numpy()
    # Dense user index 0 is intentionally unused and represents unknown users.
    uid = ratings.userId.to_numpy()
    iid = np.searchsorted(mid, ratings.movieId.to_numpy()).astype(np.int32)
    timestamp = ratings.timestamp.to_numpy()
    rating = ratings.rating.to_numpy()
    n_users = int(uid.max())+1
    n_items = len(mid)
    genre_names = sorted(set("|".join(movies.genres).split("|")) - {"(no genres listed)"})
    genres = np.array([[float(g in x.split("|")) for g in genre_names]
                        for x in movies.genres], dtype=np.float32)
    genres /= np.maximum(np.linalg.norm(genres,axis=1,keepdims=True),1)
    train = timestamp < TRAIN_END
    validation = (timestamp >= TRAIN_END) & (timestamp < VAL_END)
    test = timestamp >= VAL_END
    pos = train & (rating >= 4)
    shape=(n_users,n_items)
    seen = sp.csr_matrix((np.ones(train.sum(),dtype=np.float32),
                         (uid[train],iid[train])),shape=shape)
    positive = sp.csr_matrix((np.ones(pos.sum(),dtype=np.float32),
                             (uid[pos],iid[pos])),shape=shape)
    counts = np.asarray(positive.sum(axis=1)).ravel()
    catalog = np.asarray(seen.sum(axis=0)).ravel() > 0
    popularity = np.asarray(positive.sum(axis=0)).ravel()
    recent_mask = pos & (timestamp >= TRAIN_END - 90*86400)
    recent = np.bincount(iid[recent_mask], minlength=n_items).astype(np.float32)
    profiles = (positive @ genres) / np.maximum(counts[:,None],1)
    profiles = profiles.astype(np.float32)
    user_norm = sp.diags(1/np.maximum(np.sqrt(counts),1)) @ positive
    def truth_for(mask, history):
        target = mask & (rating >=4) & catalog[iid] & (counts[uid]>=5)
        df = pd.DataFrame({"u":uid[target],"i":iid[target]})
        truths = {int(u):np.array(v.i,dtype=np.int32) for u,v in df.groupby("u",sort=True)}
        rng = np.random.default_rng(SEED)
        pool = np.array(sorted(truths))
        users = np.sort(rng.choice(pool,min(args.eval_users,len(pool)),replace=False))
        return truths,users
    val_truth,val_users = truth_for(validation,seen)
    seen_test = sp.csr_matrix((np.ones((train|validation).sum(),dtype=np.float32),
                              (uid[train|validation],iid[train|validation])),shape=shape)
    test_truth,test_users = truth_for(test,seen_test)
    split_info = {
        "source_md5":archive_hash,"ratings":len(ratings),"movies":n_items,
        "users":n_users-1,"train_ratings":int(train.sum()),
        "validation_ratings":int(validation.sum()),"test_ratings":int(test.sum()),
        "train_positive_pairs":int(pos.sum()),"train_catalog":int(catalog.sum()),
        "val_eligible_users":len(val_truth),"test_eligible_users":len(test_truth),
        "val_evaluated_users":len(val_users),"test_evaluated_users":len(test_users),
        "test_positive_total":int((test & (rating>=4)).sum()),
        "test_positive_warm":int((test & (rating>=4) & catalog[iid] & (counts[uid]>=5)).sum()),
        "train_end_utc":"2018-01-01","validation_end_utc":"2019-01-01",
        "seed":SEED,"genre_features":len(genre_names),"python":platform.python_version(),"torch":torch.__version__,
        "device":"cpu","torch_threads":torch.get_num_threads(),
    }
    print(json.dumps(split_info),flush=True)
    Path("results/split.json").write_text(json.dumps(split_info,indent=2),encoding="utf-8")
    model = TwoTower(n_users,genres,profiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    pu,pi=uid[pos].copy(),iid[pos].copy()
    # Negative samples exclude ALL training ratings, including low ratings.
    keys = np.sort(uid[train].astype(np.int64)*n_items + iid[train])
    train_items=np.flatnonzero(catalog)
    rng = np.random.default_rng(SEED)
    def negatives(u):
        n=rng.choice(train_items,len(u))
        while True:
            encoded=u.astype(np.int64)*n_items+n
            where=np.searchsorted(keys,encoded)
            collision=(where<len(keys)) & (keys[np.minimum(where,len(keys)-1)]==encoded)
            if not collision.any():return n
            n[collision]=rng.choice(train_items,int(collision.sum()))
    log=[]
    best=-1
    for epoch in range(args.epochs):
        model.train();order=rng.permutation(len(pu));running=0;ep=time.perf_counter()
        for j in range(0,len(order),args.batch_size):
            rows=order[j:j+args.batch_size]
            u=torch.from_numpy(pu[rows].astype(np.int64))
            i=torch.from_numpy(pi[rows].astype(np.int64))
            n=torch.from_numpy(negatives(pu[rows]).astype(np.int64))
            uv,iv,nv=model.users(u),model.items(i),model.items(n)
            margin=(uv*(iv-nv)).sum(-1)/.1
            loss=F.softplus(-margin).mean()
            loss+=1e-5*(model.user_id(u).square().mean()+model.item_id(i).square().mean())
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            running+=loss.item()*len(rows)
        vectors=encode_model(model,n_users,n_items)
        # Validation only is used to select epoch. The test set is opened once below.
        quick=[]
        for u in val_users[:300]:
            s=vectors[1]@vectors[0][u];s[~catalog]=-np.inf;s[seen[u].indices]=-np.inf
            quick.append(score_list(topk(s)[0],val_truth[int(u)])[1])
        result={"epoch":epoch+1,"loss":running/len(pu),
                "seconds":time.perf_counter()-ep,"validation_ndcg_300":float(np.mean(quick))}
        log.append(result);print(json.dumps(result),flush=True)
        if result["validation_ndcg_300"]>best:
            best=result["validation_ndcg_300"];torch.save(model.state_dict(),"artifacts/two_tower.pt")
    model.load_state_dict(torch.load("artifacts/two_tower.pt",weights_only=True))
    vectors=encode_model(model,n_users,n_items)
    np.savez_compressed("artifacts/serving.npz",user_vectors=vectors[0],item_vectors=vectors[1],
                        popularity=popularity,recent=recent,catalog=catalog,
                        movie_ids=mid,genres=genres,profiles=profiles)
    sp.save_npz("artifacts/seen.npz",seen)
    movies.to_json("artifacts/movies.json",orient="records",force_ascii=False)
    np.save("artifacts/genre_names.npy",np.array(genre_names))
    out={"protocol":split_info,"training":log}
    out["validation"]=evaluate("validation",val_users,val_truth,seen,positive,user_norm,
                    genres,profiles,popularity,recent,vectors,catalog)
    out["test"]=evaluate("test",test_users,test_truth,seen_test,positive,user_norm,
                    genres,profiles,popularity,recent,vectors,catalog)
    out["total_seconds"]=time.perf_counter()-started
    Path("results/recsys_metrics.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2),flush=True)


if __name__=="__main__":main()
