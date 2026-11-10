"""Read-only typed tools. No shell, arbitrary SQL, arbitrary URL or file tools."""
import json
from pathlib import Path

import numpy as np


class Catalog:
    def __init__(self,root):
        root=Path(root)
        self.a=dict(np.load(root/'serving.npz'))
        self.movies=json.loads((root/'movies.json').read_text(encoding='utf-8'))
        self.genre_names=np.load(root/'genre_names.npy').tolist()
        history=np.load(root/'seen.npz')
        self.indptr,self.indices=history['indptr'],history['indices']
        self.ids={int(m['movieId']):i for i,m in enumerate(self.movies)}

    def seen(self,user_id):
        if not 0<user_id<len(self.indptr)-1:return np.array([],dtype=np.int32)
        return self.indices[self.indptr[user_id]:self.indptr[user_id+1]]

    def retrieve(self,user_id,genres=(),exclude=(),k=20,exclude_ids=()):
        if not set(genres).issubset(self.genre_names) or not set(exclude).issubset(self.genre_names):
            raise ValueError('Unknown genre')
        a=self.a
        known = ('user_vectors' in a and 0<user_id<len(a['user_vectors'])
                 and len(self.seen(user_id))>=5 and np.any(a['profiles'][user_id]))
        try:
            score=(a['item_vectors']@a['user_vectors'][user_id]) if known else np.log1p(a['popularity'])
            if not np.isfinite(score).all():raise ValueError('Non-finite model output')
        except (ValueError,KeyError,IndexError):
            known=False
            score=np.log1p(a['popularity'])
        score=score.copy()
        eligible=a['catalog'].copy()
        if genres:
            wanted=[self.genre_names.index(g) for g in genres]
            eligible &= (a['genres'][:,wanted]>0).any(axis=1)
        if exclude:
            excluded=[self.genre_names.index(g) for g in exclude]
            eligible &= ~(a['genres'][:,excluded]>0).any(axis=1)
        eligible[self.seen(user_id)]=False
        if exclude_ids:
            eligible[[self.ids[i] for i in exclude_ids if i in self.ids]]=False
        available=np.flatnonzero(eligible)
        chosen=available[np.lexsort((available,-score[available]))[:k]]
        return [{"movieId":int(self.movies[i]['movieId']),
                 "title":self.movies[i]['title'],"genres":self.movies[i]['genres'].split('|'),
                 "score":round(float(score[i]),4),
                 "basis":"personal_history" if known else "popular_fallback"} for i in chosen]

    def verify(self,selected,candidates,plan,k):
        by_id={m['movieId']:m for m in candidates}
        ids=[int(x['movieId']) for x in selected]
        errors=[]
        if len(ids)!=min(k,len(candidates)) or len(ids)!=len(set(ids)):errors.append('count_or_duplicate')
        if not set(ids).issubset(by_id):errors.append('unknown_movie_id')
        for x in selected:
            if x.get('movieId') not in by_id:continue
            movie=by_id[x['movieId']]
            if x.get('evidence_genre') not in movie['genres']:errors.append('unsupported_genre')
            if set(movie['genres']) & set(plan.get('exclude',[])):errors.append('excluded_genre')
            if plan.get('genres') and not (set(movie['genres']) & set(plan['genres'])):errors.append('missing_requested_genre')
        return sorted(set(errors))
