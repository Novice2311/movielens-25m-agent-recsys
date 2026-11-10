"""Fit a small typed decision model, calibrate on dev, evaluate on held-out test."""
import json
from pathlib import Path
import numpy as np
from agent.decision import features,LABELS,DIM

DATA={
 'recommend':[
  'Подбери мне фантастику','Хочу посмотреть комедию','Посоветуй фильмы на вечер',
  'Дай рекомендации по моим оценкам','Покажи что-нибудь романтическое',
  'Хочу хороший боевик','Что посмотреть с семьей','Найди мультфильм',
  'Ищу драму без ужасов','Мне нравятся триллеры','Подбери три фильма',
  'Recommend science fiction movies','Show comedy films','What should I watch tonight',
  'Find me a good drama','Suggest an animation','I want movie recommendations',
  'Порекомендуй популярное кино','Есть интересное кино для меня','Посоветуй документалку'],
 'clarify':[
  'Помоги','Не знаю','Что-нибудь','Ну','Не уверен что хочу','Мне скучно',
  'Может быть','Пока не решил','Привет','Здравствуйте','С чего начать',
  'help','hello','I do not know','Not sure','Anything','Hi there','Maybe','Кто ты','Начнем'],
 'memory':[
  'Запомни я не люблю ужасы','Сохрани мои предпочтения','Забудь мои предпочтения',
  'Удали мою историю','Запомни мне нравится фантастика','Обнови любимые жанры',
  'Не сохраняй мои данные','Что ты обо мне помнишь','Покажи мою память',
  'Запомни что мне нравятся комедии','remember my preferences','Forget my history',
  'Delete my stored data','Save my favorite genres','What do you remember about me',
  'Update my movie preferences','Erase my memory','Не забывай мои вкусы',
  'Запиши мое предпочтение','Сотри все данные обо мне']}
DEV=[('Подскажи боевик','recommend'),('Подбери романтическое кино','recommend'),
 ('Suggest a movie to watch','recommend'),('Не определился','clarify'),('Приветик','clarify'),
 ('I need help','clarify'),('Забудь всё обо мне','memory'),('Запомни любимые жанры','memory'),
 ('Remember I like action movies','memory')]
TEST=[('Посоветуй фантастическое кино без ужасов','recommend'),('Подбери пять комедий','recommend'),
 ('Хочу посмотреть драму вечером','recommend'),('Recommend a documentary film','recommend'),
 ('Найди интересный мультфильм','recommend'),('Show me science fiction','recommend'),
 ('Не могу решить','clarify'),('Добрый день','clarify'),('Пока не понимаю','clarify'),
 ('Hey there','clarify'),('I am not sure','clarify'),('Нужна помощь','clarify'),
 ('Удали все мои предпочтения','memory'),('Запомни что я люблю драмы','memory'),
 ('Что сохранилось в памяти','memory'),('Remember that I like comedy','memory'),
 ('Forget everything about me','memory'),('Сохрани список моих жанров','memory')]

def softmax(z):
 e=np.exp(z-z.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)

def main():
 texts=[];ys=[]
 for label in LABELS:
  for s in DATA[label]:texts.append(s);ys.append(LABELS.index(label))
 x=np.array([features(t) for t in texts]);y=np.eye(3)[ys]
 w=np.zeros((DIM,3));b=np.zeros(3)
 for _ in range(1500):
  error=softmax(x@w+b)-y
  w-=.8*(x.T@error/len(x)+.001*w);b-=.8*error.mean(0)
 devx=np.array([features(t) for t,_ in DEV]);devy=np.array([LABELS.index(y) for _,y in DEV])
 candidates=np.linspace(.3,2,80)
 temperature=min(candidates,key=lambda t:-np.log(softmax((devx@w+b)/t)[np.arange(len(DEV)),devy]).mean())
 testx=np.array([features(t) for t,_ in TEST]);testy=np.array([LABELS.index(y) for _,y in TEST])
 prob=softmax((testx@w+b)/temperature);pred=prob.argmax(1);conf=prob.max(1)
 ece=0
 for low in np.linspace(0,.9,10):
  mask=(conf>=low)&(conf<low+.1+1e-8)
  if mask.any():ece+=mask.mean()*abs((pred[mask]==testy[mask]).mean()-conf[mask].mean())
 results={'train_n':len(texts),'dev_n':len(DEV),'test_n':len(TEST),
          'accuracy':float((pred==testy).mean()),'brier':float(((prob-np.eye(3)[testy])**2).sum(1).mean()),
          'ece_10bins':float(ece),'coverage_at_0_8':float((conf>=.8).mean()),
          'temperature':float(temperature),
          'cases':[{'text':text,'expected':label,'predicted':LABELS[p], 'confidence':float(c)}
                   for (text,label),p,c in zip(TEST,pred,conf)],
          'limitations':'18 authored test phrases; narrow routing task; not a general Jev replacement'}
 Path('artifacts').mkdir(exist_ok=True);Path('results').mkdir(exist_ok=True)
 # The small trained gate is redistributable; it is not derived from MovieLens records.
 Path('agent/decision_weights.json').write_text(json.dumps({'weights':w.tolist(),'bias':b.tolist(),
                                        'temperature':float(temperature)}),encoding='utf-8')
 Path('results/decision_eval.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
