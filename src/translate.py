#!/bin/python

from __future__ import print_function

########################################
import argparse
parser=argparse.ArgumentParser('predict with a skipgram model')
parser.add_argument('--model_dir',type=str,required=True)
parser.add_argument('--source_words',type=str,required=True)
#parser.add_argument('--source_lang',type=str,required=True)
args = parser.parse_args()

with open(args.model_dir+'/args.json','r') as f:
    import simplejson as json
    data=json.loads(f.readline())
    args_train=type('',(),data)
    #try:
        #args_train.data
    #except:
        #args_train.data='data/escrawl'

########################################
print('loading tensorflow')
import tensorflow as tf
import model.predict
import model.common

print('creating estimator')
estimator=tf.estimator.Estimator(
    model_fn=model.predict.model_fn,
    model_dir=args.model_dir,
    params=args_train,
    )

print('creating predictor')
xs=estimator.predict(
    input_fn=lambda: model.predict.input_fn(args.source_words),
    )

labels_names=model.common.model2labels(args.model_dir)

for x in xs:
    print('source word: %s '%x['words'])
    for i in range(x['translation'].shape[1]):
        print('%s: '%labels_names[0][i],end='')
        for j in range(x['translation'].shape[0]):
            print('%10s (%0.2f %3d) '%(
                    x['translation'][j,i],
                    x['translation_scores'][j,i],
                    x['translation_word_count'][j,i]
                    ),
                end=' ',
                )
        print()
    print()
    print()

