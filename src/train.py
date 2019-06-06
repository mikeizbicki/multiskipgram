#!/bin/python

from __future__ import print_function

########################################
import argparse
import model.train

parser=argparse.ArgumentParser('PROG', usage='%(prog)s [options]')
parser.add_argument('--output_dir',type=str,default=None)
parser.add_argument('--save_summary_steps',type=int,default=100)
parser.add_argument('--steps',type=int,default=None)
parser.add_argument('--seed',type=int,default=0)
model.train.modify_parser(parser)
args = parser.parse_args()

if args.output_dir is not None:
    import os
    try:
        os.makedirs(args.output_dir)
    except:
        pass
    import simplejson as json
    args_str=json.dumps(vars(args))
    with open(args.output_dir+'/args.json','w') as f:
        f.write(args_str)

########################################
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
estimator=tf.estimator.Estimator(
    model_fn=model.train.model_fn,
    model_dir=args.output_dir,
    params=args,
    config=tf.estimator.RunConfig(
        save_summary_steps=args.save_summary_steps,
        tf_random_seed=args.seed,
        ),
    )

estimator.train(
    input_fn=lambda: model.train.input_fn(args),
    steps=args.steps,
    )

