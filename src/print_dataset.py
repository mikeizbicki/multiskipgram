#!/bin/python

from __future__ import print_function

########################################
import argparse
import model.train

parser=argparse.ArgumentParser('PROG', usage='%(prog)s [options]')
model.train.modify_parser(parser)
args = parser.parse_args()

########################################
print('loading tensorflow')
import tensorflow as tf

dataset=model.train.input_fn(args)
iterator = dataset.make_initializable_iterator()
next_op=iterator.get_next()

print('starting session')
with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    sess.run(tf.initializers.local_variables())
    sess.run(iterator.initializer)
    sess.run(tf.initialize_all_tables())

    for i in range(40):
        next=sess.run(next_op)
        print('  next=',next)
