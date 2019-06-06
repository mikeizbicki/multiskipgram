#!/bin/python

from __future__ import print_function

########################################
import argparse
import model.train

parser=argparse.ArgumentParser('PROG', usage='%(prog)s [options]')
model.train.modify_parser(parser)
args = parser.parse_args()

########################################
import tensorflow as tf

dataset=model.train.input_fn(args)
iterator=dataset.make_initializable_iterator()
next_element=iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.initialize_all_tables())
    sess.run(tf.initialize_all_variables())
    import datetime
    while True:
        start=datetime.datetime.now()
        end=datetime.datetime.now()
        i=0
        while (end-start).total_seconds() < 1.0:
            sess.run(next_element)
            i+=1
            end=datetime.datetime.now()
        print(datetime.datetime.now(),
            ': dp/sec=',args.batch_size*float(i)/(end-start).total_seconds(),
            ': batch/sec=',float(i)/(end-start).total_seconds()
            )

