#!/bin/sh

# a function to run a test and check if it passed or failed
runtest() {
    cmd=$@
    tmp=$(mktemp)
    echo "$cmd"
    $cmd 1>$tmp 2>&1
    if [ ! $? = 0 ]; then
        echo "test failed"
        cat $tmp
        exit 1
    fi
}

# set python version to python3 if available
which python3
if [ ! $? = 0 ]; then
    p=python3
else
    p=python
fi

# the training command
train="$p -u src/train.py --steps=100"

# ensure that we test the code paths that build the vocab directory
rm data/test-axis2/vocab -r

# run tests
runtest $train --data=data/test-axis2 

runtest $train --data=data/test-axis2 --dp_type=word_context
runtest $train --data=data/test-axis2 --dp_type=word_pair

runtest $train --data=data/test-axis2 --reuse_samples=true
runtest $train --data=data/test-axis2 --reuse_samples=false

runtest $train --data=data/test-axis2 --sample_method=nce
runtest $train --data=data/test-axis2 --sample_method=ns

runtest $train --data=data/test-axis2 --word_counts=none
runtest $train --data=data/test-axis2 --word_counts=fast
runtest $train --data=data/test-axis2 --word_counts=all
runtest $train --data=data/test-axis2 --word_counts=all --filtering=none
runtest $train --data=data/test-axis2 --word_counts=all --filtering=global
runtest $train --data=data/test-axis2 --word_counts=all --filtering=axis

runtest $train --data=data/test-axis2 --num_embeddings 1 2
runtest $train --data=data/test-axis2 --num_embeddings 2 1
runtest $train --data=data/test-axis2 --num_embeddings 2 2 --disable_multi inputs
runtest $train --data=data/test-axis2 --num_embeddings 2 2 --disable_multi outputs
