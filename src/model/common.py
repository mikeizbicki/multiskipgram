from __future__ import print_function

def data2labels(data):
    import os
    import tensorflow as tf

    with tf.variable_scope('model_properties',reuse=tf.AUTO_REUSE):

        files = [file for file in os.listdir(data) if os.path.isfile(os.path.join(data, file))]

        # validate input filename format
        separator_counts=[file.count('-') for file in files]
        if not separator_counts[1:] == separator_counts[:-1]:
            raise ValueError('files in ['+data+'] do not all have same number of -')
        num_axes=separator_counts[0]+1

        # extract labels from filenames
        labels=[]
        for axis in range(num_axes):
            labels_axis=sorted(list(set([ file.split('-')[axis] for file in files ])))
            labels.append(labels_axis)
            tf.get_variable(
                name='labels_axis_'+str(axis),
                initializer=labels_axis,
                dtype=tf.string,
                trainable=False,
                )

        num_axes=len(labels)
        tf.get_variable(
            name='num_axes',
            initializer=num_axes,
            dtype=tf.int32,
            trainable=False,
            )

        return labels

def model2labels(model):
    from tensorflow.python import pywrap_tensorflow
    import tensorflow as tf
    import os

    latest_ckp = tf.train.latest_checkpoint(model)
    reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
    var_to_shape_map = reader.get_variable_to_shape_map()
    labels=[]
    for key in var_to_shape_map:
        if key.startswith('model_properties/labels_axis_'):
            labels.append(reader.get_tensor(key))
    return labels

def get_vocab_index(vocab_size,data):
    import tensorflow as tf
    import pickle

    working_dir=data+'/vocab'
    vocab_filename=working_dir+'/all.vocab'

    # generate vocab index for vocab file size if it doesn't exist
    vocab_top_filename=working_dir+'/'+str(vocab_size)+'.vocab'
    try:
        with open(vocab_top_filename,'r') as f:
            vocab_top=pickle.load(f)
    except:
        # generate overall vocab index if it doesn't exist
        try:
            with open(vocab_filename,'rb') as f:
                vocab=pickle.load(f)
        except:
            import os
            try:
                os.mkdir(working_dir)
            except:
                pass
            files = [file for file in os.listdir(data) if os.path.isfile(os.path.join(data, file))]
            vocabfiles=[]
            for file in files:
                vocabfile=working_dir+'/'+os.path.basename(file)+'.vocab'
                mkVocabPickle(data+'/'+file,vocabfile)
                vocabfiles.append(vocabfile)
            mergeVocabPickles(vocabfiles,vocab_filename)
            with open(vocab_filename,'rb') as f:
                vocab=pickle.load(f)

        # trim overall vocab index to appropriate size
        vocab_top=vocab.most_common(vocab_size-1)
        vocab_top.append(('<<UNK>>',1))
        with open(vocab_top_filename,'wb') as f:
            pickle.dump(vocab_top,f)
    vocab_words=[x.encode('unicode_escape').decode('unicode_escape') for x,y in vocab_top]
    vocab_counts=[y for x,y in vocab_top]
    #vocab_words=map(lambda x,y: x.encode('unicode_escape').decode('unicode_escape'),vocab_top)
    #vocab_counts=map(lambda x,y: y,vocab_top)
    return (tf.contrib.lookup.index_table_from_tensor(
        vocab_words,
        default_value=vocab_size-1,
        ),
        vocab_words,
        vocab_counts
        )

def mkVocabPickle(datafile,vocabfile):
    from collections import Counter
    import datetime
    import os
    import re
    import pickle

    vocab=Counter()
    print('mkVocabPickle: '+datafile+'  (may take a while)')

    # do not create pickle file if already exists
    if os.path.isfile(vocabfile):
        print('  file exists')
        return

    # compute vocab
    with open(datafile,'r') as f:
        count=0
        for line in f:
            count+=1
            if count%1000000==0:
                print(datetime.datetime.now(),'count=',count)

            tokens=re.findall(r"\w+|[^\w\s]", line, re.UNICODE)
            vocab.update(tokens)

    # save output
    with open(vocabfile,'wb') as f:
        pickle.dump(vocab,f)

def mergeVocabPickles(vocabfiles,outfile):
    import pickle
    vocabs=[]
    for filename in vocabfiles:
        print('  ',filename)
        with open(filename,'rb') as f:
            vocabs.append(pickle.load(f))

    from functools import reduce
    sum_vocabs=reduce(lambda x,y:x+y,vocabs)
    with open(outfile,'wb') as f:
        pickle.dump(sum_vocabs,f)
