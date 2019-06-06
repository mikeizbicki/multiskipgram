from __future__ import print_function

def input_fn(datafile,batch_size=128):
    import tensorflow as tf
    dataset=tf.data.TextLineDataset([datafile])
    dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(1)
    return dataset

def model_fn(
    features, # tensor objects of type string, each representing an individual word
    labels,   # unused
    mode,     # must be tf.estimator.ModeKeys.PREDICT:
    params,   # FIXME
    ):
    import tensorflow as tf
    args=params

    # create vocab hash
    print('  create vocab')
    import model.common
    vocab_index,vocab_words,vocab_counts=model.common.get_vocab_index(args.vocab_size,args.data)

    # create variables
    labels_names=model.common.data2labels(args.data)
    labels_num=[len(axis) for axis in labels_names ]
    import train
    train.macro_input_variables(args)
    train.macro_model_variables(args)

    inputs_projector=inputs_projector_axis[0]

    # create word vectors
    wordvecs_raw=tf.gather(
        inputs,
        vocab_index.lookup(features),
        )
    wordvecs=tf.tensordot(
        wordvecs_raw,
        inputs_projector,
        axes=[[2],[0]],
        )
    inputs_mean=tf.reduce_mean(wordvecs,axis=2)
    wordvecs_all=tf.tensordot(
        inputs,
        inputs_projector,
        axes=[[2],[0]],
        )

    # create word vectors
    labels=model.common.data2labels(args.data)
    axis0_source=labels[0].index('cl')
    wordvecs_source=wordvecs[:,:,axis0_source]

    # create translation
    res_translation=[]
    res_scores=[]
    res_word_count=[]
    for label in range(labels_num[0]):
        cosine_similarities=tf.tensordot(
            tf.nn.l2_normalize(wordvecs_source,axis=1),
            tf.nn.l2_normalize(wordvecs_all[:,:,label],axis=1),
            axes=[[1],[1]],
            )
        vals,indices=tf.nn.top_k(cosine_similarities,k=4)
        predictions=tf.gather(vocab_words,indices)
        res_translation.append(predictions)
        res_scores.append(vals)
        res_word_count.append(tf.gather(word_counts_axis[0][:,label],indices))
    translation=tf.stack(res_translation,axis=2)
    translation_scores=tf.stack(res_scores,axis=2)
    translation_word_count=tf.stack(res_word_count,axis=2)

    # return
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions={
                'words':features,
                'translation':translation,
                'translation_scores':translation_scores,
                'translation_word_count':translation_word_count,
                },
            )

