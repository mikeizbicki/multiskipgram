from __future__ import print_function
from __future__ import absolute_import

def modify_parser(parser):
    group = parser.add_argument_group('model options')
    group.add_argument('--embedding_size',type=int,default=300)
    group.add_argument('--num_embeddings',type=int,nargs='*',default=[])
    group.add_argument('--disable_multi',choices=['inputs','outputs'],nargs='*',default=[])
    group.add_argument('--inputs_equals_outputs',action='store_true')
    group.add_argument('--init_from_checkpoint',type=str,default=None)

    group = parser.add_argument_group('optimization options')
    group.add_argument('--batch_size',type=int,default=128)
    group.add_argument('--optimizer',type=str,choices=['adam','sgd'],default='sgd')
    group.add_argument('--learning_rate',type=float,default=0.1)
    group.add_argument('--learning_rate_decay',type=float,default=1e8)
    group.add_argument('--nce_samples',type=int,default=2**8)
    group.add_argument('--sampler',type=str,choices=['log_uniform','fixed_unigram'],default='fixed_unigram')
    group.add_argument('--reg',type=float,default=0.0)
    group.add_argument('--outputs_method',type=str,choices=['averaged','per_label'],default='per_label')

    group = parser.add_argument_group('data pipeline options')
    group.add_argument('--data',type=str,required=True)
    group.add_argument('--vocab_size',type=int,default=2**16)
    group.add_argument('--shuffle_lines',type=int,default=2**14)
    group.add_argument('--shuffle_words',type=int,default=2**14)
    group.add_argument('--shuffle_seed',type=int,default=0)
    group.add_argument('--context_size',type=int,default=5)
    group.add_argument('--threshold_base',type=float,default=20.0)
    group.add_argument('--dp_type',type=str,choices=['word_pair','word_context'],default='word_pair')

    group = parser.add_argument_group('data pipeline options (debug)')
    group.add_argument('--parallel_map',type=int,default=1)
    group.add_argument('--trivial_data',action='store_true')
    group.add_argument('--word_counts',type=str,choices=['none','all','fast'],default='all')
    group.add_argument('--filtering',type=str,choices=['none','global','axis'],default='axis')
    group.add_argument('--rm_top',type=int,default=100)

################################################################################

def macro_input_variables(params):
    '''
    This function uses frame hacks to add a number of tensorflow variables
    to the calling location's global variable scope.
    This is very fragile, but I couldn't figure out a better way to get these variables easily shared between functions.
    '''
    import tensorflow as tf
    import sys
    f_globals = sys._getframe(1).f_globals
    sys._getframe(0).f_globals.update(f_globals)

    import model.common
    labels_names=model.common.data2labels(params.data)
    labels_num=[len(axis) for axis in labels_names ]
    labels_table=[tf.contrib.lookup.index_table_from_tensor(axis) for axis in labels_names]
    num_axes=len(labels_names)

    # create vocab hash
    vocab_index,vocab_words,vocab_counts=model.common.get_vocab_index(params.vocab_size,params.data)
    f_globals['vocab_index']=vocab_index
    f_globals['vocab_words']=vocab_words
    f_globals['vocab_counts']=vocab_counts
    f_globals['vocab_size']=len(vocab_counts)

    with tf.variable_scope('input_pipeline',reuse=tf.AUTO_REUSE):

        f_globals['num_lines']=tf.get_variable(
            name='num_lines',
            initializer=tf.zeros([],dtype=tf.int64),
            dtype=tf.int64,
            trainable=False,
            use_resource=True,
            )

        f_globals['num_words']=tf.get_variable(
            name='num_words',
            initializer=tf.zeros([],dtype=tf.int64),
            dtype=tf.int64,
            trainable=False,
            use_resource=True,
            )

        f_globals['num_words_filtered']=tf.get_variable(
            name='num_words_filtered',
            initializer=tf.zeros([],dtype=tf.int64),
            dtype=tf.int64,
            trainable=False,
            use_resource=True,
            )

        f_globals['word_counts']=tf.get_variable(
            name='word_counts',
            initializer=tf.zeros([vocab_size],dtype=tf.int64),
            dtype=tf.int64,
            trainable=False,
            use_resource=True,
            )

        f_globals['word_counts_axis']=[]
        f_globals['num_words_axis']=[]
        for axis in range(num_axes):
            f_globals['word_counts_axis'].append(tf.get_variable(
                name='word_counts_axis_'+str(axis),
                initializer=tf.zeros([vocab_size,labels_num[axis]],dtype=tf.int64),
                dtype=tf.int64,
                trainable=False,
                use_resource=True,
                ))
            f_globals['num_words_axis'].append(tf.get_variable(
                name='num_words_axis_'+str(axis),
                initializer=tf.zeros([labels_num[axis]],dtype=tf.int64),
                dtype=tf.int64,
                trainable=False,
                use_resource=True,
                ))

def input_fn(params):
    """
    WARNING:
    This function contains many subtle hacks designed to ensure that the Unicode
    decoding works and the strings are properly labelled with their country of origin.
    Modify carefully, constantly checking the work with `print_data.py`
    """
    import tensorflow as tf
    with tf.device('/cpu:0'):

        # create a trivial dataset
        if params.trivial_data:
            dataset = tf.data.Dataset.zip((
                #tf.data.Dataset.from_tensors(tf.constant([0,0],dtype=tf.int64)),
                #tf.data.Dataset.from_tensors(tf.constant([0],dtype=tf.int64)),
                tf.data.Dataset.from_tensor_slices(tf.constant([[1,0],[1,1],[0,0],[0,1]],dtype=tf.int64)),
                tf.data.Dataset.from_tensor_slices(tf.constant([[0],[1]],dtype=tf.int64)),
                )).repeat()
            dataset = dataset.batch(params.batch_size)
            return dataset

        # get input files
        import os
        files = [file for file in os.listdir(params.data) if os.path.isfile(os.path.join(params.data, file))]
        filenames=[os.path.join(params.data,file) for file in files]

        # get label information
        import model.common
        labels_names=model.common.data2labels(params.data)
        labels_num=[len(axis) for axis in labels_names ]
        labels_table=[tf.contrib.lookup.index_table_from_tensor(axis) for axis in labels_names]
        num_axes=len(labels_names)

        # load files into data pipeline
        def filename2labels(filename):
            basename=tf.string_split([filename],'/').values[-1]
            tokens=tf.string_split([basename],'-').values
            labels=[]
            for i in range(num_axes):
                labels.append(labels_table[i].lookup(tokens[i]))
            labels=tf.stack(labels,axis=0)
            return labels
        dataset=tf.data.Dataset.from_tensor_slices(filenames)
        dataset=dataset.interleave(
            lambda x: tf.data.Dataset.zip((
                tf.data.TextLineDataset([x]),
                tf.data.Dataset.from_tensors(filename2labels(x)).repeat(),
                )),
            cycle_length=len(filenames),
            block_length=1
            )
        dataset = dataset.shuffle(params.shuffle_lines,seed=params.shuffle_seed)

        # create variables/summaries for monitoring progress
        macro_input_variables(params)

        tf.summary.scalar('num_lines',num_lines,family='progress')
        tf.summary.scalar('num_words',num_words,family='progress')
        tf.summary.scalar('num_words_filtered',num_words_filtered,family='progress')
        filtered_ratio=tf.cast(num_words_filtered,tf.float32)/tf.cast(num_words,tf.float32)
        tf.summary.scalar('filtered_ratio',filtered_ratio,family='progress')
        for i in [1,10,100,1000,10000,100000]:
            tf.summary.scalar(
                'words_seen_'+str(i),
                tf.count_nonzero(tf.maximum(tf.zeros([],dtype=tf.int64),word_counts-i+1)),
                family='progress',
                )
        word_freq=tf.cast(word_counts,tf.float32)/tf.cast(1+num_words,tf.float32)
        tf.summary.histogram(
            'word_freq',
            word_freq,
            family='progress'
            )
        #NOTE: The 'progress_axis_XXX' summaries below don't contain any information
        # not already contained in the 'progress' summaries.
        #for axis in range(num_axes):
            #for i in [1,10,100,1000,10000,100000]:
                #tf.summary.scalar(
                    #'words_seen_'+str(i),
                    #tf.count_nonzero(tf.maximum(tf.zeros([],dtype=tf.int64),word_counts-i+1)),
                    #family='progress_axis_'+str(axis),
                    #)
            #label_freq=tf.cast(num_words_axis[axis],tf.float32)/tf.cast(1+num_words,tf.float32)
            #tf.summary.histogram(
                #'label_freq',
                #label_freq,
                #family='progress_axis_'+str(axis),
                #)


        # convert string into skipgram input
        SKIPGRAM_PAD=-2
        #SKIPGRAM_PAD='PAD'
        def line2wordpairs(line,labels):
            global word_counts
            global word_counts_axis
            global num_lines
            global num_words
            global num_words_axis
            global num_words_filtered

            # tokenize input line
            tokens=tf.string_split([line],' ').values
            tokens_ids=vocab_index.lookup(tokens)

            # filter tokens_ids that have been seen too frequently
            if params.filtering=='none' or params.word_counts=='none':
                tokens_filtered=tokens_ids
            else:
                t=params.threshold_base/params.vocab_size
                if params.filtering=='global' or params.word_counts=='fast':
                    numerator=tf.gather(word_counts,tokens_ids)
                    denominator=num_words
                    freq=tf.cast(numerator,tf.float32)/tf.cast(denominator,tf.float32)
                elif params.filtering=='axis':
                    t=params.threshold_base/params.vocab_size
                    freqs=[]
                    for axis in range(num_axes):
                        numerator=tf.gather(word_counts_axis[axis][:,labels[axis]],tokens_ids)
                        denominator=num_words_axis[axis][labels[axis]]
                        freqs.append(tf.cast(numerator,tf.float32)/tf.cast(denominator,tf.float32))
                    freq=tf.reduce_min(freqs)
                keep_prob=tf.sqrt(t/freq)
                rand=tf.random_uniform([tf.size(tokens_ids)],seed=params.shuffle_seed)
                if params.rm_top<0:
                    filter_mask=tf.greater(keep_prob,rand)
                else:
                    filter_mask=tf.logical_and(
                        tf.greater(keep_prob,rand),
                        tf.greater(tokens_ids,params.rm_top),
                        )
                tokens_filtered=tf.boolean_mask(tokens_ids,filter_mask)

            # create word_context
            if params.dp_type=='word_context':
                words=tf.expand_dims(tokens_filtered,axis=1)
                context=tf.stack([
                    tf.manip.roll(tokens_filtered,i,0)
                    for i in
                    list(range(-params.context_size,0))+list(range(1,params.context_size+1))
                    ],axis=1)
                #print('words=',words)
                #print('context=',context)

                # FIXME: add filtering?
                words_filtered=words
                context_filtered=context

            # create word pairs
            elif params.dp_type=='word_pair':
                tokens_padded=tf.pad(
                    tokens_filtered,
                    tf.constant([[params.context_size,params.context_size]],shape=[1,2]),
                    constant_values=SKIPGRAM_PAD
                    )
                context=tf.concat([
                    tf.manip.roll(tokens_padded,i,0)
                    for i in
                    list(range(-params.context_size,0))+list(range(1,params.context_size+1))
                    ],axis=0)
                words=tf.concat([tokens_padded for i in range(params.context_size+params.context_size)],axis=0)

                # filter wordpairs containing SKIPGRAM_PAD
                ids=tf.logical_and(
                    tf.not_equal(context,SKIPGRAM_PAD),
                    tf.not_equal(words,SKIPGRAM_PAD),
                    )
                context_filtered=tf.expand_dims(tf.boolean_mask(context,ids),axis=1)
                words_filtered=tf.expand_dims(tf.boolean_mask(words,ids),axis=1)

            # update all variable counters
            if params.word_counts=='none':
                update_ops=[]
            else:
                update_num_lines=tf.assign_add(num_lines,1)
                update_num_words=tf.assign_add(
                    num_words,
                    tf.size(tokens,out_type=tf.int64)
                    )
                update_num_words_filtered=tf.assign_add(
                    num_words_filtered,
                    tf.size(tokens_filtered,out_type=tf.int64)
                    )
                update_word_counts=tf.scatter_update(
                    word_counts,
                    tokens_ids,
                    tf.gather(word_counts,tokens_ids)+1
                    )
                update_axis=[]
                if params.word_counts=='all':
                    for axis in range(num_axes):
                        labels_tiled=tf.tile(
                            tf.reshape(labels[axis],[1]),
                            [tf.size(tokens_ids)],
                            )
                        indices=tf.stack([tokens_ids,labels_tiled],axis=1)
                        word_counts_axis_old=tf.gather_nd(word_counts_axis[axis],indices)
                        update_axis.append(tf.scatter_nd_update(
                            word_counts_axis[axis],
                            indices,
                            word_counts_axis_old+1
                            ))
                        update_axis.append(tf.scatter_add(
                            num_words_axis[axis],
                            tf.expand_dims(labels[axis],axis=0),
                            tf.size(tokens,out_type=tf.int64)
                            ))
                update_ops=[update_num_lines,update_num_words,update_word_counts]+update_axis

            with tf.control_dependencies(update_ops):
                return (context_filtered,words_filtered,labels)

        dataset=dataset.map(line2wordpairs,num_parallel_calls=params.parallel_map)
        dataset=dataset.flat_map(lambda x,y,z:tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(tf.concat([x,y],axis=1)),
            tf.data.Dataset.from_tensors(z).repeat(),
            )))

        # finalize dataset
        dataset=dataset.shuffle(params.shuffle_words,seed=params.shuffle_seed)
        dataset=dataset.batch(params.batch_size)
        dataset=dataset.repeat()
        dataset=dataset.prefetch(1)
        return dataset

################################################################################
def macro_model_variables(params):
    '''
    This function uses frame hacks to add a number of tensorflow variables
    to the calling location's global variable scope.
    This is very fragile, but I couldn't figure out a better way to get these variables easily shared between functions.
    '''
    import tensorflow as tf
    import sys
    f_globals = sys._getframe(1).f_globals
    sys._getframe(0).f_globals.update(f_globals)

    import model.common
    labels_names=model.common.data2labels(params.data)
    labels_num=[len(axis) for axis in labels_names ]
    labels_table=[tf.contrib.lookup.index_table_from_tensor(axis) for axis in labels_names]
    num_axes=len(labels_names)
    f_globals['labels_num']=labels_num
    f_globals['num_axes']=num_axes

    vocab_index,vocab_words,vocab_counts=model.common.get_vocab_index(params.vocab_size,params.data)
    f_globals['vocab_words']=vocab_words
    f_globals['vocab_counts']=vocab_counts
    f_globals['vocab_size']=len(vocab_counts)

    num_embeddings=params.num_embeddings
    if num_embeddings == []:
        num_embeddings = [1 for axis in range(num_axes)]
    f_globals['num_embeddings']=num_embeddings

    def my_get_variable(name,shape):
        if params.init_from_checkpoint is not None:
            #import numpy as np
            from tensorflow.python import pywrap_tensorflow
            latest_ckp = tf.train.latest_checkpoint(params.init_from_checkpoint)
            reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
            var_to_shape_map = reader.get_variable_to_shape_map()
            #var=reader.get_tensor('model/'+name)
            #for axis in range(len(var.shape)):
                #s1=var.shape[axis]
                #s2=shape[axis]
                #if not s1 == s2:
                    ##proj=np.random.randn(s1,s2)
                    #proj=np.ones([s1,s2])
                    #proj_norm=np.linalg.norm(proj,ord='fro')
                    #proj/=proj_norm
                    #var=np.tensordot(
                        #var,
                        #proj,
                        #axes=[[axis],[0]]
                        #)
                    # FIXME: if num_axes>1, do we need to do a transpose here?
            with tf.variable_scope('old'):
                oldshape=var_to_shape_map['model/'+name]
                old=tf.get_local_variable(name=name,shape=oldshape)
                #init=old.initialized_value()
                init=old.initial_value
                for axis in range(len(shape)):
                    if not oldshape[axis]==shape[axis]:
                        proj=tf.ones([oldshape[axis],shape[axis]])/tf.cast(oldshape[axis]*shape[axis],tf.float32)
                        init=tf.tensordot(
                            init,
                            proj,
                            axes=[[axis],[0]]
                            )
                        permutation=range(0,axis)+[len(shape)-1]+range(axis,len(shape)-1)
                        init=tf.transpose(init,permutation)

            return tf.get_variable(
                name=name,
                initializer=init, #.initialized_value(),
                )
        else:
            return tf.get_variable(
                name=name,
                shape=shape
                )


    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        def make_embedding(embedding):
            if embedding in params.disable_multi:
                f_globals[embedding]=tf.get_variable(
                    name=embedding,
                    shape=[vocab_size,params.embedding_size],
                    )
            else:
                # the low-dimensional representation of our embedding space
                f_globals[embedding+'_var']=my_get_variable(
                    name=embedding+'_var',
                    shape=[vocab_size,params.embedding_size]+num_embeddings,
                    )
                #if params.init_from_checkpoint is not None:
                    #from tensorflow.python import pywrap_tensorflow
                    #latest_ckp = tf.train.latest_checkpoint(params.init_from_checkpoint)
                    #reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
                    #init_embedding_var=reader.get_tensor('model/'+embedding) #+'_var')
                    #for axis in range(num_axes):
                        #init_embedding_var=tf.tensordot(
                            #init_embedding_var,
                            #tf.ones([init_embedding_var.shape[2],num_embeddings[axis]]),
                            #axes=[[2],[0]]
                            #)
                    #f_globals[embedding+'_var']=tf.get_variable(
                        #name=embedding+'_var',
                        #initializer=init_embedding_var
                        #)
                #else:
                    #f_globals[embedding+'_var']=tf.get_variable(
                        #name=embedding+'_var',
                        #shape=[vocab_size,params.embedding_size]+num_embeddings,
                        #)

                # tensors that rotate the embedding_var so that the embeddings
                # for each label are aligned
                # FIXME: I don't know how to write these equations
                f_globals[embedding]=f_globals[embedding+'_var']
                #if False:
                    #f_globals[embedding+'_rotator_axis']=[]
                    #for axis in range(num_axes):
                        #f_globals[embedding+'_rotator_axis'].append(tf.get_variable(
                            #name=embedding+'_rotator_axis_'+str(axis),
                            #shape=[params.embedding_size,params.embedding_size,num_embeddings[axis]],
                            #trainable=False
                            #))
                        #print('f_globals[embedding]=',f_globals[embedding])
                        #f_globals[embedding]=tf.matmul(
                            #tf.transpose(f_globals[embedding],[1,0,2]),
                            #f_globals[embedding+'_rotator_axis'][axis],
                            ##transpose_a=True
                            #)
                        #print('f_globals[embedding]=',f_globals[embedding])

                # the factors that project from the small tensor to the full tensor
                f_globals[embedding+'_projector_axis']=[]
                for axis in range(num_axes):
                    if num_embeddings[axis]>1:
                        f_globals[embedding+'_projector_axis'].append(my_get_variable(
                            name=embedding+'_projector_axis_'+str(axis),
                            shape=[num_embeddings[axis],labels_num[axis]],
                            ))
                        #f_globals[embedding+'_projector_axis'].append(tf.get_variable(
                            #name=embedding+'_projector_axis_'+str(axis),
                            #shape=[num_embeddings[axis],labels_num[axis]],
                            #))
                    else:
                        #f_globals[embedding+'_projector_axis'].append(None)
                        f_globals[embedding+'_projector_axis'].append(tf.get_variable(
                            name=embedding+'_projector_axis_'+str(axis),
                            initializer=tf.ones([num_embeddings[axis],labels_num[axis]]),
                            ))

        make_embedding('inputs')

        if not params.inputs_equals_outputs:
            make_embedding('outputs')
        else:
            for var in f_globals.keys():
                if var.startswith('inputs'):
                    var2='outputs'+var[6:]
                    f_globals[var2]=f_globals[var]
                    print('duplicating: ',var,' => ',var2)


def model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys
    params,   # Additional configuration
    ):
    import tensorflow as tf
    macro_model_variables(params)

    # create loss
    with tf.variable_scope('loss'):
        f0=features[:,:1]
        f0_size=4
        f0=tf.tile(f0,[1,f0_size])

        f1=features[:,1:]
        f1_size=f1.get_shape()[1]

        batch_size=features.get_shape()[0]

        # create negative samples
        nce_samples=min(params.nce_samples,vocab_size)
        if params.outputs_method=='per_dp':
            num_sampled=[batch_size,num_sampled]#FIXME: check
        else:
            num_sampled=nce_samples

        if params.sampler=='log_uniform':
            sampled_values=tf.nn.log_uniform_candidate_sampler(
                true_classes=f1,
                num_true=f1_size,
                num_sampled=num_sampled,
                unique=False,
                range_max=vocab_size,
            )
        elif params.sampler=='fixed_unigram':
            sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                true_classes=f1,
                num_true=f1_size,
                num_sampled=num_sampled,
                unique=False,
                range_max=vocab_size,
                unigrams=[x+1 for x in vocab_counts],
            )
        vocab_sampled, true_expected_count, sampled_expected_count = (
            tf.stop_gradient(s) for s in sampled_values)

        # calculate the logits of the labels in f0 and f1
        def multi2single(e,projector_axis):
            e_label=e
            for axis in range(num_axes):
                if num_embeddings[axis]==1:
                    e_label=tf.squeeze(e_label,axis=3)
                else:
                    e_proj=tf.tensordot(
                        projector_axis[axis],
                        e_label,
                        axes=[[0],[3]]
                        )
                    e_trans=tf.transpose(
                        e_proj,
                        [1,0,2]+list(range(3,3+num_axes-axis))
                        )
                    e_gather=tf.batch_gather(
                        e_trans,
                        tf.expand_dims(labels[:,axis],axis=1)
                        )
                    e_label=tf.squeeze(e_gather,axis=1)
            return e_label


        if 'inputs' in params.disable_multi:
            inputs_f0_label=tf.gather(inputs,f0)
        else:
            inputs_f0=tf.gather(inputs,f0)
            assert_shape(inputs_f0,[None,f0_size,params.embedding_size]+num_embeddings)
            inputs_f0_label=multi2single(inputs_f0,inputs_projector_axis)

        if 'outputs' in params.disable_multi:
            outputs_f1_label=tf.gather(outputs,f1)
        else:
            outputs_f1=tf.gather(outputs,f1)
            assert_shape(outputs_f1,[None,f1_size,params.embedding_size]+num_embeddings)
            outputs_f1_label=multi2single(outputs_f1,outputs_projector_axis)

        assert_shape(inputs_f0_label,[None,f0_size,params.embedding_size])
        assert_shape(outputs_f1_label,[None,f1_size,params.embedding_size])

        inputs_f0_label_reshape=tf.reshape(inputs_f0_label,[-1,f0_size,1,params.embedding_size])
        outputs_f1_label_reshape=tf.reshape(outputs_f1_label,[-1,1,f1_size,params.embedding_size])

        if f0_size==1:
            # FIXME: is _sum_rows faster?
            #logits1=_sum_rows(inputs_f0_label*outputs_f1_label)
            #logits1=tf.expand_dims(logits1,axis=1)
            logits1=tf.reduce_sum(inputs_f0_label_reshape*outputs_f1_label_reshape,axis=3)
        else:
            logits1_per_f0=[]
            for i in range(f0_size):
                logits1_per_f0.append(tf.reduce_sum(inputs_f0_label_reshape[:,i:i+1,0:1,:]*outputs_f1_label_reshape,axis=3))
            logits1=tf.concat(logits1_per_f0,axis=1)
        logits1-=tf.log(tf.reshape(true_expected_count,[-1,1,f1_size]))
        assert_shape(logits1,[None,f0_size,f1_size])

        # calculate the logits of the sampled labels
        if 'outputs' in params.disable_multi:
            outputs_sampled=tf.gather(outputs,vocab_sampled)
            logits2=tf.tensordot(inputs_f0_label,outputs_sampled,axes=[[2],[1]])
            logits2-=tf.log(sampled_expected_count)
            assert_shape(logits2,[None,f0_size,nce_samples])

        else:
            # FIXME: what is the best outputs_method?
            if params.outputs_method=='averaged':
                outputs_sampled=tf.gather(outputs,vocab_sampled)
                for axis in range(num_axes):
                    outputs_sampled=tf.reduce_sum(outputs_sampled,axis=2)
                logits2=tf.tensordot(inputs_f0_label,outputs_sampled,axes=[[2],[1]])
                logits2-=tf.log(sampled_expected_count)

            elif params.outputs_method=='per_label':
                outputs_sampled=tf.gather(outputs,vocab_sampled)
                print('outputs_sampled=',outputs_sampled)
                for axis in range(num_axes):
                    outputs_sampled=tf.tensordot(
                        outputs_sampled,
                        outputs_projector_axis[axis],
                        [[2],[0]]
                        )
                outputs_trans=tf.transpose(outputs_sampled,list(range(2,2+num_axes))+[0,1])
                outputs_labels=tf.gather_nd(outputs_trans,labels)
                print('outputs_sampled=',outputs_sampled)
                print('outputs_trans=',outputs_trans)
                print('outputs_labels=',outputs_labels)
                print('inputs_f0_label=',inputs_f0_label)
                print('labels=',labels)
                outputs_labels_reshape=tf.reshape(outputs_labels,[-1,1,nce_samples,params.embedding_size])
                inputs_f0_label_reshape=tf.reshape(inputs_f0_label,[-1,f0_size,1,params.embedding_size])
                logits2=tf.reduce_sum(inputs_f0_label_reshape*outputs_labels_reshape,axis=3)
                logits2-=tf.log(sampled_expected_count)
                print('logits2=',logits2)

        assert_shape(logits2,[None,f0_size,nce_samples])


        # calculate loss
        print('logits1=',logits1)
        print('logits2=',logits2)
        logits1=tf.reshape(logits1,[-1,f0_size*f1_size])
        logits2=tf.reshape(logits2,[-1,f0_size*nce_samples])
        logits=tf.concat([logits1,logits2],axis=1)
        nce_labels=tf.concat(
            [tf.ones_like(logits1)/tf.cast(f1_size,tf.float32),tf.zeros_like(logits2)],
            axis=1,
            )
        sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=nce_labels,
            logits=logits,
            name="sampled_losses"
            )
        loss_per_dp=_sum_rows(sampled_losses)/f0_size
        loss=tf.reduce_mean(loss_per_dp)

    # create regularization term
    with tf.variable_scope('regularization'):
        def regularize_multi(e):
            e_mean=tf.reduce_mean(e,axis=range(3,3+num_axes))
            e_mean_reshape=tf.reshape(e_mean,shape=[-1,1,params.embedding_size]+[1 for _ in range(num_axes)])
            e_diff=e_mean_reshape-e
            return tf.nn.l2_loss(e_diff)

        reg=0.0
        if params.reg>0:
            if not 'inputs' in params.disable_multi:
                reg_inputs=regularize_multi(inputs_f0)
                tf.summary.scalar('reg_inputs',reg_inputs,family='optimization')
                reg+=reg_inputs
            if not 'outputs' in params.disable_multi:
                reg_outputs=regularize_multi(outputs_f1)
                tf.summary.scalar('reg_outputs',reg_outputs,family='optimization')
                reg+=reg_outputs
            reg*=params.reg

    loss_regularized=loss+reg

    tf.summary.scalar('loss',loss,family='optimization')
    tf.summary.scalar('loss_regularized',loss_regularized,family='optimization')

    # create training op
    if mode==tf.estimator.ModeKeys.TRAIN:
        learning_rate=tf.train.polynomial_decay(
            params.learning_rate,
            tf.train.get_global_step(),
            decay_steps=params.learning_rate_decay,
            end_learning_rate=params.learning_rate/100,
            )
        tf.summary.scalar('learning_rate',learning_rate,family='optimization')

        if params.optimizer=='adam':
            optimizer=tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
        elif params.optimizer=='sgd':
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(loss_regularized, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            )

################################################################################

def assert_shape(tensor,shape):
    '''
    Uses magical stack hacks to crash the system when `tensor.get_shape()` doesn't match `shape`.
    Useful for documenting code with static assertions of tensor shapes.

    For implementation details, see:
    https://stackoverflow.com/questions/34175111/raise-an-exception-from-a-higher-level-a-la-warnings
    '''
    if tensor.get_shape().as_list()==shape:
        return
    else:
        import sys
        import traceback
        e=AssertionError('get_shape()='+str(tensor.get_shape().as_list())+' but asserted shape='+str(shape))
        print('Traceback (most recent call last):',file=sys.stderr)
        traceback.print_stack(sys._getframe().f_back)
        print(*traceback.format_exception_only(type(e),e),file=sys.stderr, sep="",end="")
        raise SystemExit(1)

################################################################################

# See: https://github.com/tensorflow/tensorflow/blob/93dd14dce2e8751bcaab0a0eb363d55eb0cc5813/tensorflow/python/ops/nn_impl.py#L1271
def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  import tensorflow as tf
  cols = tf.shape(x)[1]
  ones_shape = tf.stack([cols, 1])
  ones = tf.ones(ones_shape, x.dtype)
  return tf.reshape(tf.matmul(x, ones), [-1])
