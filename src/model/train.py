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
    group.add_argument('--learning_rate',type=float,default=0.25)
    group.add_argument('--learning_rate_decay',type=float,default=1e8)
    group.add_argument('--nce_samples',type=int,default=2**8)
    group.add_argument('--sampler',type=str,choices=['log_uniform','fixed_unigram'],default='fixed_unigram')
    group.add_argument('--reg',type=float,default=0.0)
    group.add_argument('--zero_init',type=str,choices=['inputs','outputs'],nargs='*',default=[])
    group.add_argument('--f0mod',type=str,choices=['true','false'],default='false')

    group = parser.add_argument_group('data pipeline options')
    group.add_argument('--data',type=str,required=True)
    group.add_argument('--vocab_size',type=int,default=2**16)
    group.add_argument('--shuffle_lines',type=int,default=2**14)
    group.add_argument('--shuffle_words',type=int,default=2**14)
    group.add_argument('--shuffle_seed',type=int,default=0)
    group.add_argument('--context_size',type=int,default=5)
    group.add_argument('--threshold_base',type=float,default=20.0)
    group.add_argument('--dp_type',type=str,choices=['word_pair','word_context','sentence'],default='word_pair')
    group.add_argument('--sample_method',type=str,choices=['ns','nce'],default=['ns'])
    group.add_argument('--reuse_samples',type=str,choices=['true','false'],default='true')

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
    vocab_size=len(vocab_counts)
    f_globals['vocab_index']=vocab_index
    f_globals['vocab_words']=vocab_words
    f_globals['vocab_counts']=vocab_counts
    f_globals['vocab_size']=vocab_size

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

        # convert string into skipgram input
        SKIPGRAM_PAD=-2
        #SKIPGRAM_PAD='PAD'
        def line2wordpairs(line,label):
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
                        numerator=tf.gather(word_counts_axis[axis][:,label[axis]],tokens_ids)
                        denominator=num_words_axis[axis][label[axis]]
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

            # convert tokens into dp_type
            if params.dp_type=='sentence':
                words=tf.expand_dims(tokens_filtered,axis=0)
                context=tf.expand_dims(tokens_filtered,axis=0)
                raise ValueError('dp_type==sentence not yet implemented')

            elif params.dp_type=='word_context':
                words=tf.expand_dims(tokens_filtered,axis=1)
                context=tf.stack([
                    tf.manip.roll(tokens_filtered,i,0)
                    for i in
                    list(range(-params.context_size,0))+list(range(1,params.context_size+1))
                    ],axis=1)

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
                context=tf.expand_dims(tf.boolean_mask(context,ids),axis=1)
                words=tf.expand_dims(tf.boolean_mask(words,ids),axis=1)

            # create negative samples
            nce_samples=min(params.nce_samples,vocab_size)
            def candidate_sampler(pos,pos_size):
                if params.sampler=='log_uniform':
                    sampled_values=tf.nn.log_uniform_candidate_sampler(
                        true_classes=pos,
                        num_true=pos_size,
                        num_sampled=nce_samples,
                        unique=False,
                        range_max=vocab_size,
                    )
                elif params.sampler=='fixed_unigram':
                    sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                        true_classes=pos,
                        num_true=pos_size,
                        num_sampled=nce_samples,
                        unique=False,
                        range_max=vocab_size,
                        unigrams=[x+1 for x in vocab_counts],
                    )
                return sampled_values

            if params.reuse_samples=='true':
                pos=context
                pos_size=pos.get_shape()[1]
                sampled_values=candidate_sampler(pos,pos_size)
                samples, true_expected_count, sampled_expected_count = (
                    tf.stop_gradient(s) for s in sampled_values)
                samples=tf.tile(tf.expand_dims(samples,dim=0),[tf.shape(words)[0],1])
                sampled_expected_count=tf.tile(tf.expand_dims(sampled_expected_count,dim=0),[tf.shape(words)[0],1])

            else:
                words_size=tf.shape(words)[0]
                context_size=context.get_shape()[1]
                def body(a,b,c):
                    a_size=tf.shape(a)[0]
                    pos=context[a_size:a_size+1,:]
                    pos_size=pos.get_shape()[1]
                    sampled_values=candidate_sampler(pos,pos_size)
                    vocab_sampled, true_expected_count, sampled_expected_count = sampled_values
                    return [
                        tf.concat([a,tf.expand_dims(vocab_sampled,axis=0)],axis=0),
                        tf.concat([b,true_expected_count],axis=0),
                        tf.concat([c,tf.expand_dims(sampled_expected_count,axis=0)],axis=0),
                        ]

                sampled_values = tf.while_loop(
                    lambda a,b,c: tf.less(tf.shape(a)[0],words_size), 
                    body, 
                    loop_vars=[
                        tf.zeros([0,nce_samples],dtype=tf.int64),
                        tf.zeros([0,context_size],dtype=tf.float32),
                        tf.zeros([0,nce_samples],dtype=tf.float32),
                        ],
                    shape_invariants=[
                        tf.TensorShape([None,nce_samples]),
                        tf.TensorShape([None,context_size]),
                        tf.TensorShape([None,nce_samples]),
                        ]
                    )
                samples, true_expected_count, sampled_expected_count = (
                    tf.stop_gradient(s) for s in sampled_values)

            allsamples=tf.concat([context,samples],axis=1)
            expected_count=tf.concat([true_expected_count,sampled_expected_count],axis=1)

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
                        label_tiled=tf.tile(
                            tf.reshape(label[axis],[1]),
                            [tf.size(tokens_ids)],
                            )
                        indices=tf.stack([tokens_ids,label_tiled],axis=1)
                        word_counts_axis_old=tf.gather_nd(word_counts_axis[axis],indices)
                        update_axis.append(tf.scatter_nd_update(
                            word_counts_axis[axis],
                            indices,
                            word_counts_axis_old+1
                            ))
                        update_axis.append(tf.scatter_add(
                            num_words_axis[axis],
                            tf.expand_dims(label[axis],axis=0),
                            tf.size(tokens,out_type=tf.int64)
                            ))
                update_ops=[update_num_lines,update_num_words,update_word_counts]+update_axis

            with tf.control_dependencies(update_ops):
                labels=tf.tile(tf.expand_dims(label,axis=0),[tf.shape(words)[0],1])
                features={
                    'words':words,
                    'allsamples':allsamples,
                    'labels':labels,
                    }

                # adding expected_count to the features dict causes a slowdown,
                # so only do it if it will be used by the estimator
                if params.sample_method=='nce':
                    features['expected_count']=expected_count

                return features

        dataset=dataset.map(line2wordpairs,num_parallel_calls=params.parallel_map)
        dataset=dataset.flat_map(lambda *xs: tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(x) for x in xs)))
        dataset=tf.data.Dataset.zip((dataset,tf.data.Dataset.from_tensors(1).repeat()))

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

    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        def make_embedding(embedding):

            def my_get_variable(name,shape):
                if embedding in params.zero_init:
                    return tf.get_variable(
                        name=name,
                        initializer=tf.zeros(shape),
                        )
                else:
                    return tf.get_variable(
                        name=name,
                        shape=shape,
                        )

            if embedding in params.disable_multi:
                f_globals[embedding]=my_get_variable(
                    name=embedding,
                    shape=[vocab_size,params.embedding_size],
                    )
            else:
                # the low-dimensional representation of our embedding space
                f_globals[embedding+'_var']=my_get_variable(
                    name=embedding+'_var',
                    shape=[vocab_size,params.embedding_size]+num_embeddings,
                    )

                # NOTE: this extra variable exists so that in the future it can be replaced
                # with a rotation to align the embeddings
                f_globals[embedding]=f_globals[embedding+'_var']

                # the factors that project from the small tensor to the full tensor
                f_globals[embedding+'_projector_axis']=[]
                for axis in range(num_axes):
                    if num_embeddings[axis]>1:
                        f_globals[embedding+'_projector_axis'].append(my_get_variable(
                            name=embedding+'_projector_axis_'+str(axis),
                            shape=[num_embeddings[axis],labels_num[axis]],
                            ))
                    else:
                        f_globals[embedding+'_projector_axis'].append(tf.get_variable(
                            name=embedding+'_projector_axis_'+str(axis),
                            initializer=tf.ones([num_embeddings[axis],labels_num[axis]]),
                            ))

        make_embedding('inputs')

        if not params.inputs_equals_outputs:
            make_embedding('outputs')
            if 'outputs' in params.disable_multi:
                f_globals['bias']=tf.get_variable(
                    name='bias',
                    shape=[vocab_size],
                    )
            else:
                f_globals['bias']=tf.get_variable(
                    name='bias',
                    shape=[vocab_size,1]+num_embeddings,
                    )
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
        features=features[0]
        labels=features['labels'] # FIXME: `labels` should have a more descriptive name
        f0=features['words']
        f0_size=f0.get_shape()[1]

        samples=features['allsamples']
        samples_size=samples.get_shape()[1]

        pos_size=params.context_size*2
        neg_size=samples_size-pos_size

        # calculate the logits of the labels in f0 and pos
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
            outputs_samples_label=tf.gather(outputs,samples)
            bias_samples_label=tf.expand_dims(tf.gather(bias,samples),axis=1)
        else:
            outputs_samples=tf.gather(outputs,samples)
            assert_shape(outputs_samples,[None,samples_size,params.embedding_size]+num_embeddings)
            outputs_samples_label=multi2single(outputs_samples,outputs_projector_axis)
            bias_samples=tf.gather(bias,samples)
            bias_samples_label=tf.expand_dims(tf.squeeze(multi2single(bias_samples,outputs_projector_axis),axis=2),axis=1)

        assert_shape(inputs_f0_label,[None,f0_size,params.embedding_size])
        assert_shape(outputs_samples_label,[None,samples_size,params.embedding_size])

        if params.f0mod=='true':
            inputs_f0_label=tf.expand_dims(tf.reduce_mean(inputs_f0_label,axis=1),axis=1)
            f0_size=1

        logits=tf.einsum('imk,ink->imn',inputs_f0_label,outputs_samples_label)+bias_samples_label
        if params.sample_method=='nce':
            logits-=tf.log(tf.expand_dims(features['expected_count'],axis=1))
        assert_shape(logits,[None,f0_size,samples_size])

        # calculate loss
        logits1=logits[:,:,:pos_size]
        logits2=logits[:,:,pos_size:]
        nce_labels=tf.concat(
            [tf.ones_like(logits1)/tf.cast(pos_size,tf.float32),tf.zeros_like(logits2)],
            axis=2,
            )
        sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=nce_labels,
            logits=logits,
            name="sampled_losses"
            )
        loss_per_dp=tf.reduce_sum(sampled_losses,axis=[1,2])/tf.cast(f0_size,tf.float32)
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
                reg_outputs=regularize_multi(outputs_samples)
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
    tshape=tensor.get_shape().as_list()
    if tshape==shape or (shape[0] is None and tshape[1:]==shape[1:]):
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
# FIXME: Many reduce_sum calls above can be replaced by _sum_rows,
# but it's not clear that the increased speed would be worth the complexity
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
