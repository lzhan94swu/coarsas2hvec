import os
import argparse
import math
import multiprocessing as mp
import gc
from collections import defaultdict, Counter

import numpy as np
import tqdm
from numpy import random
from six import iteritems

# import tensorflow as tf

# from gensim.models.keyedvectors import Vocab

from utils import *
from coarsen import *
from snsample import *
from eva import test


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/ACM',
                        help='Input dataset path')

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of epoch. Default is 50.')

    parser.add_argument('--batch-size', type=int, default=256,
                        help='Number of batch_size. Default is 512.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--negative-samples', type=int, default=10,
                        help='Negative samples for optimization. Default is 5.')

    parser.add_argument('--coarsen-rate', default=0.2,
                        help='Proportion of the removed nodes in coarsening. Default is 0.3')

    parser.add_argument('--coarsen-times', default=3,
                        help='Rounds of coarsen. Default is 3')

    parser.add_argument('--center-type', default='p',
                        help='Fixed type of sampling')

    return parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

def get_batches(pairs, batch_size, vocab, node_types):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size
    result = []
    for idx in range(n_batches):
        x, xt, y, yt = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(vocab[pairs[index][0]].index)
            xt.append(node_types[pairs[index][0]])
            y.append(vocab[pairs[index][1]].index)
            yt.append(node_types[pairs[index][1]])
        # result.append(np.array([x, y]).astype(np.int32))
        result.append(
            [np.array(x).astype(np.int32).reshape(-1, 1).astype(np.int32), \
            np.array(xt).astype(np.int32).reshape(-1, 1).astype(np.int32), \
            np.array(y).astype(np.int32).reshape(-1, 1).astype(np.int32), \
            np.array(yt).astype(np.int32).reshape(-1, 1).astype(np.int32)])
    return result

def generate_vocab(context):
    index2word = []
    raw_vocab = defaultdict(int)
    for word in context:
        raw_vocab[word] += 1
    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)
    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i
    return vocab, index2word

if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)

    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    num_sampled = args.negative_samples
    multi = True
    walk_length = int(args.walk_length)
    num_walks = int(args.num_walks)
    window_size = int(args.window_size)
    coar_t = int(args.coarsen_times)
    coar_r = float(args.coarsen_rate)
    training_data, all_nodes = load_training_data(file_name + '/train.txt')
    node_types, type_vocabs, type_nodes = load_node_type(file_name + '/node_type.txt')

    type_neighbor = build_neighbor(node_types, training_data)
    coar_type_neighbor = type_neighbor

    num_nodes = len(all_nodes)

    center_type = type_vocabs[args.center_type]
    type_walk_nums = None
    Ori_Graph = nx.Graph()
    Ori_Graph.add_edges_from(training_data)

    Coarsener = GraphCoarsener(node_types=node_types,
                                type_nodes=type_nodes,
                                coarsen_rate=coar_r)

    Coar_Graph = Ori_Graph
    node_pairs = []

    if coar_t > 1:
        round_sample_times = int(np.ceil(args.walk_length//(coar_t)))
        coar_round = coar_t
    elif coar_t == 1:
        round_sample_times = int(np.ceil(args.walk_length//2))
        coar_round = coar_t
    else:
        round_sample_times = args.walk_length
        coar_round = 1
    sampled_times = 1

    candidates = []
    context = []
    for t in range(coar_round):
        round_sample_times = round_sample_times * (0.5 ** t)
        if round_sample_times >= 1:
            sub_sample_times = round_sample_times
        else:
            sub_sample_times = 1
        for j in range(sub_sample_times):
        # for i in range(1):
            if sampled_times <= walk_length:
                print('Start sampling {} times'.format(sampled_times))
                Sns = SNSampler(Ori_Graph=Ori_Graph,
                                Samp_Graph=Coar_Graph,
                                context=context,
                                node_types=node_types,
                                num_walks=num_walks,
                                window_size=window_size,
                                type_walk_nums=type_walk_nums,
                                type_vocabs=type_vocabs,
                                type_neighbor=type_neighbor,
                                coar_type_neighbor=coar_type_neighbor)
                if not multi:
                    tmp_pairs, tmp_context = Sns.sn_sample(Coar_Graph, candidates)
                else:
                    num_cores = int(mp.cpu_count())
                    pool = mp.Pool(num_cores)
                    cut = int(np.ceil(num_nodes / num_cores))
                    candi_nodes = {}
                    Ori_candi_nodes = {}
                    for i in range(num_cores):
                        candi_nodes[i] = list(Coar_Graph.nodes())[i*cut: i*cut+cut-1]
                        Ori_candi_nodes[i] = candidates[i*cut: i*cut+cut-1]
                    results = [pool.apply_async(Sns.sn_sample_multi, args=(candi_nodes[candi], Ori_candi_nodes[candi],)) for candi in range(num_cores)]
                    results = [p.get() for p in results]
                    tmp_pairs = [i for x in results for i in x[0]]
                    tmp_context = [i for x in results for i in x[1]]
                    pool.close()
                    pool.join()
                node_pairs.append(tmp_pairs)
                context += tmp_context
                if j == 0:
                    context += list(Ori_Graph.nodes())
                sampled_times += 1
                gc.collect()
            else:
                break

        if len(Sns.import_prob) > 0:
            node_importance = [t[0] for t in sorted(Sns.import_prob.items(),
                        key=lambda x:x[1], reverse=True) if t[0] in Coar_Graph.nodes()]
        else:
            node_importance = [t[0] for t in sorted(Sns.degree_prob.items(),
                        key=lambda x:x[1], reverse=True) if t[0] in Coar_Graph.nodes()]

        if t+1 <= coar_t:
            print('Start coarsening for round {}'.format(t+1))
            Coar_Graph, coar_candidates = Coarsener.coarsen(Coar_Graph, node_importance, center_type=center_type)
            candidates = candidates + coar_candidates
            coar_type_neighbor = build_neighbor(node_types, Coar_Graph.edges())

    for i in range(walk_length-sampled_times+1):
        print('Start sampling {} times'.format(sampled_times))
        Sns = SNSampler(Ori_Graph=Ori_Graph,
                        Samp_Graph=Coar_Graph,
                        context=context,
                        node_types=node_types,
                        num_walks=num_walks,
                        window_size=window_size,
                        type_walk_nums=type_walk_nums,
                        type_vocabs=type_vocabs,
                        type_neighbor=type_neighbor,
                        coar_type_neighbor=coar_type_neighbor)
        if not multi:
            tmp_pairs, tmp_context = Sns.sn_sample([], Ori_Graph)
            gc.collect()
        else:
            pool = mp.Pool(num_cores)
            cut = int(np.ceil(num_nodes / num_cores))+1
            candi_nodes = {}
            for i in range(num_cores):
                candi_nodes[i] = []
                Ori_candi_nodes[i] = candidates[i*cut: i*cut+cut-1]
            results = [pool.apply_async(Sns.sn_sample_multi, args=(candi_nodes[candi], Ori_candi_nodes[candi],)) for candi in range(num_cores)]
            results = [p.get() for p in results]
            tmp_pairs = [i for x in results for i in x[0]]
            tmp_context = [i for x in results for i in x[1]]
            pool.close()
            pool.join()

        node_pairs.append(tmp_pairs)
        context += tmp_context
        sampled_times += 1
        # break
    
    import tensorflow as tf

    from gensim.models.keyedvectors import Vocab
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    np.random.seed(0)
    graph = tf.Graph()
    with graph.as_default():
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0), )

        nce_weight = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros(num_nodes))
        type_embeddings = tf.Variable(tf.truncated_normal([len(type_vocabs)**2, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))

        train_inputs = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None, 1])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        label_types = tf.placeholder(tf.int32, shape=[None, 1])

        pos_type_pair = len(type_vocabs)*train_types+label_types
        embeddings = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        pos_type = tf.nn.embedding_lookup(type_embeddings, pos_type_pair)
        pos_embed = tf.multiply(embeddings, tf.sigmoid(pos_type))
        pos_embed = tf.squeeze(pos_embed)

        loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights = nce_weight,
                    biases = nce_biases,
                    labels = train_labels,
                    inputs = pos_embed,
                    num_sampled = num_sampled,
                    num_classes = num_nodes,
                )
            )
        optimizer = tf.train.AdamOptimizer(0.025).minimize(loss)
        init = tf.global_variables_initializer()

    all_node_pairs = [i for x in node_pairs for i in x]
    vocab, index2word = generate_vocab(context)
    assert len(vocab) == num_nodes, 'Not Enough Context'

    pair_count = Counter(all_node_pairs)
    pair_count = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        print('Training')
        ep_loss = []
        for epoch in range(epochs):
            if multi:
                pool = mp.Pool(num_cores)
                cut = int(np.ceil(len(all_node_pairs) / num_cores))
                candi_pairs = {}
                for i in range(num_cores):
                    candi_pairs[i] = all_node_pairs[i::num_cores]
                for candi in range(num_cores):
                    pool.apply_async(random.shuffle, args=(candi_pairs[candi]))
                indices = list(candi_pairs.keys())
                random.shuffle(indices)
                all_node_pairs = [i for x in indices for i in candi_pairs[x]]
                pool.close()
                pool.join()

            else:
                random.shuffle(all_node_pairs)

            batches = get_batches(all_node_pairs, batch_size, vocab, node_types)
            gc.collect()

            data_iter = tqdm.tqdm(enumerate(batches),
                                    desc="EP:%d" % (epoch+1),
                                    ascii=True,
                                    total=len(batches),
                                    bar_format="{l_bar}{bar}{r_bar}")
            avg_loss = 0.0
            for i, data in data_iter:
                feed_dict = {train_inputs: data[0], \
                            train_types: data[1], \
                            train_labels: data[2], \
                            label_types: data[3], \
                }
                _, loss_value = sess.run([optimizer, loss], feed_dict)
                ep_loss.append(loss_value)

                avg_loss += loss_value

                if i % 500 == 0:
                    post_fix = {
                        "epoch": epoch+1,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))

            final_embeddings = {}
            for j in range(num_nodes):
                final_embeddings[index2word[j]] = np.array(sess.run(embeddings, {train_inputs: [[j]]})[0])

            with open('./embeddings/' + file_name.split('/')[-1] + '.embeddings', 'w') as emb_file:
                for node in final_embeddings.keys():
                    emb_file.writelines(str(node) + ' ' + ' '.join(str(it) for it in final_embeddings[node][0].tolist()) + '\n')