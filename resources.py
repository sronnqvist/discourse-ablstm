"""
Functions handling resources for LSTM modeling of discourse relation senses
author: Samuel Ronnqvist <sronnqvi@abo.fi>
source: https://github.com/sronnqvist/discourse-ablstm
"""
import json
import gensim
import numpy as np
import collections

def read_senses(path, ignore_types=[]):
    """ Like read_relations, but returns only sense labels. """
    senses = []
    print ("Reading", path)
    for line in open(path+'relations.json'):
        anno_tokens = []
        relation = json.loads(line)
        ignore = False
        for ignoree in ignore_types:
            if ignoree in relation['Type']:
                ignore = True
        if ignore:
            continue
        senses.append(relation['Sense'])
    return senses


def read_relations(path, ignore_types=[], partial_sampling=False, with_syntax=False, with_context=False):
    """ Read relations JSON and combine with parses to produce
        annotated token sequences with relation sense labels.
        Annotations mark argument and connective spans. """

    print ("Reading", path)
    print ("Parses...")
    parses = json.load(open(path+'parses.json'))
    relations = []

    print ("Relations...")
    for line in open(path+'relations.json'):
        anno_tokens = []
        arg_pos = [[],[]]
        arg_dep = [[],[]]
        arg_depd = [[],[]]
        arg_tokens = [[],[]]
        relation = json.loads(line)
        ignore = False
        for ignoree in ignore_types:
            if ignoree in relation['Type']:
                ignore = True
        if ignore:
            continue

        # Get spans
        _, _, _, arg1_beg_sent, arg1_beg_token = relation['Arg1']['TokenList'][0]
        _, _, _, arg1_end_sent, arg1_end_token = relation['Arg1']['TokenList'][-1]
        _, _, _, arg2_beg_sent, arg2_beg_token = relation['Arg2']['TokenList'][0]
        _, _, _, arg2_end_sent, arg2_end_token = relation['Arg2']['TokenList'][-1]
        try:
            _, _, _, conn_beg_sent, conn_beg_token = relation['Connective']['TokenList'][0]
            _, _, _, conn_end_sent, conn_end_token = relation['Connective']['TokenList'][-1]
        except IndexError:
            conn_beg_sent = None
            conn_end_sent = None
            conn_beg_token = None
            conn_end_to = None

        in_span = False
        dep_roles = collections.defaultdict(lambda: '[DEP]null')
        dep_depth = collections.defaultdict(lambda: -1)
        arg_idx = 0
        for sent_i in range(arg1_beg_sent, arg2_end_sent+1):
            doc = parses[relation['DocID']]
            dep_roles.update(dict([(int(src.split('-')[-1])-1, role) for role, trg, src in doc['sentences'][sent_i]['dependencies']]))
            dep_depth.update(dict([(int(token.split('-')[-1])-1, depth+1) for token, depth in traverse(build_tree(doc['sentences'][sent_i]['dependencies']))]))
            for i, token in enumerate(doc['sentences'][sent_i]['words']):
                if sent_i == arg1_beg_sent and i == arg1_beg_token:
                    anno_tokens.append("<ARG1>")
                    arg_tokens[arg_idx].append("<ARG1>")
                    arg_pos[arg_idx].append("<ARG1>")
                    arg_dep[arg_idx].append("<ARG1>")
                    arg_depd[arg_idx].append(-2)
                    in_span = True
                elif sent_i == arg2_beg_sent and i == arg2_beg_token:
                    arg_idx = 1
                    anno_tokens.append("<ARG2>")
                    arg_tokens[arg_idx].append("<ARG2>")
                    arg_pos[arg_idx].append("<ARG2>")
                    arg_dep[arg_idx].append("<ARG2>")
                    arg_depd[arg_idx].append(-2)
                    in_span = True
                elif sent_i == conn_beg_sent and i == conn_beg_token:
                    anno_tokens.append("<CONN>")
                    arg_tokens[arg_idx].append("<CONN>")
                    arg_pos[arg_idx].append("<CONN>")
                    arg_dep[arg_idx].append("<CONN>")
                    arg_depd[arg_idx].append(-2)
                    in_span = True
                if with_context or in_span:
                    anno_tokens.append(token[0])
                    arg_tokens[arg_idx].append(token[0])
                    arg_pos[arg_idx].append(token[1]['PartOfSpeech'])
                    arg_dep[arg_idx].append(dep_roles[i])
                    arg_depd[arg_idx].append(dep_depth[i])
                if sent_i == arg1_end_sent and i == arg1_end_token:
                    anno_tokens.append("</ARG1>")
                    arg_tokens[arg_idx].append("</ARG1>")
                    arg_pos[arg_idx].append("</ARG1>")
                    arg_dep[arg_idx].append("</ARG1>")
                    arg_depd[arg_idx].append(-2)
                    in_span = False
                elif sent_i == arg2_end_sent and i == arg2_end_token:
                    anno_tokens.append("</ARG2>")
                    arg_tokens[arg_idx].append("</ARG2>")
                    arg_pos[arg_idx].append("</ARG2>")
                    arg_dep[arg_idx].append("</ARG2>")
                    arg_depd[arg_idx].append(-2)
                    in_span = False
                elif sent_i == conn_end_sent and i == conn_end_token:
                    anno_tokens.append("</CONN>")
                    arg_pos[arg_idx].append("</CONN>")
                    arg_dep[arg_idx].append("</CONN>")
                    arg_depd[arg_idx].append(-2)
                    in_span = False

        if partial_sampling:
            #if not with_syntax:
            relations.append((anno_tokens, relation['Sense'][0]))
            relations.append((anno_tokens, relation['Sense'][0]))
            relations.append((arg_tokens[0], relation['Sense'][0]))
            relations.append((arg_tokens[1], relation['Sense'][0]))
            #else:
            #    relations.append(((arg_tokens[0], arg_pos[0], arg_dep[0], arg_depd[0]), relation['Sense'][0]))
            #    relations.append(((arg_tokens[1], arg_pos[1], arg_dep[1], arg_depd[1]), relation['Sense'][0]))
        #elif with_syntax:
        #    relations.append(((anno_tokens, arg_pos[0]+arg_pos[1], arg_dep[0]+arg_dep[1], arg_depd[0]+arg_depd[1]), relation['Sense'][0]))
        else:
            relations.append((anno_tokens, relation['Sense'][0]))

    print (len(relations), "read")
    print ()
    return relations


def get_vectors(vocab, token2id, path):
    """ Read pre-trained gensim/word2vec vectors """
    print ("Reading word vectors...")
    try:
        vectors = gensim.models.word2vec.Word2Vec.load(path)
    except:
        try:
            vectors = gensim.models.word2vec.Word2Vec.load_word2vec_format(path, binary=False)
        except:
            vectors = gensim.models.word2vec.Word2Vec.load_word2vec_format(path, binary=True)

    vocab_dim = len(vectors[list(vectors.wv.vocab.keys())[0]])
    nsymbols = len(vocab) + 1
    embedding_weights = np.zeros((nsymbols+1,vocab_dim))
    print ("Mapping word vectors...")
    for word, index in token2id.items():
        if word in vectors:
            embedding_weights[index,:] = vectors[word]
        elif word.lower() in vectors:
            embedding_weights[index,:] = vectors[word.lower()]
        else:
            embedding_weights[index,:] = vectors[list(vectors.wv.vocab.keys())[0]]*0
    return embedding_weights


def build_tree(dependencies):
    tree = collections.defaultdict(lambda: [])
    for rel, parent, child in dependencies:
        tree[parent].append(child)
    return tree


def traverse(tree, node='ROOT-0', depth=0):
    tokens = []
    if node not in tree:
        node = node.lower()
    for child in tree[node]:
        tokens.append((child, depth))
        tokens += traverse(tree, child, depth+1)
    return tokens
