import tensorflow as tf
import numpy as np
from poems_train import process_poems, rnn_model

start_token = 'B'
end_token = 'E'
model_dir = 'model/'
corpus_file = 'poems.txt'

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem():
    batch_size = 1
    words, dict_words, vector = process_poems(corpus_file)
    input_data = tf.Variable(tf.int32, [batch_size, None])
    end_points = rnn_model(input_data, None, vector)

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(end_points)
        sess.reset(sess, checkpoint)

        x = np.array([list(map(dict_words, start_token))])
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x})
        word = to_word(predict, vector)
        poem = ''
        i = 0
        while word != end_token:
            poem += word
            i += 1
            if i > 24:
                break
            x =np.zeros((1, 1))
            x[0, 0] = dict_words[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vector)
        return poem

def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


if __name__ == '__main__':
    # begin_char = input('## please input the first character:')
    poem = gen_poem()
    pretty_print_poem(poem=poem)
