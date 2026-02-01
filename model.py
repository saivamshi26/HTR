import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from dataloader_iam import Batch

# Disable eager mode
tf.compat.v1.disable_eager_execution()


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model:
    def __init__(self,
                 char_list: List[str],
                 decoder_type: int = DecoderType.BestPath,
                 must_restore: bool = False,
                 dump: bool = False):

        self.dump = dump
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_ID = 0

        self.is_train = tf.compat.v1.placeholder(tf.bool, name="is_train")
        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        self.sess, self.saver = self.setup_tf()

    # ================= CNN =================
    def setup_cnn(self):
        cnn_in4d = tf.expand_dims(self.input_imgs, axis=3)

        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]

        pool = cnn_in4d
        for i in range(len(pool_vals)):
            kernel = tf.Variable(
                tf.random.truncated_normal(
                    [kernel_vals[i], kernel_vals[i],
                     feature_vals[i], feature_vals[i + 1]],
                    stddev=0.1)
            )
            conv = tf.nn.conv2d(pool, kernel, strides=(1, 1, 1, 1), padding="SAME")
            conv = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            conv = tf.nn.relu(conv)
            pool = tf.nn.max_pool2d(
                conv,
                ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                strides=(1, pool_vals[i][0], pool_vals[i][1], 1),
                padding="VALID"
            )

        self.cnn_out_4d = pool

    # ================= RNN =================
    def setup_rnn(self):
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=2)

        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden) for _ in range(2)]
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)

        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            stacked, stacked, rnn_in3d, dtype=rnn_in3d.dtype
        )

        concat = tf.concat([fw, bw], axis=2)
        concat = tf.expand_dims(concat, axis=2)

        kernel = tf.Variable(
            tf.random.truncated_normal(
                [1, 1, num_hidden * 2, len(self.char_list) + 1],
                stddev=0.1)
        )

        self.rnn_out_3d = tf.squeeze(
            tf.nn.atrous_conv2d(concat, kernel, rate=1, padding="SAME"),
            axis=2
        )

    # ================= CTC =================
    def setup_ctc(self):
        self.ctc_in_3d_tbc = tf.transpose(self.rnn_out_3d, [1, 0, 2])
        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])

        self.gt_texts = tf.SparseTensor(
            tf.compat.v1.placeholder(tf.int64, [None, 2]),
            tf.compat.v1.placeholder(tf.int32, [None]),
            tf.compat.v1.placeholder(tf.int64, [2])
        )

        self.loss = tf.reduce_mean(
            tf.compat.v1.nn.ctc_loss(
                labels=self.gt_texts,
                inputs=self.ctc_in_3d_tbc,
                sequence_length=self.seq_len,
                ctc_merge_repeated=True
            )
        )

        if self.decoder_type == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(
                inputs=self.ctc_in_3d_tbc,
                sequence_length=self.seq_len,
                beam_width=50,
                top_paths=10   # 🔥 KEY FIX
            )
        else:
            self.decoder = tf.nn.ctc_greedy_decoder(
                self.ctc_in_3d_tbc, self.seq_len
            )

    # ================= TF INIT =================
    def setup_tf(self):
        print("Python:", sys.version)
        print("TensorFlow:", tf.__version__)

        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver(max_to_keep=1)

        model_dir = "../model/"
        snapshot = tf.train.latest_checkpoint(model_dir)

        if self.must_restore and not snapshot:
            raise Exception("No model found")

        if snapshot:
            print("Restoring:", snapshot)
            saver.restore(sess, snapshot)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

    # ================= DECODER =================
    def decoder_output_to_text(self, decoded, batch_size):
        results = [[] for _ in range(batch_size)]

        for path in decoded[0]:  # iterate top_paths
            batch_texts = [[] for _ in range(batch_size)]
            for i, (b, t) in enumerate(path.indices):
                batch_texts[b].append(self.char_list[path.values[i]])

            for b in range(batch_size):
                results[b].append("".join(batch_texts[b]))

        return results

    # ================= INFER =================
    def infer_batch(self, batch: Batch, *_):
        feed = {
            self.input_imgs: batch.imgs,
            self.seq_len: [batch.imgs[0].shape[0] // 4],
            self.is_train: False
        }

        decoded = self.sess.run(self.decoder, feed)
        texts = self.decoder_output_to_text(decoded, len(batch.imgs))

        # flatten single image case
        return texts[0], None
