# import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_text as text
#
# bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
#
# map_name_to_handle = {
#     'bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
#     'bert_en_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
#     'bert_multi_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
#     'small_bert/bert_en_uncased_L-2_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-2_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-2_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-2_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-4_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-4_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-4_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-4_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-6_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-6_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-6_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-6_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-8_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-8_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-8_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-8_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-10_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-10_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-10_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-10_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
#     'small_bert/bert_en_uncased_L-12_H-128_A-2':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
#     'small_bert/bert_en_uncased_L-12_H-256_A-4':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
#     'small_bert/bert_en_uncased_L-12_H-512_A-8':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
#     'small_bert/bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
#     'albert_en_base':
#         'https://tfhub.dev/tensorflow/albert_en_base/2',
#     'electra_small':
#         'https://tfhub.dev/google/electra_small/2',
#     'electra_base':
#         'https://tfhub.dev/google/electra_base/2',
#     'experts_pubmed':
#         'https://tfhub.dev/google/experts/bert/pubmed/2',
#     'experts_wiki_books':
#         'https://tfhub.dev/google/experts/bert/wiki_books/2',
#     'talking-heads_base':
#         'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
# }
#
# map_model_to_preprocess = {
#     'bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'bert_en_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-2_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-4_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-6_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-8_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-10_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-128_A-2':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-256_A-4':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-512_A-8':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'small_bert/bert_en_uncased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'bert_multi_cased_L-12_H-768_A-12':
#         'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
#     'albert_en_base':
#         'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
#     'electra_small':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'electra_base':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'experts_pubmed':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'experts_wiki_books':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
#     'talking-heads_base':
#         'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
# }
#
# tfhub_handle_encoder = map_name_to_handle[bert_model_name]
# tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
#
# print(f'BERT model selected           : {tfhub_handle_encoder}')
# print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
# bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
# bert_model = hub.KerasLayer(tfhub_handle_encoder)
#
#
# def build_bert_model():
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#     preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
#     encoder_inputs = preprocessing_layer(text_input)
#     encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
#     outputs = encoder(encoder_inputs)
#     net = outputs['pooled_output']
#     net = tf.keras.layers.Dropout(0.1)(net)
#     net = tf.keras.layers.Dense(6, activation="softmax", name='classifier')(net)
#     return tf.keras.Model(text_input, net)
