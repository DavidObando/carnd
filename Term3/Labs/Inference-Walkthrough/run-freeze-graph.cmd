python -m tensorflow.python.tools.freeze_graph --input_graph=base_graph.pb --input_checkpoint=ckpt --input_binary=true --output_graph=frozen_graph.pb --output_node_names=Softmax
