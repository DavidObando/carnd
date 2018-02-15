python -m tensorflow.python.tools.optimize_for_inference --input=frozen_graph.pb --output=optimized_graph.pb --frozen_graph=True --input_names=image_input --output_names=Softmax
