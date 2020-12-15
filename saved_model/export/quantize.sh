vai_q_tensorflow quantize \
        --input_frozen_graph=./frozen_graph.pb \
        --input_nodes=Placeholder_2 \
        --input_shapes=?,128,128,3 \
        --output_nodes=conv2d_transpose_3/conv2d_transpose \
        --input_fn=image_input_fn.calib_input \
        --output_dir=quantized \
        --calib_iter=100
