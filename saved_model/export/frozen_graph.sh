freeze_graph \
    --input_meta_graph  ../trained.meta \
    --input_checkpoint  ../trained \
    --output_graph      ./frozen_graph.pb \
    --output_node_names conv2d_transpose_3/conv2d_transpose \
    --input_binary      true