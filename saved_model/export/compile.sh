vai_c_tensorflow \
       --frozen_pb=./quantized/deploy_model.pb \
       --arch=/workspace/ultra96v2/ultra96v2_vitis_flow_tutorial_1/ULTRA96V2.json \
       --output_dir=compiled \
       --net_name=fpn_resnet18 