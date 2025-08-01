import onnx
from onnx import helper, numpy_helper, TensorProto
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser
    parser.add_argument('--input_model', type=str, required=True, help='Path to the input ONNX model')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save the modified ONNX model')
    args = parser.parse_args()
    input_model = args.input_model
    output_model = args.output_model

    # 1. 加载模型
    model = onnx.load(input_model)
    graph = model.graph

    # 2. 获取输出
    output = graph.output[0]
    output_name = output.name

    # 2.1 获取原始 shape
    # 注意：output.type.tensor_type.shape.dim 是一个 list，每个元素有 dim_value
    orig_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]

    # 3. 定义 perm
    perm = [0, 2, 1]

    # 3.1 计算新 shape
    new_shape = [orig_shape[i] for i in perm]

    # 4. 创建Transpose节点
    transpose_output_name = output_name + '_transposed'
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[output_name],
        outputs=[transpose_output_name],
        perm=perm
    )
    graph.node.append(transpose_node)

    # 5. 修改模型的输出为Transpose节点的输出
    new_output = helper.make_tensor_value_info(
        transpose_output_name,
        TensorProto.FLOAT,  # 注意类型要和原输出一致
        new_shape
    )
    graph.output.remove(output)
    graph.output.append(new_output)

    # 6. 保存新模型
    onnx.save(model, output_model)
