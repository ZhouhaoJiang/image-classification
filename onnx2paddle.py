from x2paddle import convert

# 调用 x2paddle 的函数来转换模型
convert.onnx2paddle(model_path='./onnx_model/chat.onnx',
                    save_dir='pd_model',
                    convert_to_lite=True,
                    lite_valid_places="arm",
                    lite_model_type="naive_buffer")
