import os
import chardet

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10KB内容检测
        result = chardet.detect(raw_data)
        return result['encoding']

def convert_file(input_path, output_path):
    """单个文件转换"""
    try:
        # 检测原始编码
        encoding = detect_encoding(input_path)

        # 读取内容
        with open(input_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()

        # 写入UTF-8
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        #print(f"成功转换: {input_path} -> {output_path}")
        return True

    except Exception as e:
        #print(f"转换失败 {input_path}: {str(e)}")
        return False

def batch_convert(input_path, output_path):
    """批量转换"""
    if os.path.isfile(input_path):
        convert_file(input_path, output_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                convert_file(os.path.join(root, file), output_path)
    #else:
        #print(f"路径不存在: {input_path}")

if __name__ == "__main__":
    # 设定输入和输出路径
    input_path = 'train_data.txt'  # 输入文件路径
    output_path = 'training_data.txt'  # 输出文件路径

    # 直接进行文件转换
    batch_convert(input_path, output_path)
