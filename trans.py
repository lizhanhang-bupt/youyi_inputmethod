input_file = 'input.txt'  # 原始中文txt文件
output_file = 'output.txt'  # 保存为train.txt文件

# 读取文件并将内容保存为UTF-8编码
try:
    with open(input_file, 'r', encoding='gbk', errors='ignore') as f:  # 假设原文件是gbk编码
        content = f.read()

    # 将内容以UTF-8编码保存到新文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
except Exception as e:
    print(f"发生错误: {e}")
