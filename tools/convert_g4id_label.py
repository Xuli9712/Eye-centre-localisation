def convert_txt_to_csv(file_path):
    csv_file_path = file_path.replace('.txt', '.csv')

    with open(file_path, 'r') as txt_file, open(csv_file_path, 'w') as csv_file:
        for line in txt_file:
            line = line.replace('\t', ',')  # 替换制表符为逗号
            line = line.replace(' ', ',')  # 替换空格为逗号
            line = line.rstrip(',')  # 删除行尾多余的逗号
            line += '\n'  # 重新添加换行符
            csv_file.write(line)

    print(f"转换完成！CSV 文件已保存为: {csv_file_path}")

if __name__ == "__main__":
    # 指定要转换的文本文件路径
    txt_file_path = r'D:\gi4e_database\image_labels.txt'

    # 调用函数进行转换
    convert_txt_to_csv(txt_file_path)
