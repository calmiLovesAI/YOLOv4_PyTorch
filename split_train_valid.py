import random

from configuration import Config


def write_list_to_file(a, filename):
    with open(file=filename, mode="a+", encoding="utf-8") as f:
        f.writelines(a)


if __name__ == '__main__':
    with open(file=Config.txt_file_dir, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        len_dataset = len(lines)
    print("Train : valid = {} : {}".format(1 - Config.valid_ratio, Config.valid_ratio))
    num_valid = int(Config.valid_ratio * len_dataset)
    valid_list = random.sample(lines, num_valid)
    train_list = [x for x in lines if x not in valid_list]

    write_list_to_file(a=train_list, filename=Config.train_txt)
    write_list_to_file(a=valid_list, filename=Config.valid_txt)
