import os


def create_class_list():
    train_data_dir = ''
    class_list = os.listdir(train_data_dir)
    number = len(class_list)
    with open('class_idx.txt', 'w') as f:
        for idx, classes in enumerate(class_list):
            if idx != (number - 1):
                f.write(classes + '\n')
            else:
                f.write(classes)


if __name__ == '__main__':
    create_class_list()