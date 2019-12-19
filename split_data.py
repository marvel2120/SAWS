import random

path = "data/ghosh/"
name = path + "clean_data.txt"


def split_data():
    """
    Randomly split the data into train and test set
    :return:None
    """
    train = path + "train.txt"
    test = path + "test.txt"
    with open(name) as f:
        contents = []
        for line in f.readlines():
            sp = line.strip().split()
            contents.append(sp)
    f.close()
    random.shuffle(contents)
    index = int(len(contents)*0.9)
    f1 = open(train, 'w')
    for i in contents[:index]:
        k = " ".join([str(j) for j in i])
        f1.write(k + "\n")
    f1.close()
    f2 = open(test, 'w')
    for i in contents[index:]:
        k = " ".join([str(j) for j in i])
        f2.write(k + "\n")
    f2.close()

split_data()
