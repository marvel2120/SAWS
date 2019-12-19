# # # remove the sentence less than 5 tokens
# # # remove duplicate samples
# # # remove special marks
# # # transfer all tokens into lowercase
# # # remove the words only appear once


def preprocess_data(data_file):
    contents = []
    word_count = {}
    with open(data_file) as f:
        for line in f.readlines():
            sp = line.lower().split()
            if len(sp) < 6:
                continue
            contents.append(sp)
    print("total samples：")
    print(len(contents))
    new_contents = []
    for i in contents:
        if i not in new_contents:
            new_contents.append(i)
    print("remove duplicate samples：")
    print(len(new_contents))
    # record the words only appear once
    for _ in new_contents:
        for word in _[2:]:
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1
    remove_word = []
    for i in word_count.keys():
        if word_count[i] == 1:
            remove_word.append(i)
    print("remove word：", remove_word)
    for i in range(len(new_contents)):
        for j in range(len(new_contents[i])):
            if new_contents[i][j] in remove_word:
                new_contents[i][j] = "UNK"
            if len(new_contents[i][j]) != 1:
                new_contents[i][j] = new_contents[i][j].strip("#")
    for i in new_contents:
        k = ' '.join([str(j) for j in i])
        f.write(k + "\n")
    f.close()
