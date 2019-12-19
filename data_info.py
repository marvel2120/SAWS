from collections import Counter


# Get data information
def get_data_info(data_path):
    sent_max_len = 0
    avg_len = 0
    labels = []
    stat = dict()
    stat[10] = 0
    stat[20] = 0
    stat[30] = 0
    stat[40] = 0
    stat[50] = 0
    stat["other"] = 0
    with open(data_path) as f:
        for line in f.readlines():
            sp = line.strip().split()
            labels.append(sp[-1])
            avg_len += len(sp[:-1])
            if len(sp[:-1]) > sent_max_len:
                sent_max_len = len(sp[:-1])
            if len(sp[:-1]) < 10:
                stat[10] += 1
            elif len(sp[:-1]) > 10 and len(sp[:-1]) < 20:
                stat[20] += 1
            elif len(sp[:-1]) > 20 and len(sp[:-1]) < 30:
                stat[30] += 1
            elif len(sp[:-1]) > 30 and len(sp[:-1]) < 40:
                stat[40] += 1
            elif len(sp[:-1]) > 40 and len(sp[:-1]) < 50:
                stat[50] += 1
            else:
                stat["other"] += 1
    print("Data information: ")
    print("Total samples：", len(labels))
    print("Max sentence length：", sent_max_len)
    print("Avg sentence length：", avg_len / len(labels))
    counter = Counter(labels)
    print("Label distribution：")
    for w in counter:
        print(w, counter[w])
    print(stat)


# get_data_info("data/Mishra/clean_data.txt")
# get_data_info("data/ghosh/test.txt")
