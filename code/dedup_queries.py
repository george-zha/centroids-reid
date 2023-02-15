import os

verkada_query = '/home/georgez/datasets/verkada_data/query/'
new_query = '/home/georgez/datasets/verkada_data/query_dedup/'
os.mkdir(new_query)

seen = {}
for i in os.listdir(verkada_query):
    l = i.split("_")
    pid = l[0]

    if pid in seen:
        seen[pid] += 1 
    else:
        seen[pid] = 1

    if seen[pid] > 5:
        continue
    else:
        os.system('cp ' + verkada_query + i + ' ' + new_query + i)