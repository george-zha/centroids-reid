import numpy as np
import os

ar = np.load('vis_verkada/results.npy', allow_pickle=True)
results = ar.item()

counter = 0
for key in results:
    nn = results[key]['paths'][1]
    result_id = nn.split("/")[-1].split("_")[0]
    query_id = key.split("/")[-1].split("_")[0]
    if query_id != result_id:
        
        if not os.path.exists(key):
            continue
        if not os.path.exists(nn):
            continue
        os.makedirs(os.path.join("visualise_fp",str(counter)))
        try:
            os.system(f"cp {key} visualise_fp/{counter}/query_{query_id}")
            os.system(f"cp {nn} visualise_fp/{counter}/result_{result_id}")
        except:
            import ipdb; ipdb.set_trace()
        counter+=1
    if counter>=100:
        break




