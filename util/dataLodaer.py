import pandas as pd
import os
from util.logger import logger


def load_data(path, used_fp):
    res = {}
    try:
        json_fps = os.listdir(path)
        used = pd.read_csv(used_fp)['md5'].to_list()
        used = set(used)
        for jfp in json_fps:
            k = jfp.split('/')[-1].replace('.json','')
            if k in used:
                res[k] = os.path.join(path, jfp)
    except:
        logger.warning("[setg-semantic generation warning]  empyt res")
    return res
