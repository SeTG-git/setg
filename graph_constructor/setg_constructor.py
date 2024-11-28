import json
import pandas as pd
from util.logger import logger
from collections import defaultdict
import os
from datetime import datetime


raw_labels = []
app_names = []
how_many_deuedroid = {'Financial':0,  'Gambling':0,  'Porn':0}


def process_transitions(json_file_path):
   csv_file_path = 'setg_' + json_file_path.split('/')[-1].replace('.json', '.csv')
   csv_file_path = os.path.join('/'.join(json_file_path.split('/')[:-1]),csv_file_path )
   with open(json_file_path, 'r') as f:
       data = json.load(f)
   df_semantics = pd.read_csv(csv_file_path.replace('.json', '.csv'))
   new_transitions = []
   for transition in data['transitions']:
        src = transition['scr']
        dest = transition['dest']
        src_semantics = df_semantics[df_semantics['activity_name'] == src]['semantics'].values[0] if len(df_semantics[df_semantics['activity_name'] == src]['semantics'].values) > 0 else None
        dest_semantics = df_semantics[df_semantics['activity_name'] == dest]['semantics'].values[0] if len(df_semantics[df_semantics['activity_name'] == dest]['semantics'].values) > 0 else None
        if len(src_semantics) > 0 or len(dest_semantics) > 0:
            new_transitions.append({
                'scr': src_semantics,
                'dest': dest_semantics,
            })
   return new_transitions


def pre_process_goodware(root,foldn='Normal'):
    subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
    tss = []
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        json_file_path = os.path.join(folder, folder_name + '.json')
        if os.path.exists(json_file_path):
            transitions = process_transitions(json_file_path)
            if transitions == '' or len(transitions) ==0 :
                continue
            tss.append(transitions)
            if foldn != 'Normal':
                app_names.append('default')
                raw_labels.append(foldn)
                how_many_deuedroid[foldn] += 1
    return tss

def pre_process(apps_info, md5_set, root):
    tss = []
    raw_l1 = []
    for md5 in md5_set:
        res = apps_info[apps_info['file_md5']==md5]['related_label']
        if res.empty:
            rl = 'default'
            app_names.append('default')
        else:
            rl = res
            ap = apps_info[apps_info['file_md5']==md5]['app_name']
            app_names.append(ap)
        json_file_path = os.path.join(root,md5+'.json')
        transitions = process_transitions(json_file_path)
        tss.append(transitions)
        raw_l1.append(rl)
    return tss, raw_l1


def tss2adj(tss_allware):
    all_ware_adj = []
    all_raw_labels = []
    i = 0
    for cate, tss in tss_allware.items():
        for ts in tss:
            rl = 'Normal'
            app_name = 'Default'
            adject_list = defaultdict(list)
            for t in ts:
                src, dest = t['scr'], t['dest']
                adject_list[src].append(dest)
            if cate == 'goodware':
                all_raw_labels.append('Normal')
            else:
                all_raw_labels.append(raw_labels[i])
                rl = raw_labels[i]
                app_name = app_names[i]
                i+= 1
            all_ware_adj.append((adject_list, cate, rl,app_name))
    logger.info(f"len(all_ware_adj), all_ware_adj[0]:{len(all_ware_adj), all_ware_adj[0]}")
    return all_ware_adj

def cls_text(text):
    candidate = "defaultdict(list,{"
    cleaned_text = text.strip(candidate)
    return cleaned_text

def adj2df(all_ware_adj):
    df = {'Text':[],'category':[],'raw_labels':[],'app_name':[]}
    for i in all_ware_adj:
        adject_list, cate, raw,an= i
        df['Text'].append(cls_text(str(adject_list)))
        df['category'].append(cate)
        df['raw_labels'].append(raw)
        df['app_name'].append(an)
    return df

def tss2df(tss_allware):
    return pd.DataFrame(adj2df(tss2adj(tss_allware)))

def _helper_deue(flag_scam = False):
    if flag_scam:
        tss_scamware = []
        fold_name = ['Financial',  'Gambling',  'Porn']
        for foldn in fold_name:
            scamware_root = f'/mnt/inside_15T/PPG_dataset/DeUEDroid_result/{foldn}/'
            tss_scamware.append(pre_process(scamware_root,foldn))
        return tss_scamware
    else:
        return pre_process_goodware('/mnt/inside_15T/PPG_dataset/DeUEDroid_result/Normal')

def construct_tss(apps_info_g, apps_info_s):
    apps_info = pd.read_csv(apps_info_g)
    apps_info = pd.read_csv(apps_info_s)
    tss_goodware = pre_process_goodware('/mnt/inside_15T/PPG_dataset/DescribeCTX_result/')
    tss_scamware, _ = pre_process(apps_info, apps_info[['file_md5']],  "/mnt/maldetect_NAS/aibot/oppo/oppo/native_stat_json_res")
    tss_goodware.extend(_helper_deue(False))
    tss_scamware.extend(_helper_deue(True))
    tss_allware = {'goodware': tss_goodware,
                'scamware': tss_scamware}
    return tss_allware

def main():
    apps_info_g = '/home/aibot/workspace/SquiDroidAgent/gensetg/data/goodware4setg.csv'
    apps_info_s = '/home/aibot/workspace/SquiDroidAgent/gensetg/data/scamware4SeTG.csv'
    tss_allware = construct_tss(apps_info_g, apps_info_s)
    df = tss2df(tss_allware)
    print(len(df),df.head(), )
    save_path = f'model_input_adjacency_{datetime.now().strftime("%H_%M_%S")}.csv'
    df.to_csv(save_path, index=False)

def test_case():
    raw_labels.append("E-commerce")
    app_names.append("青春计步管家")
    jf = "/home/aibot/workspace/SquiDroidAgent/gensetg/data/1b76d42256ec8e9ce6b699aeef85fbbb.json"
    # sf = "/home/aibot/workspace/SquiDroidAgent/gensetg/data/setg_1b76d42256ec8e9ce6b699aeef85fbbb.csv"
    transitions = process_transitions(jf)
    tcts = []
    tcts.append(transitions)
    df = tss2df({'scamware': tcts})
    print(len(df),df.head(), )
    save_path = f'/home/aibot/workspace/SquiDroidAgent/gensetg/data/test_case_adjacency_{datetime.now().strftime("%H_%M_%S")}.csv'
    df['label']=[1]
    df.to_csv(save_path, index=False,encoding='utf-8-sig')

if __name__ ==  "__main__":
    test_case()