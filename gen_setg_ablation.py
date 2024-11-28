# %%
import os
import pandas as pd
import json
import shutil
import numpy as np
from tqdm import tqdm
import json
import openai
from util.dataLodaer import load_data
from util.logger import logger


llm_kwargs = {
    "llm_model": "qwen-max", 
              "max_length": 6000,
              "temperature": 1.0,
    "top_p": 1.0
}

def request_gpt_model_in_new_thread(
    analysis_prompt,
    llm_kwargs,
    chatbot,
    history=None,
    sys_prompt="",
    key_index=0
):
    if history is None:
        history = []
    
    client = openai.OpenAI(
        base_url="http://10.176.46.12:5000/v1", 
        api_key="sk-no-key-required"  
    )

    max_length = llm_kwargs.get("max_length", 6000) 

    messages = [{"role": "system", "content": sys_prompt}]
    
    for entry in history:
        if isinstance(entry, str):
            messages.append({"role": "user", "content": entry})
        else:
            messages.append({"role": "user", "content": str(entry)})

    messages.append({"role": "user", "content": analysis_prompt})

    total_length = sum(len(msg["content"]) for msg in messages)

    if total_length > max_length:
        available_length = max_length - sum(len(msg["content"]) for msg in messages[:-1])
        analysis_prompt = analysis_prompt[:available_length]
        messages[-1]["content"] = analysis_prompt

    completion = client.chat.completions.create(
        model=llm_kwargs.get("llm_model", "gpt-3.5-turbo"),
        messages=messages,
        max_tokens=llm_kwargs.get("max_length", 6000),  
        temperature=llm_kwargs.get("temperature", 1.0),  
        top_p=llm_kwargs.get("top_p", 1.0)  
    )

    response = completion.choices[0].message
    return response.content


def analyze_activity(activity, src_activities_desc=""):
    '''
        # 请提供以下信息：
        # 1. 这个活动可能的用途
        # 2. 布局的可能结构
        # 3. 如何与前置活动（如果有）进行连接和交互
        # 4. 可能的用户交互流程
    '''
    chatbot = []
    # analysis_prompt = f"""
    # 分析以下活动并提供见解：
    # 活动名称：{activity['name']}
    # 布局信息：{activity['layouts']}
    # {f'前置活动描述：{src_activities_desc}' if src_activities_desc else ''}
    # """
    analysis_prompt_no_cot = f"""
    分析以下活动并提供见解：

    活动名称：{activity['name']}
    布局信息：{activity['layouts']}

    {f'前置活动描述：{src_activities_desc}' if src_activities_desc else ''}
    
    请用一句话描述该activity的功能。例如：“splashActivity页面功能是启动这个健康APP”
    """
    analysis_result = request_gpt_model_in_new_thread(
        analysis_prompt_no_cot,
        llm_kwargs,
        chatbot,
        history=[],
        sys_prompt="你是一位经验丰富的移动应用程序分析师，专门分析应用程序结构和流程。",
        key_index=0
    )
    return analysis_result

def ana_json(json_data):
    activity_insights = {}
    activity_sources = {activity['name']: [] for activity in json_data['activities']}
    for transition in json_data['transitions']:
        if transition['dest'] not in activity_sources or transition['scr'] not in activity_sources:
            continue
        activity_sources[transition['dest']].append(transition['scr'] for activity in json_data['activities'])
    analyzed_activities = set()

    def analyze_activity_recursive(activity_name):
        insight = None
        if activity_name in analyzed_activities:
            return activity_insights[activity_name]

        if activity_name in activity_sources:
            src_activities = activity_sources[activity_name]
            src_desc = ""
            if src_activities:
                tres = []
                for src in src_activities:
                    t = analyze_activity_recursive(src)
                    t = "" if t is None else t
                    tres.append(t)
                src_desc = "; ".join(tres)
        for act in json_data['activities']:
            if not act['name'] == activity_name:
                continue
            activity = act
            insight = analyze_activity(activity, src_desc)
            activity_insights[activity_name] = insight
            analyzed_activities.add(activity_name)
            print(f"已分析 {activity_name}:")
            logger.debug(f"已分析 {activity_name}:")
            print(insight)
            logger.debug(insight)
            print("\n" + "="*50 + "\n")
            logger.debug("\n" + "="*50 + "\n")

        return insight

    df = 0
    for activity in tqdm(json_data['activities']):
        analyze_activity_recursive(activity['name'])

    return activity_insights

def save_to_file(ana_res, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_no_cot', fn='setg_default.csv'):
    sfn = os.path.join(sp, fn)
    hd = ['activity_name', 'semantics']
    df = pd.DataFrame.from_dict(ana_res, orient='index', columns=['semantics'])
    df['activity_name'] =df.index
    df = df[['activity_name', 'semantics']]
    print(f'df: {df}')
    df.to_csv(sfn,encoding='utf-8-sig',header=hd,index=False)


def main():
    d_dir = "/mnt/maldetect_NAS/aibot/oppo/oppo"
    jfp = os.path.join(d_dir, "native_stat_json_res")
    ufp = os.path.join(d_dir, "scamware4SeTG.csv")
    finished_dir = './setg'
    finished_name = 'finish_name.txt'
    fd = os.path.join(finished_dir, finished_name)
    ana_scam_j = load_data(jfp,fd, ufp)
    generated = os.listdir('/home/aibot/workspace/SquiDroidAgent/data/setg_no_cot')
    for k in ana_scam_j:
        if ('setg_'+k+'_4.csv') in generated:
            continue
        logger.debug(f'gen setg_{k} to csv file')
        p = ana_scam_j[k]
        print(p)
        json_data = {}
        with open(p,mode="r",encoding="utf-8") as f:
            json_data = json.load(f)
        for i in range(5):
            if ('setg_'+k+f'_{i}.csv') in generated:
                continue
            ana_res = ana_json(json_data)
            save_to_file(ana_res, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_no_cot', fn=f'setg_{k}_{i}.csv')
            logger.debug(f'save setg_{k}_{i} to setg succ.')

        save_to_file(ana_res, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_no_cot', fn=f'setg_{k}.csv')
        logger.debug(f'save setg_{k} to setg succ.')

if __name__ == "__main__":
    save_to_file({'k':'v','k2':'v2','k3':'v3'}, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_no_cot', fn=f'setg_no_cot.csv')
    main()