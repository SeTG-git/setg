# %%
import os
import pandas as pd
import json
import shutil
import numpy as np
from tqdm import tqdm
import json
from util.dataLodaer import load_data
from util.logger import logger
import openai



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
    
    # Initialize the OpenAI client
    client = openai.OpenAI(
        base_url="http://10.176.46.12:5000/v1",  # Base URL of your API server
        api_key="sk-no-key-required"  # API key (in this case, it's not required)
    )

    # Extract max_length from llm_kwargs
    max_length = llm_kwargs.get("max_length", 6000)  # Default to 6000 if not specified

    # Create the prompt messages based on the input parameters
    messages = [{"role": "system", "content": sys_prompt}]
    
    # Process each history entry and add it to the messages
    for entry in history:
        if isinstance(entry, str):
            messages.append({"role": "user", "content": entry})
        else:
            # If entry is not a string, convert it to string (handle formatted strings)
            messages.append({"role": "user", "content": str(entry)})

    # Append the user's current input (analysis_prompt)
    messages.append({"role": "user", "content": analysis_prompt})

    # Calculate total message length
    total_length = sum(len(msg["content"]) for msg in messages)

    # Ensure total length does not exceed max_length
    if total_length > max_length:
        # Adjust the analysis_prompt to fit within the max_length
        available_length = max_length - sum(len(msg["content"]) for msg in messages[:-1])
        analysis_prompt = analysis_prompt[:available_length]
        messages[-1]["content"] = analysis_prompt

    # Send the request to the model
    completion = client.chat.completions.create(
        model=llm_kwargs.get("llm_model", "gpt-3.5-turbo"),
        messages=messages,
        max_tokens=llm_kwargs.get("max_length", 6000),  # Optional: to further control token output
        temperature=llm_kwargs.get("temperature", 1.0),  # Use temperature from llm_kwargs
        top_p=llm_kwargs.get("top_p", 1.0)  # Use top_p from llm_kwargs
    )

    # Print or return the completion result
    response = completion.choices[0].message
    return response.content


def analyze_activity(activity, src_activities_desc=""):
    chatbot = []
    
    analysis_prompt = f"""
    分析以下活动并提供见解：

    活动名称：{activity['name']}
    布局信息：{activity['layouts']}

    {f'前置活动描述：{src_activities_desc}' if src_activities_desc else ''}

    请提供以下信息：
    1. 这个活动可能的用途
    2. 布局的可能结构
    3. 如何与前置活动（如果有）进行连接和交互
    4. 可能的用户交互流程
    """

    analysis_result = request_gpt_model_in_new_thread(
        analysis_prompt,
        llm_kwargs,
        chatbot,
        history=[],
        sys_prompt="你是一位经验丰富的移动应用程序分析师，专门分析应用程序结构和流程。",
        key_index=0
    )
    analysis_result2 = request_gpt_model_in_new_thread(
        analysis_result,
        llm_kwargs,
        chatbot,
        history=["""{f'前置活动描述：{src_activities_desc}' if src_activities_desc else ''}"""],
        sys_prompt="你是一位经验丰富的阅读理解大师，你阅读上面activity描述，用一句话概括activity功能. 例如：“splashActivity页面功能是启动这个健康APP”",
        key_index=0
    )

    return analysis_result2

def ana_json(json_data):
    activity_insights = {}

    # 创建一个字典来存储每个activity的前置activities
    activity_sources = {activity['name']: [] for activity in json_data['activities']}
    for transition in json_data['transitions']:
        if transition['dest'] not in activity_sources or transition['scr'] not in activity_sources:
            continue
        activity_sources[transition['dest']].append(transition['scr'] for activity in json_data['activities'])
    # activity_contents = {activity['name']: activity["layouts"]}

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
        # find the activity info 
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

    # 开始分析所有活动
    df = 0
    for activity in tqdm(json_data['activities']):
        analyze_activity_recursive(activity['name'])
        # df += 1
        # if df >3:
        #     break

    return activity_insights

def save_to_file(ana_res, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_test', fn='setg_default.csv'):
    sfn = os.path.join(sp, fn)
    hd = ['activity_name', 'semantics']
    df = pd.DataFrame.from_dict(ana_res, orient='index', columns=['semantics'])
    df['activity_name'] =df.index
    df = df[['activity_name', 'semantics']]
    print(f'df: {df}')
    df.to_csv(sfn,encoding='utf-8-sig',header=hd,index=False)
    # with open(sfn, encoding='utf_8_sig',mode='a') as f:
    #     f.write(f'activity_name, semantics\n')
    #     for k, v in ana_res.items():
    #         f.write(f'{k}, {v}')
        


def main():
    d_dir = "/mnt/maldetect_NAS/aibot/oppo/oppo"
    jfp = os.path.join(d_dir, "native_stat_json_res")
    ufp = os.path.join(d_dir, "scamware4SeTG.csv")
    finished_dir = './setg'
    finished_name = 'finish_name.txt'
    fd = os.path.join(finished_dir, finished_name)
    ana_scam_j = load_data(jfp,fd, ufp)
    generated = os.listdir('/home/aibot/workspace/SquiDroidAgent/data/setg_test')
    for k in ana_scam_j:
        if ('setg_'+k+'_4.csv') in generated:
            continue
        logger.debug(f'gen setg_{k} to csv file')
        p = ana_scam_j[k]
        # p = ana_scam_j['1b9edaa86ddf0cfec866463e7201b5f3']
        print(p)
        json_data = {}
        with open(p,mode="r",encoding="utf-8") as f:
            json_data = json.load(f)
        # print(json_data)
        # ana_res = ana_json(json_data)
        for i in range(5):
            if ('setg_'+k+f'_{i}.csv') in generated:
                continue
            ana_res = ana_json(json_data)
            save_to_file(ana_res, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_test', fn=f'setg_{k}_{i}.csv')
            logger.debug(f'save setg_{k}_{i} to setg succ.')

        save_to_file(ana_res, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_test', fn=f'setg_{k}.csv')
        logger.debug(f'save setg_{k} to setg succ.')

if __name__ == "__main__":
    save_to_file({'k':'v','k2':'v2','k3':'v3'}, sp='/home/aibot/workspace/SquiDroidAgent/data/setg_test', fn=f'setg_test.csv')
    main()


# /home/aibot/workspace/SquiDroidAgent/data/setg_test/setg_560fcc6f847830bdacfca48caadf5ac1_1 & 0.csv 
# 被覆盖了 需要重新生成一下