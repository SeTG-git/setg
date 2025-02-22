import time
import os
from llm_connector.toolbox.toolbox import update_ui, get_conf, update_ui_lastest_msg
from llm_connector.toolbox.toolbox import check_packages, report_exception

model_name = 'Qwen'

def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False, key_index=0):
    """
        ⭐多线程方法
        函数的说明请见 request_llms/bridge_all.py
    """
    watch_dog_patience = 5
    response = ""

    from .com_qwenapi import QwenRequestInstance
    sri = QwenRequestInstance(key_index)
    # print(f'inputs的数据格式: {type(inputs)}， inputs的数据样例： {inputs}')
    for response in sri.generate(inputs, llm_kwargs, history, sys_prompt):
        if len(observe_window) >= 1:
            observe_window[0] = response
        # if len(observe_window) >= 2:
            # if (time.time()-observe_window[1]) > watch_dog_patience: raise RuntimeError("程序终止。")
    return response

def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None,key_index=0):
    """
        ⭐单线程方法
        函数的说明请见 request_llms/bridge_all.py
    """
    chatbot.append((inputs, ""))
    yield from update_ui(chatbot=chatbot, history=history)

    # 尝试导入依赖，如果缺少依赖，则给出安装建议
    try:
        check_packages(["dashscope"])
    except:
        yield from update_ui_lastest_msg(f"导入软件依赖失败。使用该模型需要额外依赖，安装方法```pip install --upgrade dashscope```。",
                                         chatbot=chatbot, history=history, delay=0)
        return

    # 检查DASHSCOPE_API_KEY
    if get_conf("DASHSCOPE_API_KEY") == "":
        yield from update_ui_lastest_msg(f"请配置 DASHSCOPE_API_KEY。",
                                         chatbot=chatbot, history=history, delay=0)
        return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    # 开始接收回复
    from .com_qwenapi import QwenRequestInstance
    sri = QwenRequestInstance()
    response = f"[Local Message] 等待{model_name}响应中 ..."
    for response in sri.generate(inputs, llm_kwargs, history, system_prompt):
        chatbot[-1] = (inputs, response)
        yield from update_ui(chatbot=chatbot, history=history)

    # 总结输出
    if response == f"[Local Message] 等待{model_name}响应中 ...":
        response = f"[Local Message] {model_name}响应异常 ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)