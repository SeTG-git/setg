
"""
    该文件中主要包含2个函数，是所有LLM的通用接口，它们会继续向下调用更底层的LLM模型，处理多模型并行等细节

    不具备多线程能力的函数：正常对话时使用，具备完备的交互功能，不可多线程
    1. predict(...)

    具备多线程调用能力的函数：在函数插件中被调用，灵活而简洁
    2. predict_no_ui_long_connection(...)
"""
import tiktoken, copy
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from origin_academic.toolbox.toolbox import get_conf, trimmed_format_exc, apply_gpt_academic_string_mask


from .bridge_qwen import predict_no_ui_long_connection as qwen_noui
from .bridge_qwen import predict as qwen_ui
from .bridge_qwen_local import predict as qwen_local_ui
from .bridge_qwen_local import predict_no_ui_long_connection as  qwen_local_noui



class LazyloadTiktoken(object):
    def __init__(self, model):
        self.model = model

    @staticmethod
    @lru_cache(maxsize=128)
    def get_encoder(model):
        print('正在加载tokenizer，如果是第一次运行，可能需要一点时间下载参数')
        tmp = tiktoken.encoding_for_model(model)
        print('加载tokenizer完毕')
        return tmp

    def encode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model)
        return encoder.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model)
        return encoder.decode(*args, **kwargs)
# 获取tokenizer
tokenizer_gpt35 = LazyloadTiktoken("gpt-3.5-turbo")
tokenizer_gpt4 = LazyloadTiktoken("gpt-4")
get_token_num_gpt35 = lambda txt: len(tokenizer_gpt35.encode(txt, disallowed_special=()))
get_token_num_gpt4 = lambda txt: len(tokenizer_gpt4.encode(txt, disallowed_special=()))
model_info={
    "qwen-turbo": {
        "fn_with_ui": qwen_ui,
        "fn_without_ui": qwen_noui,
        "endpoint": None,
        "max_token": 6144,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "qwen-plus": {
        "fn_with_ui": qwen_ui,
        "fn_without_ui": qwen_noui,
        "endpoint": None,
        "max_token": 30720,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },
    "qwen-max": {
        "fn_with_ui": qwen_ui,
        "fn_without_ui": qwen_noui,
        "endpoint": None,
        "max_token": 28672,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    }
}
model_info.update({
        "qwen-local": {
            "fn_with_ui": qwen_local_ui,
            "fn_without_ui": qwen_local_noui,
            "can_multi_thread": False,
            "endpoint": None,
            "max_token": 4096,
            "tokenizer": tokenizer_gpt35,
            "token_cnt": get_token_num_gpt35,
        }
    })



def LLM_CATCH_EXCEPTION(f):
    """
    装饰器函数，将错误显示出来
    """
    def decorated(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience):
        try:
            return f(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
        except Exception as e:
            tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
            observe_window[0] = tb_str
            return tb_str
    return decorated


def predict_no_ui_long_connection(inputs, llm_kwargs, history, sys_prompt, observe_window=[], console_slience=False,key_index=0):
    """
    发送至LLM，等待回复，一次性完成，不显示中间过程。但内部用stream的方法避免中途网线被掐。
    inputs：
        是本次问询的输入
    sys_prompt:
        系统静默prompt
    llm_kwargs：
        LLM的内部调优参数
    history：
        是之前的对话列表
    observe_window = None：
        用于负责跨越线程传递已经输出的部分，大部分时候仅仅为了fancy的视觉效果，留空即可。observe_window[0]：观测窗。observe_window[1]：看门狗
    """
    import threading, time, copy

    inputs = apply_gpt_academic_string_mask(inputs, mode="show_llm")
    model = llm_kwargs['llm_model']
    n_model = 1
    if '&' not in model:
        assert not model.startswith("tgui"), "TGUI不支持函数插件的实现"

        # 如果只询问1个大语言模型：
        method = model_info[model]["fn_without_ui"]
        return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience,key_index)


def predict(inputs, llm_kwargs, *args, **kwargs):
    """
    发送至LLM，流式获取输出。
    用于基础的对话功能。
    inputs 是本次问询的输入
    top_p, temperature是LLM的内部调优参数
    history 是之前的对话列表（注意无论是inputs还是history，内容太长了都会触发token数量溢出的错误）
    chatbot 为WebUI中显示的对话列表，修改它，然后yeild出去，可以直接修改对话界面内容
    additional_fn代表点击的哪个按钮，按钮见functional.py
    """

    inputs = apply_gpt_academic_string_mask(inputs, mode="show_llm")
    method = model_info[llm_kwargs['llm_model']]["fn_with_ui"]  # 如果这里报错，检查config中的AVAIL_LLM_MODELS选项
    yield from method(inputs, llm_kwargs, *args, **kwargs)

