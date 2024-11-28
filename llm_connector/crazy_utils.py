from llm_connector.toolbox.toolbox import update_ui, get_conf, trimmed_format_exc, get_max_token, Singleton
import threading
import os
import logging

def input_clipping(inputs, history, max_token_limit):
    import numpy as np
    from request_llms.bridge_all import model_info
    enc = model_info["gpt-3.5-turbo"]['tokenizer']
    def get_token_num(txt): return len(enc.encode(txt, disallowed_special=()))

    mode = 'input-and-history'
    # 当 输入部分的token占比 小于 全文的一半时，只裁剪历史
    input_token_num = get_token_num(inputs)
    if input_token_num < max_token_limit//2:
        mode = 'only-history'
        max_token_limit = max_token_limit - input_token_num

    everything = [inputs] if mode == 'input-and-history' else ['']
    everything.extend(history)
    n_token = get_token_num('\n'.join(everything))
    everything_token = [get_token_num(e) for e in everything]
    delta = max(everything_token) // 16 # 截断时的颗粒度

    while n_token > max_token_limit:
        where = np.argmax(everything_token)
        encoded = enc.encode(everything[where], disallowed_special=())
        clipped_encoded = encoded[:len(encoded)-delta]
        everything[where] = enc.decode(clipped_encoded)[:-1]    # -1 to remove the may-be illegal char
        everything_token[where] = get_token_num(everything[where])
        n_token = get_token_num('\n'.join(everything))

    if mode == 'input-and-history':
        inputs = everything[0]
    else:
        pass
    history = everything[1:]
    return inputs, history


def request_gpt_model_in_new_thread(
        inputs, llm_kwargs,
        chatbot, history, sys_prompt, refresh_interval=0.2,
        handle_token_exceed=True,
        retry_times_at_unknown_error=2,
        key_index=0
        ):
    """
    Request GPT model，请求GPT模型同时维持用户界面活跃。

    输入参数 Args （以_array结尾的输入变量都是列表，列表长度为子任务的数量，执行时，会把列表拆解，放到每个子线程中分别执行）:
        inputs (string): List of inputs （输入）
        top_p (float): Top p value for sampling from model distribution （GPT参数，浮点数）
        temperature (float): Temperature value for sampling from model distribution（GPT参数，浮点数）
        chatbot: chatbot inputs and outputs （用户界面对话窗口句柄，用于数据流可视化）
        history (list): List of chat history （历史，对话历史列表）
        sys_prompt (string): List of system prompts （系统输入，列表，用于输入给GPT的前提提示，比如你是翻译官怎样怎样）
        refresh_interval (float, optional): Refresh interval for UI (default: 0.2) （刷新时间间隔频率，建议低于1，不可高于3，仅仅服务于视觉效果）
        handle_token_exceed：是否自动处理token溢出的情况，如果选择自动处理，则会在溢出时暴力截断，默认开启
        retry_times_at_unknown_error：失败时的重试次数

    输出 Returns:
        future: 输出，GPT返回的结果
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    from llm_connector.request_llms.bridge_all import predict_no_ui_long_connection
    # 用户反馈
    # yield from update_ui(chatbot=chatbot, history=[]) # 刷新界面
    executor = ThreadPoolExecutor(max_workers=16)
    mutable = ["", time.time(), ""]
    # 看门狗耐心
    watch_dog_patience = 5
    # 请求任务
    def _req_gpt(inputs, history, sys_prompt):
        retry_op = retry_times_at_unknown_error
        exceeded_cnt = 0
        while True:
            # watchdog error
            # if len(mutable) >= 2 and (time.time()-mutable[1]) > watch_dog_patience:
            #     raise RuntimeError("检测到程序终止。")
            try:
                # 【第一种情况】：顺利完成
                result = predict_no_ui_long_connection(
                    inputs=inputs, llm_kwargs=llm_kwargs,
                    history=history, sys_prompt=sys_prompt, observe_window=mutable,key_index=key_index)
                return result
            except ConnectionAbortedError as token_exceeded_error:
                # 【第二种情况】：Token溢出
                if handle_token_exceed:
                    exceeded_cnt += 1
                    # 【选择处理】 尝试计算比例，尽可能多地保留文本
                    from llm_connector.toolbox.toolbox import get_reduce_token_percent
                    p_ratio, n_exceed = get_reduce_token_percent(str(token_exceeded_error))
                    MAX_TOKEN = get_max_token(llm_kwargs)
                    EXCEED_ALLO = 512 + 512 * exceeded_cnt
                    inputs, history = input_clipping(inputs, history, max_token_limit=MAX_TOKEN-EXCEED_ALLO)
                    mutable[0] += f'[Local Message] 警告，文本过长将进行截断，Token溢出数：{n_exceed}。\n\n'
                    continue # 返回重试
                else:
                    # 【选择放弃】
                    tb_str = '```\n' + trimmed_format_exc() + '```'
                    mutable[0] += f"[Local Message] 警告，在执行过程中遭遇问题, Traceback：\n\n{tb_str}\n\n"
                    return mutable[0] # 放弃
            except:
                # 【第三种情况】：其他错误：重试几次
                tb_str = '```\n' + trimmed_format_exc() + '```'
                print(tb_str)
                mutable[0] += f"[Local Message] 警告，在执行过程中遭遇问题, Traceback：\n\n{tb_str}\n\n"
                if retry_op > 0:
                    retry_op -= 1
                    mutable[0] += f"[Local Message] 重试中，请稍等 {retry_times_at_unknown_error-retry_op}/{retry_times_at_unknown_error}：\n\n"
                    if ("Rate limit reached" in tb_str) or ("Too Many Requests" in tb_str):
                        time.sleep(30)
                    time.sleep(5)
                    continue # 返回重试
                else:
                    time.sleep(5)
                    return mutable[0] # 放弃

    # 提交任务
    future = executor.submit(_req_gpt, inputs, history, sys_prompt)
    
    final_result = future.result()
    # chatbot[-1] = [chatbot[-1][0], final_result]
    return final_result


def can_multi_process(llm):
    if llm.startswith('gpt-'): return True
    if llm.startswith('api2d-'): return True
    if llm.startswith('azure-'): return True
    if llm.startswith('spark'): return True
    if llm.startswith('zhipuai') or llm.startswith('glm-'): return True
    return False

def read_and_clean_pdf_text(fp):
    import fitz, copy
    import re
    import numpy as np
    from colorful import print亮黄, print亮绿
    fc = 0  # Index 0 文本
    fs = 1  # Index 1 字体
    fb = 2  # Index 2 框框
    REMOVE_FOOT_NOTE = True # 是否丢弃掉 不是正文的内容 （比正文字体小，如参考文献、脚注、图注等）
    REMOVE_FOOT_FFSIZE_PERCENT = 0.95 # 小于正文的？时，判定为不是正文（有些文章的正文部分字体大小不是100%统一的，有肉眼不可见的小变化）
    def primary_ffsize(l):
        """
        提取文本块主字体
        """
        fsize_statiscs = {}
        for wtf in l['spans']:
            if wtf['size'] not in fsize_statiscs: fsize_statiscs[wtf['size']] = 0
            fsize_statiscs[wtf['size']] += len(wtf['text'])
        return max(fsize_statiscs, key=fsize_statiscs.get)

    def ffsize_same(a,b):
        """
        提取字体大小是否近似相等
        """
        return abs((a-b)/max(a,b)) < 0.02

    with fitz.open(fp) as doc:
        meta_txt = []
        meta_font = []

        meta_line = []
        meta_span = []
        for index, page in enumerate(doc):
            text_areas = page.get_text("dict")  # 获取页面上的文本信息
            for t in text_areas['blocks']:
                if 'lines' in t:
                    pf = 998
                    for l in t['lines']:
                        txt_line = "".join([wtf['text'] for wtf in l['spans']])
                        if len(txt_line) == 0: continue
                        pf = primary_ffsize(l)
                        meta_line.append([txt_line, pf, l['bbox'], l])
                        for wtf in l['spans']: # for l in t['lines']:
                            meta_span.append([wtf['text'], wtf['size'], len(wtf['text'])])
            meta_txt.extend([" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                '- ', '') for t in text_areas['blocks'] if 'lines' in t])
            meta_font.extend([np.mean([np.mean([wtf['size'] for wtf in l['spans']])
                             for l in t['lines']]) for t in text_areas['blocks'] if 'lines' in t])
            if index == 0:
                page_one_meta = [" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                    '- ', '') for t in text_areas['blocks'] if 'lines' in t]

        ############################## <第 2 步，获取正文主字体> ##################################
        try:
            fsize_statiscs = {}
            for span in meta_span:
                if span[1] not in fsize_statiscs: fsize_statiscs[span[1]] = 0
                fsize_statiscs[span[1]] += span[2]
            main_fsize = max(fsize_statiscs, key=fsize_statiscs.get)
            if REMOVE_FOOT_NOTE:
                give_up_fize_threshold = main_fsize * REMOVE_FOOT_FFSIZE_PERCENT
        except:
            raise RuntimeError(f'抱歉, 我们暂时无法解析此PDF文档: {fp}。')
        ############################## <第 3 步，切分和重新整合> ##################################
        mega_sec = []
        sec = []
        for index, line in enumerate(meta_line):
            if index == 0:
                sec.append(line[fc])
                continue
            if REMOVE_FOOT_NOTE:
                if meta_line[index][fs] <= give_up_fize_threshold:
                    continue
            if ffsize_same(meta_line[index][fs], meta_line[index-1][fs]):
                # 尝试识别段落
                if meta_line[index][fc].endswith('.') and\
                    (meta_line[index-1][fc] != 'NEW_BLOCK') and \
                    (meta_line[index][fb][2] - meta_line[index][fb][0]) < (meta_line[index-1][fb][2] - meta_line[index-1][fb][0]) * 0.7:
                    sec[-1] += line[fc]
                    sec[-1] += "\n\n"
                else:
                    sec[-1] += " "
                    sec[-1] += line[fc]
            else:
                if (index+1 < len(meta_line)) and \
                    meta_line[index][fs] > main_fsize:
                    # 单行 + 字体大
                    mega_sec.append(copy.deepcopy(sec))
                    sec = []
                    sec.append("# " + line[fc])
                else:
                    # 尝试识别section
                    if meta_line[index-1][fs] > meta_line[index][fs]:
                        sec.append("\n" + line[fc])
                    else:
                        sec.append(line[fc])
        mega_sec.append(copy.deepcopy(sec))

        finals = []
        for ms in mega_sec:
            final = " ".join(ms)
            final = final.replace('- ', ' ')
            finals.append(final)
        meta_txt = finals

        ############################## <第 4 步，乱七八糟的后处理> ##################################
        def 把字符太少的块清除为回车(meta_txt):
            for index, block_txt in enumerate(meta_txt):
                if len(block_txt) < 100:
                    meta_txt[index] = '\n'
            return meta_txt
        meta_txt = 把字符太少的块清除为回车(meta_txt)

        def 清理多余的空行(meta_txt):
            for index in reversed(range(1, len(meta_txt))):
                if meta_txt[index] == '\n' and meta_txt[index-1] == '\n':
                    meta_txt.pop(index)
            return meta_txt
        meta_txt = 清理多余的空行(meta_txt)

        def 合并小写开头的段落块(meta_txt):
            def starts_with_lowercase_word(s):
                pattern = r"^[a-z]+"
                match = re.match(pattern, s)
                if match:
                    return True
                else:
                    return False
            # 对于某些PDF会有第一个段落就以小写字母开头,为了避免索引错误将其更改为大写
            if starts_with_lowercase_word(meta_txt[0]):
                meta_txt[0] = meta_txt[0].capitalize()
            for _ in range(100):
                for index, block_txt in enumerate(meta_txt):
                    if starts_with_lowercase_word(block_txt):
                        if meta_txt[index-1] != '\n':
                            meta_txt[index-1] += ' '
                        else:
                            meta_txt[index-1] = ''
                        meta_txt[index-1] += meta_txt[index]
                        meta_txt[index] = '\n'
            return meta_txt
        meta_txt = 合并小写开头的段落块(meta_txt)
        meta_txt = 清理多余的空行(meta_txt)

        meta_txt = '\n'.join(meta_txt)
        # 清除重复的换行
        for _ in range(5):
            meta_txt = meta_txt.replace('\n\n', '\n')

        # 换行 -> 双换行
        meta_txt = meta_txt.replace('\n', '\n\n')

        ############################## <第 5 步，展示分割效果> ##################################
        # for f in finals:
        #    print亮黄(f)
        #    print亮绿('***************************')

    return meta_txt, page_one_meta


def get_files_from_everything(txt, type): # type='.md'
    import glob, os

    success = True
    if txt.startswith('http'):
        # 网络的远程文件
        import requests
        from data.llm_connector.toolbox.toolbox import get_conf
        from data.llm_connector.toolbox.toolbox import get_log_folder, gen_time_str
        proxies = get_conf('proxies')
        try:
            r = requests.get(txt, proxies=proxies)
        except:
            raise ConnectionRefusedError(f"无法下载资源{txt}，请检查。")
        path = os.path.join(get_log_folder(plugin_name='web_download'), gen_time_str()+type)
        with open(path, 'wb+') as f: f.write(r.content)
        project_folder = get_log_folder(plugin_name='web_download')
        file_manifest = [path]
    elif txt.endswith(type):
        # 直接给定文件
        file_manifest = [txt]
        project_folder = os.path.dirname(txt)
    elif os.path.exists(txt):
        # 本地路径，递归搜索
        project_folder = txt
        file_manifest = [f for f in glob.glob(f'{project_folder}/**/*'+type, recursive=True)]
        if len(file_manifest) == 0:
            success = False
    else:
        project_folder = None
        file_manifest = []
        success = False

    return success, file_manifest, project_folder



@Singleton
class nougat_interface():
    def __init__(self):
        self.threadLock = threading.Lock()

    def nougat_with_timeout(self, command, cwd, timeout=3600):
        import subprocess
        from llm_connector.toolbox.toolbox import ProxyNetworkActivate
        logging.info(f'正在执行命令 {command}')
        with ProxyNetworkActivate("Nougat_Download"):
            process = subprocess.Popen(command, shell=True, cwd=cwd, env=os.environ)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            print("Process timed out!")
            return False
        return True


    def NOUGAT_parse_pdf(self, fp, chatbot, history):
        from llm_connector.toolbox.toolbox import update_ui_lastest_msg

        yield from update_ui_lastest_msg("正在解析论文, 请稍候。进度：正在排队, 等待线程锁...",
                                         chatbot=chatbot, history=history, delay=0)
        self.threadLock.acquire()
        import glob, threading, os
        from data.llm_connector.toolbox.toolbox import get_log_folder, gen_time_str
        dst = os.path.join(get_log_folder(plugin_name='nougat'), gen_time_str())
        os.makedirs(dst)

        yield from update_ui_lastest_msg("正在解析论文, 请稍候。进度：正在加载NOUGAT... （提示：首次运行需要花费较长时间下载NOUGAT参数）",
                                         chatbot=chatbot, history=history, delay=0)
        self.nougat_with_timeout(f'nougat --out "{os.path.abspath(dst)}" "{os.path.abspath(fp)}"', os.getcwd(), timeout=3600)
        res = glob.glob(os.path.join(dst,'*.mmd'))
        if len(res) == 0:
            self.threadLock.release()
            raise RuntimeError("Nougat解析论文失败。")
        self.threadLock.release()
        return res[0]




def try_install_deps(deps, reload_m=[]):
    import subprocess, sys, importlib
    for dep in deps:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', dep])
    import site
    importlib.reload(site)
    for m in reload_m:
        importlib.reload(__import__(m))


def get_plugin_arg(plugin_kwargs, key, default):
    # 如果参数是空的
    if (key in plugin_kwargs) and (plugin_kwargs[key] == ""): plugin_kwargs.pop(key)
    # 正常情况
    return plugin_kwargs.get(key, default)


def top1(res,tempalte):
    sim_score = [similarity(res[i],t) for i, t in enumerate(tempalte)]
    return sim_score.index(max(sim_score)), max(sim_score)


def similarity(a, b):
    """
    计算两个字符串的相似度
    """
    import difflib
    return difflib.SequenceMatcher(None, a, b).quick_ratio()