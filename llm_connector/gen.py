from .crazy_utils import read_and_clean_pdf_text,top1
from .crazy_utils import request_gpt_model_in_new_thread
from bs4 import BeautifulSoup
import hashlib

fast_debug = False


def remove_element_by_id(html_content, element_id):
    soup = BeautifulSoup(html_content, 'html.parser')
    element = soup.find(id=element_id)
    if element:
        element.extract()
    return str(soup)

def extract_text_by_id(html_content, element_id):
    soup = BeautifulSoup(html_content, 'html.parser')
    element = soup.find(id=element_id)
    if element:
        return element.get_text().strip()
    else:
        element = soup.find('p')
    if element:
        return element.get_text().strip()
    else:
        return ''

def read_and_clean_html_text(html_file):
    return extract_text_by_id(html_file, "com.tencent.mm:id/bkl")

def clean_html(html_file):
    return remove_element_by_id(html_file, "com.tencent.mm:id/bkl")



def com_html(html_file,fraud_list, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, case="聊天诈骗"):
    prev_md5 = None
    view_description = None
    history_con = None
    def _con_html(html_file,fraud_list, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
        print('begin analysis on:', file_content[:100])
        file_content = read_and_clean_html_text(html_file)
        clean_html_str = clean_html(html_file)
        file_content = file_content.encode('utf-8', 'ignore').decode()   # avoid reading non-utf8 chars
        if prev_md5 == hashlib.md5(clean_html_str.encode('utf-8', 'ignore')).hexdigest():
            pass
        ############################## <第 1 步，从摘要中提取高价值信息，放到history中> ##################################
        ############################## <第 2 步，迭代地历遍整个文章，提取精炼信息> ##################################
        else:
            i_say = f'首先你在安卓布局语法的基础上，根据给定的html信息，理解这个页面的功能。html内容如下：{html_file}';
            gpt_say = yield from request_gpt_model_in_new_thread(i_say,   # i_say=真正给chatgpt的提问
                                                                            llm_kwargs, chatbot, 
                                                                            history=[""], # 迭代上一次的结果
                                                                            sys_prompt="你是一名{case}专家, 分析手机软件界面和界面上的对话内容，使用中文回答问题."  # 提示
                                                                            )           # 用户提示
            view_description = gpt_say
            history_con = "无"
        conversation = file_content[1]  # 对方：。。。，我：。。。
        t_res = []
        for fraud_con in fraud_list:
            i_say = f'请你阅读这段对话内容，并根据对话内容和历史对话信息，进行后续对话的生成。\\n
                        你续写的对话应该与下述的对话风格一致：{fraud_con}。\\n
                        当前对话信息{conversation}。';
            gpt_say = yield from request_gpt_model_in_new_thread(i_say,   # i_say=真正给chatgpt的提问
                                                                            llm_kwargs, chatbot, 
                                                                            history=["上一个页面的对话信息概括如下：", history_con], # 迭代上一次的结果
                                                                            sys_prompt=f"你是一名{case}专家, 分析手机软件界面和界面上的对话内容，使用中文回答问题."  # 提示
                                                                            )           # 用户提示
            t_res.append(gpt_say)

        pre_res, top1_score = top1(t_res, fraud_list)
        if top1_score < 0.5:
            pre_res = ""
        else:
            pre_res = t_res[pre_res]

        i_say = f'在综合页面描述、对话信息、可能走势；三方面的内容进行总结。该页面描述如下：{view_description}，对话信息如下{conversation}，可能的走势如下{pre_res}'; 
        cur_history = gpt_say = yield from request_gpt_model_in_new_thread(i_say,   # i_say=真正给chatgpt的提问
                                                                            llm_kwargs, chatbot, 
                                                                            history=["上一个页面的对话信息概括如下：", history_con], # 迭代上一次的结果
                                                                            sys_prompt=f"你是一名{case}专家, 分析手机软件界面和界面上的对话内容，使用中文回答问题."  # 提示
                                                                            )           # 用户提示
        history_con = cur_history
        return history_con
    return _con_html(html_file,fraud_list, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)

    iteration_results = []
    MAX_WORD_TOTAL = 4096
    n_fragment = len(paper_fragments)
    if n_fragment >= 20: print('文章极长，不能达到预期效果')
    for i in range(n_fragment):
        NUM_OF_WORD = MAX_WORD_TOTAL // n_fragment
        i_say = f"Read this section, recapitulate the content of this section with less than {NUM_OF_WORD} words: {paper_fragments[i]}"
        i_say_show_user = f"[{i+1}/{n_fragment}] Read this section, recapitulate the content of this section with less than {NUM_OF_WORD} words: {paper_fragments[i][:200]} ...."
        gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(i_say, i_say_show_user,  # i_say=真正给chatgpt的提问， i_say_show_user=给用户看的提问
                                                                           llm_kwargs, chatbot, 
                                                                           history=["The main idea of the previous section is?", last_iteration_result], # 迭代上一次的结果
                                                                           sys_prompt="Extract the main idea of this section, answer me with Chinese."  # 提示
                                                                        ) 
        iteration_results.append(gpt_say)
        last_iteration_result = gpt_say

    ############################## <第 3 步，整理history> ##################################
    final_results.extend(iteration_results)
    final_results.append(f'接下来，你是一名专业的学术教授，利用以上信息，使用中文回答我的问题。')
    # 接下来两句话只显示在界面上，不起实际作用
    i_say_show_user = f'接下来，你是一名专业的学术教授，利用以上信息，使用中文回答我的问题。'; gpt_say = "[Local Message] 收到。"
    chatbot.append([i_say_show_user, gpt_say])

    ############################## <第 4 步，设置一个token上限，防止回答时Token溢出> ##################################
    from .crazy_utils import input_clipping
    _, final_results = input_clipping("", final_results, max_token_limit=3200)
