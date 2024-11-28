import logging



LOG_LEVEL=logging.DEBUG
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',    filename='gen_setg.log',    filemode='a')  # 这个参数可选，'w'表示覆盖写入，'a'表示追加写入)
logger = logging.getLogger(__name__)