import os
import datetime
from concurrent_log_handler import ConcurrentRotatingFileHandler
import logging.config

# 如果本应用包括多个APIs，需要根据不同api区分日志
# api名：function1， function2，上线前改成自己的名字；不够可以加。
def init_logger():
    # 日志相关配置
    LOG_DIR = './log'
    os.makedirs(LOG_DIR, exist_ok=True)
    cur_time = datetime.datetime.now()
    # 日志文件名用物理机端口开头，可以在开多路服务时，区分是哪一路服务的日志
    log_prefix = ('chicken_weight_' + cur_time.strftime('%Y-%m-%d'))
    log_prefix2 = ('self_training_weight' + cur_time.strftime('%Y-%m-%d'))
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,  # 防止在模块层次创建了logger而使自定义的logger失效
        'formatters': {  # 设置日志输出的基本格式
            'standard': {  # 日志格式对象
                # 'format': '%(name)s %(levelname)s %(asctime)s %(filename)s#%(lineno)d: %(message)s',
                'format': ' %(asctime)s %(name)s %(levelname)s %(filename)s#%(lineno)d: %(message)s',
                # 'datefmt': '%Y-%m-%d %A %H:%M:%S', # 去掉星期记录
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {  # 配置日志的处理器对象
            'chicken_weight': {  # 日志处理器对象的名称
                'level': 'INFO',  # 日志处理其对象的等级
                'class': 'concurrent_log_handler.ConcurrentRotatingFileHandler',  # 日志处理器对象所属的类
                'maxBytes': 1024 * 1024 * 512, # 512M 当日志文件超过512MB，自动拆分，放置查日志时文件太大卡死。
                'backupCount': 50, #最多保持50个老文件
                'delay': True,
                'formatter': 'standard',  # 指定日志格式对象
                'filename': os.path.join(LOG_DIR, log_prefix + '.log'),  # 日志处理器对象写入的日志路径
            },
            'self_training_weight': {
                'level': 'INFO',
                'class': 'concurrent_log_handler.ConcurrentRotatingFileHandler',  # 日志处理器对象所属的类
                'maxBytes': 1024 * 1024 * 512,  # 512M
                'backupCount': 50,
                'delay': True,
                'formatter': 'standard',
                'filename': os.path.join(LOG_DIR, log_prefix2 + '.log'),
            },
        },

        'loggers': {
            'chicken_weight': {  # 日志对象
                'handlers': ['chicken_weight'],  # 指定日志处理器对象
                'level': 'INFO',  # 日志等级
                'propagate': True  # 子级logger是否向父级logger传递
            },
            'self_training_weight': {
                'handlers': ['self_training_weight'],
                'level': 'INFO',
                'propagate': True
            },
        }})

