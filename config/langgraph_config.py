import os
from dotenv import load_dotenv
from typing import Dict, Any
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / '.env')


class LangGraphConfig:
    """
    OpenAI兼容接口配置
    搜索引擎配置
    智能体协作配置
    模型参数配置
    """

    # OpenAI兼容接口配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', "https://api.deepseek.com")
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'deepseek-v4-flash')

    # 搜索引擎配置
    BOCHA_API_KEY = os.getenv('BOCHA_API_KEY')
    BOCHA_COUNT = 10
    BOCHA_FRESHNESS = 'noLimit'
    BOCHA_SUMMARY = True

    # 智能体协作配置
    MAX_ITERATIONS = 50     # 最大迭代次数
    RECURSION_LIMIT = 100   # 最大递归次数

    # 模型参数配置
    TEMPERATURE = 0.7
    MAX_TOKENS = 4000
    TOP_P = 0.9

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """
        获取OpenAI兼容接口配置
        :return:
        """
        config = {
            'model': cls.OPENAI_MODEL,
            'temperature': cls.TEMPERATURE,
            'max_tokens': cls.MAX_TOKENS,
            'top_p': cls.TOP_P,
        }
        if cls.OPENAI_API_KEY:
            config['api_key'] = cls.OPENAI_API_KEY
        if cls.OPENAI_BASE_URL:
            config['base_url'] = cls.OPENAI_BASE_URL
        return config

    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """
        获取搜索引擎参数
        :return:
        """
        return {
            'count': cls.BOCHA_COUNT,
            'freshness': cls.BOCHA_FRESHNESS,
            'summary': cls.BOCHA_SUMMARY,
        }

    @classmethod
    def get_search_headers(cls) -> Dict[str, Any]:
        """
        搜索引擎鉴权
        :return:
        """
        return {'Authorization': 'Bearer' + ' ' + cls.BOCHA_API_KEY}

    @classmethod
    def validate_config(cls) -> bool:
        """
        验证配置完整性
        :return:
        """
        if not cls.OPENAI_API_KEY:
            print("警告: 环境变量中未找到 OPENAI_API_KEY")
            print("请在.env文件中设置 OPENAI_API_KEY 或配置兼容网关的密钥")
            return False
        if not cls.BOCHA_API_KEY:
            print("警告: 环境变量中未找到 BOCHA_API_KEY")
            print("请在.env文件中设置 BOCHA_API_KEY 或配置兼容网关的密钥")
            return False
        return True


# 初始化配置实例
langgraph_config = LangGraphConfig()

# 在导入时验证配置
if not langgraph_config.validate_config():
    print("配置验证失败")
    print("请检查您的.env文件并确保设置了 OPENAI_API_KEY或BOCHA_API_KEY")
else:
    print("LangGraph配置加载成功")
