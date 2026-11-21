import logging
import requests
from config.langgraph_config import langgraph_config
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _bocha_search(query: str) -> list[dict]:
    params = langgraph_config.get_search_config()
    headers = langgraph_config.get_search_headers()
    r = requests.post(url="https://api.bochaai.com/v1/web-search", headers=headers,
                      json={"query": query, **params}, timeout=30)
    result = r.json()
    if result.get("code") == 200:
        items = result["data"]["webPages"]["value"]
    else:
        items = []
    return items


@tool
def search_destination_info(query: str) -> str:
    """
    使用bocha搜索目的地信息

    这个工具专门用于搜索旅游目的地的综合信息，
    包括景点、旅游指南、文化背景等。

    参数：
    - query: 搜索查询字符串（目的地名称）

    返回： 格式化的搜索结果字符串

    功能说明：
    1.构建专门的搜索查询
    2.使用_bocha_search()进行搜索
    3.格式化结果供智能体理解
    4.处理搜索错误和异常情况
    """
    logger.info(f"调用目的地信息搜索工具 - 查询: {query}")

    try:
        search_query = query + "旅游目的地指南景点"
        results = _bocha_search(search_query)

        logger.info(f"搜索返回 {len(results)} 个结果")

        if not results:
            error_msg = f"未找到目的地搜索结果: {query}"
            logger.warning(error_msg)
            return error_msg

        formatted_results = []
        for i, result in enumerate(results[:5], 1):
            title = result.get('name', '无标题')
            body = result.get('summary', '无描述')
            href = result.get('url', '无URL')

            formatted_results.append(
                f"{i}. {title}\n"
                f"   {body}\n"
                f"   来源: {href}\n"
            )

            logger.debug(f"结果 {i}: {title[:50]}...")

        result_text = "\n".join(formatted_results)
        logger.info(f"目的地信息搜索成功，返回 {len(formatted_results)} 个格式化结果")
        return result_text

    except Exception as e:
        error_msg = f"搜索目的地信息时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def search_attractions(destination: str, interests: str = "") -> str:
    """
    搜索目的地景点和活动

    这个工具专门用于搜索特定目的地的热门景点、
    活动和必游之地，可以根据兴趣进行筛选。

    参数：
    - destination: 目的地名称
    - interests: 兴趣关键词（可选）

    返回：格式化的景点信息字符串
    """

    logger.info(f"调用景点搜索工具 - 目的地: {destination}, 兴趣: {interests}")

    try:
        attraction_query = f"{destination} 热门景点 活动 {interests} 必游之地"
        results = _bocha_search(attraction_query)

        logger.info(f"搜索返回 {len(results)} 个结果")

        if not results:
            error_msg = f"未找到{destination}的景点信息"
            logger.warning(error_msg)
            return error_msg

        attractions = []
        for i, result in enumerate(results[:6], 1):
            title = result.get('name', '景点')
            body = result.get('summary', '无描述')[:200]
            href = result.get('url', '无URL')

            attractions.append(
                f"{i}. {title}\n"
                f"   {body}\n"
                f"   来源: {href}\n"
            )

            logger.debug(f"结果 {i}: {title[:50]}...")

        result_text = f"{destination}的热门景点: \n" + "\n".join(attractions)
        logger.info(f"景点搜索成功，返回 {len(attractions)} 个格式化结果")
        return result_text

    except Exception as e:
        error_msg = f"搜索景点信息时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def search_budget_info(destination: str, duration: str = "") -> str:
    """
    搜索预算和费用信息

    这个工具专门用于搜索特定目的地的旅行预算、
    日常开销和费用估算信息。

    参数：
    - destination: 目的地名称
    - duration: 旅行时长（可选）

    返回：格式化的预算信息字符串
    """
    logger.info(f"调用预算信息搜索工具 - 目的地: {destination}, 时长: {duration}")

    try:
        budget_query = f"{destination} 旅行预算 日常开销 费用 {duration} 花费"
        results = _bocha_search(budget_query)

        logger.info(f"搜索返回 {len(results)} 个结果")

        if not results:
            error_msg = f"未找到{destination}的旅行预算信息"
            logger.warning(error_msg)
            return error_msg

        budget_info = []
        for i, result in enumerate(results[:3], 1):
            title = result.get('name', '预算信息')
            body = result.get('summary', '无详细信息')[:200]
            href = result.get('url', '无URL')

            budget_info.append(
                f"{i}. {title}\n"
                f"   {body}\n"
                f"   来源: {href}\n"
            )

            logger.debug(f"结果 {i}: {title[:50]}...")

        result_text = f"{destination}的预算信息: \n" + "\n".join(budget_info)
        logger.info(f"预算信息搜索成功，返回 {len(budget_info)} 个格式化结果")
        return result_text

    except Exception as e:
        error_msg = f"搜索预算信息时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def search_hotels(destination: str, budget: str = '中等预算') -> str:
    """
    搜索酒店信息和价格

    这个工具专门用于搜索特定目的地的酒店选择、
    包括住宿选项，价格信息和最佳住宿地点。

    参数：
    - destination: 目的地名称
    - budget: 住宿预算（默认'中等预算'）

    返回：格式化的预算信息字符串
    """
    logger.info(f"调用酒店信息搜索工具 - 目的地: {destination}, 住宿预算: {budget}")

    try:
        hotel_query = f"{destination} 酒店推荐 价格信息 {budget} 最佳住宿"
        results = _bocha_search(hotel_query)

        logger.info(f"搜索返回 {len(results)} 个结果")

        if not results:
            error_msg = f"未找到{destination}的酒店信息"
            logger.warning(error_msg)
            return error_msg

        budget_info = []
        for i, result in enumerate(results[:3], 1):
            title = result.get('name', '酒店信息')
            body = result.get('summary', '无详细信息')[:200]
            href = result.get('url', '无URL')

            budget_info.append(
                f"{i}. {title}\n"
                f"   {body}\n"
                f"   来源: {href}\n"
            )

            logger.debug(f"结果 {i}: {title[:50]}...")

        result_text = f"{destination}的酒店信息 ({budget}): \n" + "\n".join(budget_info)
        logger.info(f"酒店信息搜索成功，返回 {len(budget_info)} 个格式化结果")
        return result_text

    except Exception as e:
        error_msg = f"搜索酒店信息时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def search_restaurants(destination: str, cuisine: str = "") -> str:
    """
    搜索餐厅和用餐选择

    这个工具专门用于搜索特定目的地的餐厅推荐，
    包括当地美食、特色菜系和用餐推荐。

    参数：
    - destination: 目的地
    - cuisine: 菜系类型（可选）

    返回：格式化的餐厅推荐字符串
    """
    logger.info(f"调用餐厅搜索工具 - 目的地: {destination}, 菜系: {cuisine}")

    try:
        restaurants_query = f"{destination} 最佳餐厅 美食推荐 {cuisine} 当地美食"
        results = _bocha_search(restaurants_query)

        logger.info(f"搜索返回 {len(results)} 个结果")

        if not results:
            error_msg = f"未找到{destination}的餐厅推荐信息"
            logger.warning(error_msg)
            return error_msg

        restaurants = []
        for i, result in enumerate(results[:5], 1):
            title = result.get('name', '餐厅信息')
            body = result.get('summary', '无详细信息')[:200]
            href = result.get('url', '无URL')

            restaurants.append(
                f"{i}. {title}\n"
                f"   {body}\n"
                f"   来源: {href}\n"
            )

            logger.debug(f"餐厅结果 {i}: {title[:50]}...")

        result_text = f"{destination}的餐厅推荐信息：\n" + "\n".join(restaurants)
        logger.info(f"餐厅搜索成功，返回 {len(restaurants)} 个格式化结果")
        return result_text
    except Exception as e:
        error_msg = f"搜索餐厅信息时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def search_local_tips(destination: str) ->str:
    """
    搜索当地贴士、文化和内部信息

    这个工具专门用于搜索目的地的当地文化、
    礼仪习俗和内部旅行贴士。

    参数：
    - destination: 目的地名称

    返回：格式化的当地贴士字符串
    """
    logger.info(f"调用当地贴士搜索工具 - 目的地: {destination}")

    try:
        tips_query = f"{destination} 当地贴士 旅行指南 文化礼仪 注意事项"
        results = _bocha_search(tips_query)

        logger.info(f"当地贴士搜索返回 {len(results)} 个结果")

        if not results:
            error_msg = f"未找到{destination}的当地贴士"
            logger.warning(error_msg)
            return error_msg

        tips = []
        for i, result in enumerate(results[:3], 1):
            title = result.get('name', '当地贴士')
            body = result.get('summary', '无详细信息')[:200]

            tips.append(
                f"• {title}\n"
                f"  {body}...\n"
            )

            logger.debug(f"当地贴士结果 {i}: {title[:50]}...")

        result_text = f"{destination}的当地贴士:\n" + "\n".join(tips)
        logger.info(f"当地贴士搜索成功，返回 {len(tips)} 个格式化结果")
        return result_text

    except Exception as e:
        error_msg = f"搜索当地贴士时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
async def get_daily_forecast(location: str, days: int = 3) -> str:
    """获取指定城市未来几天天气预报，返回逐日天气文本。"""
    from tools.weather_mcp_server import get_daily_forecast as _impl
    try:
        return await _impl(location, max(1, min(int(days), 10)))
    except Exception as exc:
        return "天气查询失败：" + str(exc)
