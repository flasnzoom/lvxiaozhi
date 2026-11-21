import os
from datetime import datetime, timedelta
import httpx
from mcp.server.fastmcp import FastMCP
import logging
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urljoin
from typing import Optional, Dict, Any, Union
from pypinyin import lazy_pinyin, Style

logger = logging.getLogger(__name__)
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path)

mcp = FastMCP("weather")

QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
QWEATHER_API_BASE = os.getenv("QWEATHER_API_BASE")


def _normalize_base_url(raw_base: Optional[str]) -> str:
    """
    确保基础 URL 包含协议并以单个斜杠结尾，兼容 .env 中未写协议的情况
    """
    if not raw_base:
        raise RuntimeError("未配置 QWEATHER_API_BASE 环境变量")

    base = raw_base.strip()
    if not base.startswith(("http://", "https://")):
        base = f"https://{base.lstrip('/')}"

    if not base.endswith("/"):
        base = f"{base}/"

    return base


try:
    _QWEATHER_BASE_URL = _normalize_base_url(QWEATHER_API_BASE)
except RuntimeError as err:
    logger.error(f"[配置错误]{err}")
    _QWEATHER_BASE_URL = None


async def make_qweather_request(endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    向和风天气API发送请求
    :param endpoint: API端点路径（不包含基础URL）
    :param params: API请求参数
    :return: 成功返回JSON响应，失败返回None
    """
    if not _QWEATHER_BASE_URL:
        logger.error("QWEATHER_API_BASE 未正确配置，已跳过请求。")
        return None

    if not QWEATHER_API_KEY:
        logger.error("QWEATHER_API_KEY 未设置，已跳过请求。")
        return None

    safe_endpoint = endpoint.lstrip("/")
    url = urljoin(_QWEATHER_BASE_URL, safe_endpoint)

    headers = {
        "X-QW-Api-Key": QWEATHER_API_KEY
    }

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"QWeather请求: url={url}, params={params}")
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            logger.info(f"QWeather响应状态: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            logger.info(f"QWeather响应内容大小: {len(str(result))} 字符")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 状态错误: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"API 请求错误: {type(e).__name__}: {e}")
            return None


def format_warning(warning: Dict[str, Any]) -> str:
    """
    将天气预警数据格式化为可读字符串
    :param warning: 天气预警数据对象
    :return: 格式化预警信息
    """
    return f"""
    预警ID: {warning.get('id', '未知')}
    标题: {warning.get('headline', '未知')}
    发布时间: {warning.get('issuedTime', '未知')}
    开始时间: {warning.get('effectiveTime', '未知')}
    结束时间: {warning.get('expireTime', '未知')}
    预警类型: {warning.get('eventType', {}).get("name", "未知")}
    预警等级: {warning.get('severity', '未知')} ({warning.get('color', {}).get("code", "未知")})
    发布单位: {warning.get('senderName', '未知')}
    详细信息: {warning.get('description', '无详细信息')}
    """


def _convert_chinese_to_pinyin(chinese_text: str) -> str:
    """
    将中文城市名转换为拼音（全拼）

    Args:
        chinese_text: 中文城市名，如 "西宁"

    Returns:
        str: 拼音全拼，如 "xining"
    """
    try:
        # 使用 pypinyin 将中文转换为拼音
        pinyin_list = lazy_pinyin(chinese_text, style=Style.NORMAL)
        pinyin = ''.join(pinyin_list)
        logger.info(f"中文转拼音: {chinese_text} → {pinyin}")
        return pinyin
    except Exception as e:
        logger.error(f"中文转拼音失败: {chinese_text} - {str(e)}")
        return chinese_text  # 转换失败时返回原文本


async def _resolve_location(raw: Union[str, int]) -> str:
    """
    位置转换经纬度
    :param raw: 位置信息，支持以下格式：
                    - 中文城市名（如"北京"、"西宁"等，会自动转换为拼音再查找城市ID）
                    - 城市拼音（如"xining"表示西宁，会自动转换为城市ID）
                    - 城市ID（如"101010100"表示北京）
                    - 经纬度坐标（如"116.41,39.92"）
    :return: 经纬度字符串
    """
    text = str(raw).strip()
    # 经纬度直接透传
    if "," in text:
        logger.info(f"检测到经纬度")
        city_lon, city_lat = text.split(",")
        return f"{city_lat}/{city_lon}"
    # 数字ID转换经纬度
    if text.isdigit():
        logger.info(f"检测到城市ID，查找对应经纬度")
        lookup = await make_qweather_request("geo/v2/city/lookup", {"location": text, "lang": "zh"})
    # 中文 → 拼音
    elif any('\u4e00' <= ch <= '\u9fff' for ch in text):
        py = _convert_chinese_to_pinyin(text)
        logger.info(f"中文转拼音: {text} → {py}")
        lookup = await make_qweather_request("geo/v2/city/lookup", {"location": py, "lang": "zh"})
    # 拼音
    elif text.isalpha() and text.islower():
        py = text
        logger.info(f"检测到拼音，开始查找: {py}")
        lookup = await make_qweather_request("geo/v2/city/lookup", {"location": py, "lang": "zh"})
    else:
        logger.info(f"未识别的格式，原样使用: {text}")
        return text

    if not lookup or lookup.get("code") != "200":
        logger.warning(f"城市查找失败，回退原值: {text}")
        return text
    locations = lookup.get("location", [])
    if not locations:
        logger.info(f"无匹配结果，回退原值: {text}")
        return text
    chosen = next((loc for loc in locations if loc.get("type") == "city"), locations[0])
    city_lat = chosen.get("lat")
    city_lon = chosen.get("lon")
    logger.info(f"解析完成: {text} → 经度{city_lat}, 纬度{city_lon}")
    return f"{city_lat}/{city_lon}"


@mcp.tool()
async def get_weather_warning(location: Union[str, int]) -> str:
    """
    获取指定位置的天气灾害预警
    :param location: 位置信息，支持以下格式：
                    - 中文城市名（如"北京"、"西宁"等，会自动转换为拼音再查找城市ID）
                    - 城市拼音（如"xining"表示西宁，会自动转换为城市ID）
                    - 城市ID（如"101010100"表示北京）
                    - 经纬度坐标（如"116.41,39.92"）
    :return: 格式化预警信息
    """

    resolved = await _resolve_location(location)

    params = {
        "lang": "zh"
    }

    logger.info(f"调用 get_weather_warning | params={params}")
    data = await make_qweather_request(f"weatheralert/v1/current/{resolved}", params)

    if not data:
        logger.warning("get_weather_warning 返回空或失败")
        return "无法获取预警信息或API请求失败。"

    if data.get("metadata", {}).get("zeroResult"):
        logger.info(f"get_weather_warning 无活动预警 | location={location}")
        return f"当前位置 {location} 没有活动预警。"

    warnings = data.get("alerts", [])

    formatted_warnings = [format_warning(warning) for warning in warnings]
    joined = "\n---\n".join(formatted_warnings)
    logger.info(f"get_weather_warning 返回长度: {len(joined)} 字符")
    return joined


def _utc_to_local(raw: Any, fmt: str = "%Y-%m-%d") -> str:
    """把 2026-11-03T16:00Z 这种 UTC 时间转成北京时间字符串。"""
    text = str(raw or "")
    try:
        return (datetime.strptime(text, "%Y-%m-%dT%H:%MZ") + timedelta(hours=8)).strftime(fmt)
    except ValueError:
        return text or "未知"


def format_daily_forecast(daily: Dict[str, Any]) -> str:
    """
    将天气预报数据格式化为可读字符串

    参数:
        daily: 天气预报数据对象

    返回:
        格式化后的预报信息
    """
    humidity = daily.get('daytime', {}).get("humidity")
    humidity_str = f"{round(humidity * 100)}%" if humidity is not None else "未知"
    return f"""
日期: {_utc_to_local(daily.get('forecastStartTime'))}
日出: {_utc_to_local(daily.get('astro', {}).get("sunrise"), "%H:%M")}  
日落: {_utc_to_local(daily.get('astro', {}).get("sunset"), "%H:%M")}
最高温度: {daily.get('temperatureMax', {}).get("value", "未知")}°C  
最低温度: {daily.get('temperatureMin', {}).get("value", "未知")}°C
白天天气: {daily.get('daytime', {}).get("condition", {}).get("text", "未知")}  
夜间天气: {daily.get('nighttime', {}).get("condition", {}).get("text", "未知")} 
白天风向: {daily.get('daytime', {}).get("wind", {}).get("direction", {}).get("compass", "未知")}
白天蒲福风级: {daily.get('daytime', {}).get("wind", {}).get("scale", "未知")}
白天风速 ({daily.get('daytime', {}).get("wind", {}).get("speed", {}).get("value", "未知")}m/s)
夜间风向: {daily.get('nighttime', {}).get("wind", {}).get("direction", {}).get("compass", "未知")}
夜间蒲福风级: {daily.get('nighttime', {}).get("wind", {}).get("scale", "未知")}
夜间风速 ({daily.get('nighttime', {}).get("wind", {}).get("speed", {}).get("value", "未知")}m/s)
相对湿度: {humidity_str}
降水量: {daily.get('daytime', {}).get("precipitation", {}).get("amount", {}).get("value", "未知")}mm
紫外线指数: {daily.get('uvIndexMax', '未知')}
"""


@mcp.tool()
async def get_daily_forecast(location: Union[str, int], days: int = 3) -> str:
    """
    获取指定位置的天气预报

    参数：
        location (str): 位置信息，支持以下格式：
                       - 中文城市名（如"北京"、"西宁"等，会自动转换为拼音再查找城市ID）
                       - 城市拼音（如"xining"表示西宁，会自动转换为城市ID）
                       - 城市ID（如"101010100"表示北京）
                       - 经纬度坐标（如"116.41,39.92"）
        days (int): 预报天数，可选值为 1-10，默认为 3
    :return:格式化天气预报字符串
    """
    resolved = await _resolve_location(location)

    # 确保天数有效
    if days not in range(1, 11):
        logger.warning("get_daily_forecast 收到非法 days=%r，默认为3天", days)
        days = 3

    params = {
        "days": days,
        "lang": "zh"
    }

    logger.info(f"调用 get_daily_forecast | endpoint={resolved}, params={params}")
    data = await make_qweather_request(f"weather/v1/daily/{resolved}", params=params)

    if not data:
        logger.error("get_daily_forecast 返回空或失败")
        return "无法获取天气预报或API请求失败。"

    daily_list = data.get("days", [])
    joined = "\n---\n".join([format_daily_forecast(daily) for daily in daily_list])
    logger.info(f"get_daily_forecast 返回长度: {len(joined)} 字符")
    return joined

if __name__ == '__main__':
    logger.info("正在启动 MCP 天气服务器…")
    logger.info("提供工具: get_weather_warning, get_daily_forecast")
    logger.info("请确保环境变量 QWEATHER_API_KEY 已设置")
    logger.info("使用 Ctrl+C 停止服务器")

    # 初始化并运行服务器
    mcp.run(transport='stdio')
