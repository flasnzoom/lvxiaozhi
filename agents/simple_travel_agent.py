from langchain_openai import ChatOpenAI
from typing import Any, Dict, List
import logging
from datetime import datetime
import sys
import os

logger = logging.getLogger(__name__)

# 添加backend目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.langgraph_config import langgraph_config as config

class SimpleTravelAgent:
    """简化版"""

    def __init__(self):
        """初始化智能体"""
        llm_config = config.get_llm_config()
        self.llm = ChatOpenAI(**llm_config)

    def run_travel_planning(self, travel_request: Dict[str, Any]) -> Dict[str, Any]:
        """运行简易规划"""

        try:
            logger.info("开始简化版旅行规划...")

            # 提取信息
            destination = travel_request.get("destination", "")
            duration = travel_request.get("duration", 3)
            budget_range = travel_request.get("budget_range", "中等预算")
            interests = travel_request.get("interests", [])
            group_size = travel_request.get("group_size", 1)
            travel_dates = travel_request.get("travel_dates", "")

            # 构建提示词
            prompt = self._build_prompt(destination,duration,budget_range,interests,group_size,travel_dates)

            logger.info("正在生成旅行规划...")

            # 调用模型
            response = self.llm.invoke(prompt)
            plan_content = response.content

            logger.info("完成规划")

            # 返回结果
            result = {
                "success": True,
                "travel_plan": {
                    "destination": destination,
                    "duration": duration,
                    "budget_range": budget_range,
                    "interests": interests,
                    "group_size": group_size,
                    "travel_dates": travel_dates,
                    "planning_method": "简化版AI规划",
                    "generated_at": datetime.now().isoformat(),
                    "content": plan_content
                },
                "agent_outputs": {
                    "simple_agent": {
                        "status": "completed",
                        "response": plan_content,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                "total_iterations": 1,
                "planning_complete": True
            }

            return result

        except Exception as e:
            logger.error(f"简化版规划失败: {str(e)}")
            return {
                "success": False,
                "error": f"简化版规划失败: {str(e)}",
                "travel_plan": {},
                "agent_outputs": {},
                "total_iterations": 0,
                "planning_complete": False
            }

    def _build_prompt(self, destination: str, duration: int, budget_range: str,
                      interests: List[str], group_size: int, travel_dates: str) -> str:
        interests_str = "、".join(interests) if interests else "无特殊偏好"

        prompt = f"""
你是一个专业的旅行规划师，请为以下需求制定详细的旅行计划：

📍 目的地: {destination}
📅 旅行时长: {duration}天
💰 预算范围: {budget_range}
👥 团队人数: {group_size}人
🎯 兴趣爱好: {interests_str}
📆 旅行日期: {travel_dates}

请提供一个详细的旅行规划，包括：

## 🌍 行程概览
- 旅行主题和特色
- 推荐的旅行节奏

## 📅 日程安排
- 每日详细行程
- 主要景点和活动
- 用餐建议

## 💰 预算估算
- 住宿费用预估
- 餐饮费用预估
- 交通费用预估
- 景点门票预估
- 总预算范围

## 🏨 住宿推荐
- 推荐住宿区域
- 住宿类型建议
- 性价比推荐

## 🚗 交通指南
- 到达方式
- 当地交通
- 交通费用

## 🍽️ 美食推荐
- 当地特色美食
- 推荐餐厅
- 用餐预算

## 📝 实用贴士
- 最佳旅行时间
- 注意事项
- 实用建议

请确保规划内容详细、实用，符合预算范围和兴趣偏好。
"""

        return prompt


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    plan_agent = SimpleTravelAgent()
    travel_request = {
        "destination": "成都",
        "duration": 3,
        "budget_range": "",
        "interests": ["美食"],
        "group_size": 2,
        "travel_dates": "11-01~03",
    }
    result = plan_agent.run_travel_planning(travel_request)
    print(result)
