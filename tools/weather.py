#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    :   weather.py
@Time    :   2026/03/10 22:38:13
@Author  :   lemonnmin
@Version :   1.0
@Desc    :   可供调用的天气查询工具
"""
import requests

class WeatherTool:
    """
    天气查询工具
    """

    name = "weather"
    description = "查询指定城市的实时天气"

    WEATHER_TRANSLATE = {
        "Overcast": "阴天",
        "Partly cloudy": "多云",
        "Cloudy": "阴",
        "Sunny": "晴",
        "Clear": "晴朗",
        "Rain": "雨",
        "Light rain": "小雨",
        "Heavy rain": "大雨",
        "Thunderstorm": "雷暴",
        "Snow": "雪"
    }

    def get_weather(self, city: str):

        url = f"https://wttr.in/{city}?format=j1"

        response = requests.get(url, timeout=20)
        response.raise_for_status()

        data = response.json()

        current = data["current_condition"][0]

        weather = current["weatherDesc"][0]["value"]
        temp = current["temp_C"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]

        weather_cn = self.WEATHER_TRANSLATE.get(weather, weather)

        return {
            "city": city,
            "weather": weather_cn,
            "temperature": f"{temp}°C",
            "humidity": f"{humidity}%",
            "wind_speed": f"{wind} km/h"
        }

    def run(self, city: str):
        try:

            weather_data = self.get_weather(city)

            result = [{
                "rank": 1,
                "content": f"{weather_data['city']} 当前天气 {weather_data['weather']}，温度 {weather_data['temperature']}，湿度 {weather_data['humidity']}，风速 {weather_data['wind_speed']}",
                "source": "weather_api",
                "metadata": weather_data
            }]

            return {
                "tool": self.name,
                "success": True,
                "query": city,
                "count": 1,
                "data": result,
                "error": None
            }

        except Exception as e:

            return {
                "tool": self.name, 
                "success": False,
                "query": city,
                "count": 0,
                "data": [],
                "error": str(e)
            }

# 测试
if __name__ == "__main__":

    weather = WeatherTool()

    print(weather.run("台湾省"))