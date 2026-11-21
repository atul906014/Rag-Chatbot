import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from config import TAVILY_API_KEY
mcp = FastMCP("NewsTools")
tavily = TavilyClient(api_key=TAVILY_API_KEY)


@mcp.tool()
def get_current_date(timezone: str = "UTC") -> dict:
    """Get the current date and time. Call this first whenever the user says
    'today', 'latest', 'this week', or anything time-relative.
    timezone: IANA name such as 'Asia/Kolkata' or 'UTC'."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return {"error": f"Unknown timezone: {timezone}"}
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": timezone,
    }


@mcp.tool()
def get_current_news(query: str, max_results: int = 5, time_range: str = "week") -> list[dict]:
    """Search recent news on a topic using Tavily.
    query: the topic, e.g. 'Azure AI Foundry updates'.
    time_range: 'day', 'week', 'month' or 'year'."""
    resp = tavily.search(
        query=query,
        topic="news",
        max_results=min(max_results, 10),
        time_range=time_range,
    )
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "summary": r.get("content"),
            "published": r.get("published_date"),
        }
        for r in resp.get("results", [])
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")