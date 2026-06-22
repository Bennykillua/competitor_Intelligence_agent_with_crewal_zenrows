import os

from crewai import Agent
from crewai.tools import tool
from zenrows import ZenRowsClient

# retries=2 retries failed requests (429 and 5xx) with exponential back-off
client = ZenRowsClient(os.environ["ZENROWS_API_KEY"], retries=2)


@tool("Scrape competitor page as Markdown")
def scrape_page_markdown(url: str) -> str:
    """Fetch the full content of a single competitor web page and return it as
    Markdown. Pass one URL as a string. Use this to read pricing pages, product
    pages, and positioning copy from sites that block standard scrapers."""
    response = client.get(
        url,
        params={
            "js_render": "true",
            "premium_proxy": "true",
            "response_type": "markdown",
        },
    )
    response.raise_for_status()
    return response.text

researcher = Agent(
    role="Competitor Pricing Researcher",
    goal=(
        "Scrape each competitor pricing page and extract product names, "
        "prices, and positioning claims"
    ),
    backstory=(
        "You read competitor pricing pages and report exactly what is on the "
        "page. You never invent numbers."
    ),
    tools=[scrape_page_markdown],
    verbose=True,
)