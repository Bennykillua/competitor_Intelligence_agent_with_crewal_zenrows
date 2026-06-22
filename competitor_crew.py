
import os
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from zenrows import ZenRowsClient

# -----------------------------
# ZenRows Client
# -----------------------------
zenrows = ZenRowsClient(os.environ["ZENROWS_API_KEY"], retries=2)

# -----------------------------
# Scraping Tool (wait added so client-side content renders)
# -----------------------------
@tool("Scrape page as Markdown")
def scrape_page_markdown(url: str) -> str:
    """Scrape a page and return it as Markdown. Return only visible scraped content."""
    response = zenrows.get(
        url,
        params={
            "js_render": True,
            "premium_proxy": True,
            "response_type": "markdown",
            "wait": 3000,  # wait 3s for client-side content to load. requires js_render
        },
    )
    response.raise_for_status()
    return response.text

# -----------------------------
# LLM
# -----------------------------
llm = LLM(model="anthropic/claude-sonnet-4-6", temperature=0.1)

# -----------------------------
# TARGETS (product pages, not category pages)
# Swap these for two comparable products. Confirm they resolve first.
# -----------------------------
TARGETS = [
    "https://www.ikea.com/us/en/p/markus-office-chair-vissle-dark-gray-90289172/",
    "https://www.amazon.com/dp/B00FS3VJAO",  # comparable office chair
]

# -----------------------------
# RESEARCHER (STRICT SCRAPER)
# -----------------------------
researcher = Agent(
    role="Retail Product Data Extractor",
    goal=(
        "Extract ONLY visible product name and price from retail product pages. "
        "No inference, no assumptions."
    ),
    backstory=(
        "You are a deterministic extraction system. "
        "If data is not visible in scraped content, it does not exist."
    ),
    tools=[scrape_page_markdown],
    llm=llm,
    verbose=True,
)

# -----------------------------
# ANALYST (STRICT GROUNDED)
# -----------------------------
analyst = Agent(
    role="Retail Pricing Analyst",
    goal=(
        "Compare product pricing across two retailers using ONLY scraped data. "
        "No external knowledge or assumptions."
    ),
    backstory=(
        "You ONLY use provided scraped outputs. "
        "Missing fields must be marked as 'NOT FOUND IN SCRAPE'."
    ),
    llm=llm,
    verbose=True,
)

# -----------------------------
# TASK 1: SCRAPING
# -----------------------------
research_task = Task(
    description=(
        "Scrape each product page below and extract the product name and its "
        "current price. Use only what is visible on the page. If a value is not "
        "present, write NOT FOUND IN SCRAPE.\n\n"
        + "\n".join(TARGETS)
    ),
    expected_output="Product name and current price for each URL, from scraped content only.",
    agent=researcher,
)

# -----------------------------
# TASK 2: ANALYSIS
# -----------------------------
analysis_task = Task(
    description=(
        "Compare the two products using only the researcher's scraped data. "
        "Do not add outside knowledge. Mark anything missing as NOT FOUND IN SCRAPE.\n\n"
        "Produce:\n"
        "- a table of product name and price per retailer\n"
        "- which product is cheaper\n"
        "- one or two insights drawn from the data"
    ),
    expected_output="A short pricing comparison: a table and a brief takeaway.",
    agent=analyst,
    context=[research_task],
)

# -----------------------------
# CREW
# -----------------------------
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True,
)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n\n===== FINAL OUTPUT =====\n")
    print(result)