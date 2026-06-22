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
# Scraping Tool (autoparse -> structured JSON)
# -----------------------------
@tool("Scrape competitor page as structured JSON")
def scrape_page_structured(url: str) -> str:
    """Fetch a single competitor product or pricing page and return auto-parsed
    structured data as a JSON string. Pass one URL as a string. Use this when
    you need discrete fields such as product name and price rather than prose."""
    response = zenrows.get(
        url,
        params={
            "js_render": True,
            "premium_proxy": True,
            "autoparse": True,  # replaces response_type; autoparse returns JSON
            "wait": 3000,       # let client-side content (e.g. IKEA) load. requires js_render
        },
    )
    response.raise_for_status()
    return response.text

# -----------------------------
# LLM
# -----------------------------
llm = LLM(model="anthropic/claude-sonnet-4-6", temperature=0.1)

# -----------------------------
# TARGETS (product pages). Confirm they resolve first.
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
        "Extract ONLY the product name and price present in the scraped JSON. "
        "No inference, no assumptions."
    ),
    backstory=(
        "You are a deterministic extraction system. "
        "If a field is not in the JSON, it does not exist."
    ),
    tools=[scrape_page_structured],
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
        "Scrape each product page below and read the returned JSON. Extract the "
        "product name and its current price. Use only what is in the JSON. If a "
        "value is not present, write NOT FOUND IN SCRAPE.\n\n"
        + "\n".join(TARGETS)
    ),
    expected_output="Product name and current price for each URL, from the scraped JSON only.",
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