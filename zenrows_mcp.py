import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import MCPServerAdapter

# Claude drives every agent. ANTHROPIC_API_KEY is read from the environment.
llm = LLM(model="anthropic/claude-sonnet-4-6", temperature=0.1)

# Product pages to compare. Confirm they resolve first.
TARGETS = [
    "https://www.ikea.com/us/en/p/markus-office-chair-vissle-dark-gray-90289172/",
    "https://www.amazon.com/dp/B00FS3VJAO",  # comparable office chair
]

# Hosted ZenRows MCP server. No Node, no npx, just your API key.
SERVER_PARAMS = {
    "url": "https://mcp.zenrows.com/mcp",
    "transport": "streamable-http",
    "headers": {"Authorization": f"Bearer {os.environ['ZENROWS_API_KEY']}"},
}


class NullStrippedTool(BaseTool):
    """Wraps an MCP tool and removes None arguments before the call.

    The MCP adapter fills unset optional fields with null, which the ZenRows
    server rejects. Stripping None here keeps only the values that were set.
    """

    inner: Any = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return self.inner._run(*args, **clean)


def wrap(tools):
    return [
        NullStrippedTool(
            name=t.name,
            description=t.description,
            args_schema=t.args_schema,
            inner=t,
        )
        for t in tools
    ]


def build_and_run():
    # The MCP tools are only live inside this block.
    with MCPServerAdapter(SERVER_PARAMS, "scrape") as mcp_tools:
        scrape_tools = wrap(mcp_tools)

        researcher = Agent(
            role="Retail Product Data Extractor",
            goal=(
                "Extract ONLY the visible product name and current price from "
                "each product page. No inference, no assumptions."
            ),
            backstory=(
                "You are a deterministic extraction system. If a value is not "
                "visible in the scraped content, it does not exist."
            ),
            tools=scrape_tools,
            llm=llm,
            verbose=True,
        )

        analyst = Agent(
            role="Retail Pricing Analyst",
            goal=(
                "Compare product pricing across two retailers using ONLY "
                "scraped data. No external knowledge or assumptions."
            ),
            backstory=(
                "You ONLY use provided scraped outputs. Missing fields must be "
                "marked as 'NOT FOUND IN SCRAPE'."
            ),
            llm=llm,
            verbose=True,
        )

        research_task = Task(
            description=(
                "Scrape each product page below and extract the product name "
                "and its current price. Use only what is visible on the page. "
                "If a value is not present, write NOT FOUND IN SCRAPE.\n\n"
                "When you call the scrape tool, enable JavaScript rendering and "
                "premium proxy, and wait for the page content to load, since "
                "these pages render product data client-side.\n\n"
                + "\n".join(TARGETS)
            ),
            expected_output=(
                "Product name and current price for each URL, from scraped "
                "content only."
            ),
            agent=researcher,
        )

        analysis_task = Task(
            description=(
                "Compare the two products using only the researcher's scraped "
                "data. Do not add outside knowledge. Mark anything missing as "
                "NOT FOUND IN SCRAPE.\n\n"
                "Produce:\n"
                "- a table of product name and price per retailer\n"
                "- which product is cheaper\n"
                "- one or two insights drawn from the data"
            ),
            expected_output="A short pricing comparison: a table and a brief takeaway.",
            agent=analyst,
            context=[research_task],
        )

        crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        print("\n\n===== FINAL OUTPUT =====\n")
        print(result)


if __name__ == "__main__":
    build_and_run()