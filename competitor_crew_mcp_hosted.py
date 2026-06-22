

import os

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter

# Claude drives every agent. ANTHROPIC_API_KEY is read from the environment.
llm = LLM(model="anthropic/claude-sonnet-4-6", temperature=0.2)

# Competitor pages to analyse. Replace with the market you track.
TARGETS = [
    "https://asana.com/pricing",
    "https://monday.com/pricing",
]

# Hosted ZenRows MCP server. No Node, no npx, just your API key.
SERVER_PARAMS = {
    "url": "https://mcp.zenrows.com/mcp",
    "transport": "streamable-http",
    "headers": {"Authorization": f"Bearer {os.environ['ZENROWS_API_KEY']}"},
}


def build_and_run():
    # The MCP tools are only live inside this block, so build the crew and
    # call kickoff() here, not outside.
    with MCPServerAdapter(SERVER_PARAMS, "scrape") as mcp_tools:
        researcher = Agent(
            role="Competitor Pricing Researcher",
            goal=(
                "Scrape each competitor pricing page and extract every plan "
                "name, its price, and any positioning claim on the page"
            ),
            backstory=(
                "You read competitor pricing pages and report exactly what is "
                "on the page. You never invent numbers."
            ),
            tools=mcp_tools,
            llm=llm,
            verbose=True,
        )

        analyst = Agent(
            role="Competitive Intelligence Analyst",
            goal="Turn the researcher's scraped data into a structured competitive brief",
            backstory=(
                "You compare pricing and positioning across competitors and "
                "write a brief a product team can act on. You analyse; you do "
                "not scrape."
            ),
            llm=llm,
            verbose=True,
        )

        research_task = Task(
            description=(
                "Scrape each competitor pricing page below and extract every "
                "plan name, its price, and any positioning claim on the page. "
                "If a page is JavaScript-heavy or blocks the request, enable "
                "JavaScript rendering and premium proxy on the scrape.\n\n"
                + "\n".join(TARGETS)
            ),
            expected_output=(
                "Per competitor: a list of plan names, prices, and positioning "
                "claims taken verbatim from each page."
            ),
            agent=researcher,
        )

        analysis_task = Task(
            description=(
                "Using the researcher's findings, write a competitive brief "
                "that compares the competitors on price, plan structure, and "
                "positioning. Name the cheapest entry plan, the most expensive "
                "plan, and one pricing gap a new product could occupy. Present "
                "a comparison table followed by three takeaways."
            ),
            expected_output="A brief with a comparison table and three takeaways.",
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
        print("\n\n===== FINAL BRIEF =====\n")
        print(result)


if __name__ == "__main__":
    build_and_run()