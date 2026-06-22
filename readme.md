# competitor_Intelligence_agent_with_crewal_zenrows

A two-agent CrewAI crew that scrapes competitor pages with ZenRows and turns the
results into a structured comparison, powered by Claude. A researcher agent
scrapes each target page through a ZenRows tool, and an analyst agent compares
the scraped data and writes the brief. The analyst works only from scraped
content and marks anything missing as `NOT FOUND IN SCRAPE`, so the output never
contains invented numbers.

## How it works

1. The researcher agent calls a ZenRows-backed tool to fetch each target page.
   ZenRows handles JavaScript rendering and anti-bot bypass, so protected pages
   return real content instead of a challenge screen.
2. The analyst agent receives the researcher's output and produces a comparison
   table plus a short takeaway, using only what was scraped.
3. Claude (`claude-sonnet-4-6`) drives both agents through CrewAI.

## Prerequisites

- Python 3.10 to 3.13 (CrewAI requires this range).
- A ZenRows API key. Create an account at zenrows.com and copy the key from the
  dashboard.
- An Anthropic API key. Generate one in the Anthropic Console.

## Setup

Install the dependencies into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with both keys:

```dotenv
ANTHROPIC_API_KEY=your_anthropic_key_here
ZENROWS_API_KEY=your_zenrows_key_here
```

To change what gets compared, edit the `TARGETS` list at the top of the file.

## Integration paths

The repo includes two ways to give the researcher ZenRows access.
- Custom tool
- MCP server 

## Configuration

- `TARGETS`: the list of URLs to scrape. Use product or pricing pages, not
  category or search pages, which lazy-load their content.
- Model: change the `LLM(model=...)` string to switch Claude models, for example
  `anthropic/claude-opus-4-8`.
- ZenRows parameters: `js_render` runs a headless browser, `premium_proxy` routes
  through residential IPs, and `wait` gives client-side content time to load.
  `wait` requires `js_render`.
