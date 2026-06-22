from crewai_tools import ScrapeWebsiteTool

tool = ScrapeWebsiteTool(
    website_url="https://www.yelp.com"
)

text = tool.run()
print(text)