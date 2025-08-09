import traceback

from agent import Agent
from llm import GeneralLLM
from tools import EDGARSearch, GoogleWebSearch, ParseHtmlPage, RetrieveInformation


async def get_agent(model_name: str, parameters: dict, *args, **kwargs):
    max_turns = parameters.get("max_turns", 50)
    available_tools = {
        "google_web_search": GoogleWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "edgar_search": EDGARSearch,
    }

    provider, model_key = model_name.split("/", 1)

    selected_tools = {}
    for tool in parameters.get("tools", available_tools.keys()):
        if tool not in available_tools:
            raise Exception(
                f"Tool {tool} not found in tools. Available tools: {available_tools.keys()}"
            )
        if tool == "google_web_search" and provider == "openai" and model_key == "gpt-5":
            continue
        selected_tools[tool] = available_tools[tool]()

    llm = GeneralLLM(
        provider=provider,
        model_name=model_key,
        max_tokens=parameters.get("max_output_tokens", 16384),
        temperature=parameters.get("temperature", 0.0),
    )

    agent = Agent(llm=llm, tools=selected_tools, max_turns=max_turns)

    return agent
