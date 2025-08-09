import traceback

from agent import Agent
from llm import GeneralLLM
from tools import EDGARSearch, GoogleWebSearch, OpenAIWebSearch, ParseHtmlPage, RetrieveInformation


async def get_agent(model_name: str, parameters: dict, *args, **kwargs):
    max_turns = parameters.get("max_turns", 50)
    provider, model_key = model_name.split("/", 1)
    
    if provider == "openai":
        available_tools = {
            "web_search_preview": OpenAIWebSearch,
            "retrieve_information": RetrieveInformation,
            "parse_html_page": ParseHtmlPage,
            "edgar_search": EDGARSearch,
        }
        tool_mapping = {"google_web_search": "web_search_preview"}
    else:
        available_tools = {
            "google_web_search": GoogleWebSearch,
            "retrieve_information": RetrieveInformation,
            "parse_html_page": ParseHtmlPage,
            "edgar_search": EDGARSearch,
        }
        tool_mapping = {}

    selected_tools = {}
    for tool in parameters.get("tools", available_tools.keys()):
        actual_tool = tool_mapping.get(tool, tool)
        if actual_tool not in available_tools:
            raise Exception(
                f"Tool {actual_tool} not found in tools. Available tools: {available_tools.keys()}"
            )
        selected_tools[actual_tool] = available_tools[actual_tool]()

    llm = GeneralLLM(
        provider=provider,
        model_name=model_key,
        max_tokens=parameters.get("max_output_tokens", 16384),
        temperature=parameters.get("temperature", 0.0),
    )

    agent = Agent(llm=llm, tools=selected_tools, max_turns=max_turns)

    return agent
