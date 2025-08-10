import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any

from agents import Agent, Runner, WebSearchTool, function_tool
import aiohttp
import backoff
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
import logging

logging.basicConfig(level=logging.INFO)
tool_logger = logging.getLogger(__name__)


def is_429(exception):
    is429 = (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
        or "429" in str(exception)
    )
    if is429:
        tool_logger.error(f"429 error: {exception}")
    return is429


def retry_on_429(func):
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        max_tries=8,
        base=2,
        factor=3,
        jitter=backoff.full_jitter,
        giveup=lambda e: not is_429(e),
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


conversation_data_storage = {}


@function_tool
async def edgar_search(
    query: str,
    form_types: list[str],
    ciks: list[str],
    start_date: str,
    end_date: str,
    page: str,
    top_n_results: int,
) -> str:
    """
    Search the EDGAR Database through the SEC API.
    You should provide a query, a list of form types, a list of CIKs, a start date, an end date, a page number, and a top N results.
    The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing.
    
    Args:
        query: The keyword or phrase to search, such as 'substantial doubt' OR 'material weakness'
        form_types: Limits search to specific SEC form types (e.g., ['8-K', '10-Q']) list of strings. Default is None (all form types)
        ciks: Filters results to filings by specified CIKs, type list of strings. Default is None (all filers).
        start_date: Start date for the search range in yyyy-mm-dd format. Used with endDate to define the date range. Example: '2024-01-01'. Default is 30 days ago
        end_date: End date for the search range, in the same format as startDate. Default is today
        page: Pagination for results. Default is '1'
        top_n_results: The top N results to return after the query. Useful if you are not sure the result you are looking for is ranked first after your query.
    """
    sec_api_key = os.getenv("SEC_API_KEY")
    if sec_api_key is None:
        raise Exception("SEC_API_KEY is not set")
    
    sec_api_url = "https://api.sec-api.io/full-text-search"
    
    @retry_on_429
    async def _execute_search():
        payload = {
            "query": query,
            "formTypes": form_types,
            "ciks": ciks,
            "startDate": start_date,
            "endDate": end_date,
            "page": page,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": sec_api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                sec_api_url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()

        return result.get("filings", [])[: int(top_n_results)]
    
    try:
        results = await _execute_search()
        return json.dumps(results)
    except Exception as e:
        tool_logger.error(f"SEC API error: {e}")
        raise


@function_tool
async def parse_html_page(url: str, key: str) -> str:
    """
    Parse an HTML page. This tool is used to parse the HTML content of a page and saves the content outside of the conversation to avoid context window issues.
    You should provide both the URL of the page to parse, as well as the key you want to use to save the result in the agent's data structure.
    The data structure is a dictionary.
    
    Args:
        url: The URL of the HTML page to parse
        key: The key to use when saving the result in the conversation's data structure (dict).
    """
    headers = {"User-Agent": "ValsAI/antoine@vals.ai"}
    
    @retry_on_429
    async def _parse_html_page(url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url, headers=headers, timeout=60
                ) as response:
                    response.raise_for_status()
                    html_content = await response.text()
            except Exception as e:
                if len(str(e)) == 0:
                    raise TimeoutError(
                        "Timeout error when parsing HTML page after 60 seconds. The URL might be blocked or the server is taking too long to respond."
                    )
                else:
                    raise e

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
    
    try:
        text_output = await _parse_html_page(url)
        
        tool_result = ""
        if key in conversation_data_storage:
            tool_result = "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
        tool_result += (
            f"SUCCESS: The result has been saved to the data storage under the key: {key}."
            + "\n"
        )

        conversation_data_storage[key] = text_output

        keys_list = "\n".join(conversation_data_storage.keys())
        tool_result += (
            f"""
        The data_storage currently contains the following keys:
        {keys_list}
        """.strip()
            + "\n"
        )

        return tool_result
    except Exception as e:
        tool_logger.error(f"HTML parsing error: {e}")
        raise


@function_tool(strict_mode=False)
async def retrieve_information(
    prompt: str,
    input_character_ranges: dict | None = None,
) -> str:
    """
    Retrieve information from the conversation's data structure (dict) and allow character range extraction.
    
    IMPORTANT: Your prompt MUST include at least one key from the data storage using the exact format: {{key_name}}
    
    For example, if you want to analyze data stored under the key "financial_report", your prompt should look like:
    "Analyze the following financial report and extract the revenue figures: {{financial_report}}"
    
    The {{key_name}} will be replaced with the actual content stored under that key before being sent to the LLM.
    If you don't use this exact format with double braces, the tool will fail to retrieve the information.
    
    You can optionally specify character ranges for each document key to extract only portions of documents. That can be useful to avoid token limit errors or improve efficiency by selecting only part of the document.
    For example, if "financial_report" contains "Annual Report 2023" and you specify a range [1, 5] for that key,
    only "nnual" will be inserted into the prompt.
    
    The output is the result from the LLM that receives the prompt with the inserted data.
    
    Args:
        prompt: The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder.
        input_character_ranges: A dictionary mapping document keys to their character ranges. Each range should be an array where the first element is the start index and the second element is the end index. Can be used to only read portions of documents. By default, the full document is used. To use the full document, set the range to an empty list [].
    """
    import re
    
    if input_character_ranges is None:
        input_character_ranges = {}

    # Verify that the prompt contains at least one placeholder in the correct format
    if not re.search(r"{{[^{}]+}}", prompt):
        raise ValueError(
            "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. Please try again with the correct format."
        )

    # Find all keys in the prompt
    keys = re.findall(r"{{([^{}]+)}}", prompt)
    formatted_data = {}

    # Apply character range to each document before substitution
    for key in keys:
        if key not in conversation_data_storage:
            raise KeyError(
                f"ERROR: The key '{key}' was not found in the data storage. Available keys are: {', '.join(conversation_data_storage.keys())}"
            )

        # Extract the specified character range from the document if provided
        doc_content = conversation_data_storage[key]

        if key in input_character_ranges:
            char_range = input_character_ranges[key]
            if len(char_range) == 0:
                formatted_data[key] = doc_content
            elif len(char_range) != 2:
                raise ValueError(
                    f"ERROR: The character range for key '{key}' must be an list with two elements or an empty list. Please try again with the correct format."
                )
            else:
                start_idx = int(char_range[0])
                end_idx = int(char_range[1])
                formatted_data[key] = doc_content[start_idx:end_idx]
        else:
            # Use the full document if no range is specified
            formatted_data[key] = doc_content

    # Convert {{key}} format to Python string formatting
    formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)

    try:
        final_prompt = formatted_prompt.format(**formatted_data)
    except KeyError as e:
        raise KeyError(
            f"ERROR: The key {str(e)} was not found in the data storage. Available keys are: {', '.join(conversation_data_storage.keys())}"
        )

    from openai import AsyncOpenAI
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for retrieve_information tool")
    
    client = AsyncOpenAI(api_key=openai_api_key)
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=4096,
            temperature=0.0,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        tool_logger.error(f"LLM retrieval error: {e}")
        raise


async def run_tests_parallel(
    output_dir,
    questions=[],
    model_name="openai/gpt-4o",
    max_concurrent=5,
    save_results=False,
    parameters={},
):
    """Run multiple questions in parallel using the OpenAI Agents SDK"""
    
    tools = []
    
    if "google_web_search" in parameters.get("tools", []):
        tools.append(WebSearchTool())
    
    available_tools = {
        "edgar_search": edgar_search,
        "parse_html_page": parse_html_page,
        "retrieve_information": retrieve_information,
    }
    
    for tool_name in parameters.get("tools", []):
        if tool_name in available_tools:
            tools.append(available_tools[tool_name])
    
    agent = Agent(
        name="Finance Agent",
        instructions="""You are a finance agent that helps answer questions about companies, financial statements, and SEC filings.

You have access to the following tools:
- web_search_preview: Search the web for information
- edgar_search: Search the SEC's EDGAR database for filings
- parse_html_page: Parse and extract content from web pages
- retrieve_information: Access stored information from previous steps

When you have a final answer, format it clearly and provide sources when possible.

FINAL ANSWER: [Your comprehensive answer here]

{sources: ["source1", "source2"]}""",
        tools=tools,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question):
        async with semaphore:
            try:
                global conversation_data_storage
                conversation_data_storage = {}
                
                result = await Runner.run(
                    agent, 
                    input=question,
                    max_turns=parameters.get("max_turns", 50)
                )
                return result.final_output
            except Exception as e:
                tool_logger.error(f"Error processing question '{question}': {e}")
                return f"Error: {str(e)}"

    tasks = [process_question(question) for question in questions]

    results = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for i, (question, result) in enumerate(zip(questions, results)):
        if isinstance(result, Exception):
            formatted_results.append(
                {"question": question, "success": False, "error": str(result)}
            )
        else:
            formatted_results.append(
                {"question": question, "success": True, "result": result}
            )

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"results_openai_agents_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(formatted_results, f, indent=2)

    return formatted_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the OpenAI Agents SDK harness for the finance agent benchmark"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Maximum number of output tokens for completion generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--questions", type=str, nargs="+", help="List of questions to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model to use to generate completions",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="Path to file containing questions (one per line)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=[
            "google_web_search",
            "retrieve_information",
            "parse_html_page",
            "edgar_search",
        ],
        choices=[
            "google_web_search",
            "retrieve_information",
            "parse_html_page",
            "edgar_search",
        ],
        help="List of tools to make available to the agent",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of turns for the agent to take before stopping",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results to.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel requests to make to the model",
    )
    args = parser.parse_args()

    # Set logging level
    logging_level = args.log_level
    tool_logger.setLevel(logging_level)

    # Get questions from file if provided, otherwise use command line args
    if args.question_file:
        with open(args.question_file, "r") as f:
            questions = [line.strip() for line in f if line.strip()]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception(
            "No questions provided. One of --question-file or --questions must be used."
        )

    parameters = {
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "tools": args.tools,
    }

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    asyncio.run(
        run_tests_parallel(
            output_dir=args.results_dir,
            questions=questions,
            model_name=args.model,
            max_concurrent=args.parallelism,
            save_results=True,
            parameters=parameters,
        )
    )


if __name__ == "__main__":
    main()
