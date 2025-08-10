# OpenAI Agents SDK Harness for Finance Agent Benchmark

This harness converts the finance agent benchmark to use the OpenAI Agents SDK instead of the custom agent implementation.

## Key Changes

- **Web Search**: Replaced SERP API with OpenAI's `WebSearchTool`
- **Function Tools**: Converted all finance agent tools to use `@function_tool` decorator:
  - `edgar_search`: Search SEC EDGAR database
  - `parse_html_page`: Parse and extract content from web pages  
  - `retrieve_information`: Access stored information from previous steps
- **Agent Framework**: Uses OpenAI Agents SDK `Agent` and `Runner` instead of custom agent loop
- **Parallel Processing**: Maintains async/await support for concurrent question processing

## Usage

```bash
# Basic usage with web search only
python openai_agents_harness.py --questions "What is Apple's revenue?" --tools "google_web_search"

# Full finance agent with all tools
python openai_agents_harness.py --question-file sample_questions.txt --tools "google_web_search" "edgar_search" "parse_html_page" "retrieve_information"

# Parallel processing
python openai_agents_harness.py --questions "Question 1" "Question 2" --parallelism 2
```

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI Agents SDK and retrieve_information tool
- `SEC_API_KEY`: Required for edgar_search tool

## Dependencies

The harness requires the following additional packages:
- `openai-agents`: OpenAI Agents SDK
- `beautifulsoup4`: HTML parsing
- `aiohttp`: Async HTTP requests
- `backoff`: Retry logic for API calls
- `tqdm`: Progress bars

Install with:
```bash
pip install openai-agents beautifulsoup4 aiohttp backoff tqdm
```
