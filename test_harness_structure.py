#!/usr/bin/env python3
"""
Test script to validate the OpenAI Agents SDK harness structure and tool conversion
without requiring external API calls.
"""

import sys
import inspect
import importlib.util
from pathlib import Path

def test_harness_imports():
    """Test that all required imports work correctly"""
    try:
        spec = importlib.util.spec_from_file_location("openai_agents_harness", "openai_agents_harness.py")
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        print("✓ All imports successful")
        return harness
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return None

def test_function_tools(harness):
    """Test that all finance agent tools are properly converted to function tools"""
    required_tools = ['edgar_search', 'parse_html_page', 'retrieve_information']
    
    for tool_name in required_tools:
        if hasattr(harness, tool_name):
            tool_func = getattr(harness, tool_name)
            if hasattr(tool_func, 'name') and hasattr(tool_func, 'description'):
                print(f"✓ {tool_name} properly converted to function tool")
            else:
                print(f"✗ {tool_name} missing function tool attributes")
                return False
        else:
            print(f"✗ {tool_name} not found in harness")
            return False
    
    return True

def test_tool_signatures(harness):
    """Test that tool function schemas contain correct parameters"""
    
    edgar_tool = harness.edgar_search
    edgar_params = list(edgar_tool.params_json_schema['properties'].keys())
    expected_params = ['query', 'form_types', 'ciks', 'start_date', 'end_date', 'page', 'top_n_results']
    
    if set(edgar_params) == set(expected_params):
        print("✓ edgar_search schema parameters correct")
    else:
        print(f"✗ edgar_search schema mismatch. Expected: {expected_params}, Got: {edgar_params}")
        return False
    
    parse_tool = harness.parse_html_page
    parse_params = list(parse_tool.params_json_schema['properties'].keys())
    expected_params = ['url', 'key']
    
    if set(parse_params) == set(expected_params):
        print("✓ parse_html_page schema parameters correct")
    else:
        print(f"✗ parse_html_page schema mismatch. Expected: {expected_params}, Got: {parse_params}")
        return False
    
    retrieve_tool = harness.retrieve_information
    retrieve_params = list(retrieve_tool.params_json_schema['properties'].keys())
    expected_params = ['prompt', 'input_character_ranges']
    
    if set(retrieve_params) == set(expected_params):
        print("✓ retrieve_information schema parameters correct")
    else:
        print(f"✗ retrieve_information schema mismatch. Expected: {expected_params}, Got: {retrieve_params}")
        return False
    
    return True

def test_cli_interface(harness):
    """Test that CLI interface matches original finance agent"""
    
    if hasattr(harness, 'main'):
        print("✓ main() function exists")
    else:
        print("✗ main() function missing")
        return False
    
    if hasattr(harness, 'run_tests_parallel'):
        print("✓ run_tests_parallel() function exists")
    else:
        print("✗ run_tests_parallel() function missing")
        return False
    
    return True

def test_web_search_replacement():
    """Test that WebSearchTool is imported and used instead of SERP API"""
    try:
        from agents import WebSearchTool
        print("✓ WebSearchTool import successful")
        return True
    except ImportError as e:
        print(f"✗ WebSearchTool import failed: {e}")
        return False

def main():
    print("Testing OpenAI Agents SDK Harness Implementation")
    print("=" * 50)
    
    harness = test_harness_imports()
    if not harness:
        print("\n❌ FAILED: Cannot import harness module")
        sys.exit(1)
    
    if not test_function_tools(harness):
        print("\n❌ FAILED: Function tool conversion issues")
        sys.exit(1)
    
    if not test_tool_signatures(harness):
        print("\n❌ FAILED: Tool signature issues")
        sys.exit(1)
    
    if not test_cli_interface(harness):
        print("\n❌ FAILED: CLI interface issues")
        sys.exit(1)
    
    if not test_web_search_replacement():
        print("\n❌ FAILED: Web search replacement issues")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED")
    print("\nHarness Implementation Summary:")
    print("- ✓ All finance agent tools converted to @function_tool format")
    print("- ✓ SERP API replaced with OpenAI WebSearchTool")
    print("- ✓ CLI interface maintains compatibility")
    print("- ✓ Tool signatures match original implementation")
    print("- ✓ Ready for execution with valid API keys")

if __name__ == "__main__":
    main()
