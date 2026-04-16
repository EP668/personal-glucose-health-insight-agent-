"""Core ReACT Agent Logic for the glucose-focused PHIA variant."""

from typing import Sequence, Optional
import pandas as pd
import numpy as np
import io
import contextlib
import os

from tavily import TavilyClient
from onetwo.agents import react
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import python_tool_use

from prompt_templates import (
    build_exemplars,
    AGENT_PREAMBLE,
    PHIA_REACT_PROMPT_TEXT,
)


def tavily_search_func(query: str, api_key: Optional[str] = None) -> str:
    """Performs web search using Tavily API."""
    try:
        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            return (
                "Error: Tavily API key not found. "
                "Please set the key when creating the agent or set the "
                "TAVILY_API_KEY environment variable."
            )

        client = TavilyClient(api_key=api_key)

        search_results = client.search(
            query=query,
            max_results=5,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced",
        )

        formatted_results = []

        if search_results.get("answer"):
            formatted_results.append(f"Summary: {search_results['answer']}\n")

        if search_results.get("results"):
            formatted_results.append("Search Results:")
            for i, result in enumerate(search_results["results"], 1):
                formatted_results.append(f"\n{i}. {result.get('title', 'No title')}")
                formatted_results.append(f"   URL: {result.get('url', 'No URL')}")
                formatted_results.append(
                    f"   Content: {result.get('content', 'No content available')[:200]}..."
                )
                if result.get("score") is not None:
                    try:
                        formatted_results.append(
                            f"   Relevance Score: {float(result['score']):.2f}"
                        )
                    except Exception:
                        pass

        return "\n".join(formatted_results) if formatted_results else "No search results found."

    except Exception as e:
        return f"Error performing search: {str(e)}"


def simple_python_executor(code: str, globals_dict: dict) -> str:
    """Executes python code and captures the output."""
    local_scope = {}
    string_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(string_buffer):
            code_lines = code.strip().split("\n")
            last_line = code_lines[-1].strip() if code_lines else ""

            is_expression = (
                last_line
                and not last_line.startswith(
                    (
                        "import ",
                        "from ",
                        "def ",
                        "class ",
                        "if ",
                        "for ",
                        "while ",
                        "with ",
                        "try:",
                        "except",
                        "finally",
                        "elif ",
                        "else:",
                    )
                )
                and "=" not in last_line.split("#")[0]
                and not last_line.startswith("#")
            )

            if is_expression and len(code_lines) > 1:
                exec("\n".join(code_lines[:-1]), globals_dict, local_scope)
                result = eval(last_line, {**globals_dict, **local_scope}, local_scope)

                printed_output = string_buffer.getvalue()
                output = printed_output if printed_output else str(result)
            else:
                exec(code, globals_dict, local_scope)
                printed_output = string_buffer.getvalue()

                if not printed_output and is_expression:
                    try:
                        result = eval(last_line, {**globals_dict, **local_scope}, local_scope)
                        output = str(result)
                    except Exception:
                        output = printed_output or "Code executed successfully."
                else:
                    output = printed_output or "Code executed successfully."

        return output

    except Exception as e:
        return f"Error executing code: {e}"


def get_react_agent(
    cgm_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    meal_features_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    meal_rule_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    example_files: Sequence[str],
    tavily_api_key: Optional[str] = None,
    use_mock_search: bool = False,
):
    """Initializes and returns a glucose-focused ReAct agent."""

    sandbox_globals = {
        "pd": pd,
        "np": np,
        "cgm_df": cgm_df,
        "meals_df": meals_df,
        "meal_features_df": meal_features_df,
        "daily_df": daily_df,
        "meal_rule_df": meal_rule_df,
        "risk_df": risk_df,
        "profile_df": profile_df,
        "profile": profile_df,  # compatibility alias
    }

    python_tool = llm_tool_use.Tool(
        name="tool_code",
        function=lambda code: simple_python_executor(code, globals_dict=sandbox_globals),
        description=(
            "Python interpreter. Executes code on pandas DataFrames "
            "(cgm_df, meals_df, meal_features_df, daily_df, meal_rule_df, risk_df, profile_df)."
        ),
    )

    if use_mock_search:
        search_tool = llm_tool_use.Tool(
            name="search",
            function=lambda query: "Web search is not available in mock mode.",
            description="Mock search engine.",
        )
    else:
        search_tool = llm_tool_use.Tool(
            name="search",
            function=lambda query: tavily_search_func(query, api_key=tavily_api_key),
            description="Web search engine for background glucose-related knowledge.",
        )

    finish_tool = llm_tool_use.Tool(
        name="finish",
        function=lambda x: x,
        description="Returns the final answer.",
    )

    agent_tools = [python_tool, search_tool, finish_tool]

    examples = build_exemplars(example_files)

    agent = react.ReActAgent(
        exemplars=examples,
        environment_config=python_tool_use.PythonToolUseEnvironmentConfig(
            tools=agent_tools,
        ),
        max_steps=10,
        stop_prefix="",
    )

    agent.prompt = react.ReActPromptJ2(
        text=AGENT_PREAMBLE + PHIA_REACT_PROMPT_TEXT
    )

    return agent


QUESTION_PREFIX = (
    "Use tools such as tool_code to execute Python code and search to find "
    "external relevant information as needed. "
    "Use the user's glucose-related tables whenever relevant. "
    "Prioritize meal_features_df, meal_rule_df, daily_df, and risk_df for most questions. "
    "Only use cgm_df for fine-grained raw time-series inspection. "
    "Use search only for background supplementation. "
    "Do not provide a formal medical diagnosis. "
    "Do not stop at generic summaries if actionable insights can be extracted from the data. "
    "Prioritize the most important 1 to 3 findings, explain why they matter, "
    "and suggest practical non-diagnostic next steps when appropriate. "
    "Take into account that questions may have typos or grammatical mistakes "
    "and try to reinterpret them before answering. "
    "Follow all instructions and the ReAct template carefully. "
    "Question: "
)