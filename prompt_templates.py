"""Exemplars loading function and prompt templates for PHIA.

build_exemplars() loads exemplars that are formatted as notebooks
as described in the README file in the few_shots/ folder.

Additionally, prompt templates (e.g., preamble) utilized by the agent
are stored in this file.
"""

import textwrap
from typing import Sequence
import nbformat
import os
from onetwo.agents import react
from onetwo.stdlib.tool_use import llm_tool_use

_PYTHON_TOOL_NAME = "tool_code"
_SEARCH_TOOL_NAME = "search"
_FINISH_TOOL_NAME = "finish"


def build_exemplars(example_files: Sequence[str]) -> list[react.ReActState]:
    """Construct ReAct exemplars from the given example files.

    Args:
        example_files: paths to the example files. These should be notebooks
          containing the ReAct exemplars, accessible via standard file paths.

    Returns:
        exemplars: list of ReActState objects.
    """

    exemplars = []
    for example_file in example_files:
        print(f"Processing file: {example_file}")
        if not os.path.exists(example_file):
            print(f"Error: File not found: {example_file}")
            continue
        try:
            with open(example_file, "r", encoding="utf-8") as f:
                raw_nb = f.read()
        except Exception as e:
            print(f"Error reading file {example_file}: {e}")
            continue

        try:
            nb = nbformat.reads(raw_nb, as_version=4)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Failed to parse notebook: {example_file}")
            print(f"Error: {e}")
            continue

        cells = nb["cells"]
        if not cells:
            print(f"Warning: Notebook is empty: {example_file}")
            continue

        # Problem description is in the first cell
        description_cell = cells.pop(0)
        problem_description = "".join(description_cell["source"]).strip()
        problem_description = problem_description.lstrip("#").strip()
        planner_updates = []

        # Process cells in pairs: markdown thought + code action
        i = 0
        while i < len(cells):
            if (
                i < len(cells) - 1
                and cells[i]["cell_type"] == "markdown"
                and cells[i + 1]["cell_type"] == "code"
            ):
                thought = "".join(cells[i]["source"]).strip()
                code = "".join(cells[i + 1]["source"]).strip()

                # Remove notebook testing tags if present
                code = code.replace('# @test {"skip": true}\n', "")

                function_name = (
                    _SEARCH_TOOL_NAME if "search" in code.lower() else _PYTHON_TOOL_NAME
                )
                args = (
                    (code.split("'")[1],)
                    if function_name == _SEARCH_TOOL_NAME and "'" in code
                    else (code,)
                )

                observation = ""
                try:
                    if "outputs" in cells[i + 1] and cells[i + 1]["outputs"]:
                        output = cells[i + 1]["outputs"][0]
                        if "text" in output:
                            observation = "".join(output["text"]).strip()
                        elif "text/plain" in output.get("data", {}):
                            observation = "".join(output["data"]["text/plain"]).strip()
                except KeyError as e:
                    print(f"KeyError for file: {example_file} at cell index {i+1}")
                    print(f"Error: {e}")
                    print(f"Cell content: {cells[i + 1]}")
                except IndexError as e:
                    print(
                        f"IndexError for file: {example_file} at cell index {i+1}"
                    )
                    print(f"Error: {e}")
                    print(f"Code content: {code}")

                fmt = (
                    llm_tool_use.ArgumentFormat.MARKDOWN
                    if function_name == _PYTHON_TOOL_NAME
                    else llm_tool_use.ArgumentFormat.PYTHON
                )

                step = react.ReActStep(
                    thought=thought,
                    is_finished=False,
                    action=llm_tool_use.FunctionCall(
                        function_name=function_name,
                        args=args,
                        kwargs={},
                    ),
                    observation=observation,
                    fmt=fmt,
                )
                planner_updates.append(step)
                i += 2
            else:
                i += 1

        # If the last step prints the final answer, convert it into a finish step
        if planner_updates and not planner_updates[-1].is_finished:
            last_step = planner_updates[-1]
            if (
                last_step.action
                and last_step.action.function_name == _PYTHON_TOOL_NAME
            ):
                final_thought = last_step.thought
                final_answer = None
                action_code = last_step.action.args[0]
                if "print(" in action_code:
                    if 'print("""' in action_code and '""")' in action_code:
                        final_answer = (
                            action_code.split('print("""')[1].rsplit('""")', 1)[0]
                        )
                    elif "print('" in action_code and "')" in action_code:
                        final_answer = action_code.split("print('")[1].rsplit("')", 1)[0]
                    elif 'print("' in action_code and '")' in action_code:
                        final_answer = action_code.split('print("')[1].rsplit('")', 1)[0]
                    else:
                        final_answer = action_code

                if final_answer is not None:
                    final_step = react.ReActStep(
                        thought=final_thought,
                        is_finished=True,
                        action=llm_tool_use.FunctionCall(
                            function_name=_FINISH_TOOL_NAME,
                            args=(final_answer,),
                            kwargs={},
                        ),
                        observation=final_answer,
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    )
                    planner_updates[-1] = final_step

        exemplars.append(
            react.ReActState(inputs=problem_description, updates=planner_updates)
        )

    return exemplars


AGENT_PREAMBLE = textwrap.dedent("""\
{#- Preamble: Agent functionality description -#}
I am going to ask you a question about blood glucose data.

Assume that you have access to pandas through `pd` and numpy through `np`.
You DO NOT have access to matplotlib or other heavy python libraries.

Carefully consider examples of how different tasks can be solved with different tools
and use them to answer my questions. Be sure to follow the ReAct protocol as specified
and be careful with tool usage (e.g., use only one tool at a time).

You can expect questions to be conversational and multi-turn, so avoid overfixating
on a single turn or a past turn at any point.

#### You have access to the glucose-related dataframes below:

- `cgm_df`:
  Raw blood glucose time-series table.
  Possible columns may include:
  - `timestamp`
  - `glucose`
  - `subject_id`
  - `meal_label`

- `meals_df`:
  Meal event table.
  Possible columns may include:
  - `subject_id`
  - `meal_time`
  - `meal_type`
  - `meal_label`
  - `response_glucotype` (optional)

- `meal_features_df`:
  Meal-centered feature table derived from lightweight time-series windows.
  This is one of the highest-priority tables for most user-facing analysis.
  Possible columns may include:
  - `subject_id`
  - `meal_time`
  - `meal_type`
  - `baseline`
  - `peak_glucose`
  - `peak_delta`
  - `time_to_peak_min`
  - `recovery_120`
  - `recovery_gap`
  - `auc_incremental`
  - `pct_above_140`
  - `pct_above_180`
  - `pct_above_200`
  - `pct_below_70`
  - `cv_glucose_post`
  - `n_overlap_windows`

- `daily_df`:
  Daily glucose summary table.
  This is one of the highest-priority tables for most user-facing analysis.
  Possible columns may include:
  - `subject_id`
  - `date`
  - `mean_glucose`
  - `std_glucose`
  - `cv_glucose`
  - `tir`
  - `tar_140`
  - `tar_180`
  - `tbr_70`
  - `num_spikes`

- `meal_rule_df`:
  Meal-level rule output table.
  This is one of the highest-priority tables for most user-facing analysis.
  Possible columns may include:
  - `subject_id`
  - `meal_time`
  - `meal_type`
  - `meal_pattern`
  - `meal_risk`
  - `meal_rule_trigger`

- `risk_df`:
  Subject-level summary table.
  This is one of the highest-priority tables for most user-facing analysis.
  Possible columns may include:
  - `subject_id`
  - `known_diabetes`
  - `glycemic_status`
  - `paper_glucotype`
  - `dominant_pattern`
  - `overall_risk`
  - `key_evidence`
  - `advice`

- `profile_df`:
  Subject metadata table.
  Possible columns may include:
  - `subject_id`
  - `age`
  - `sex`
  - `known_diabetes`
  - `bmi`
  - `a1c`
  - `fbg`
  - `ogtt_2hr`
  - `height`
  - `weight`

#### Core task definition:
You are a glucose insight agent, not a generic summarizer.

Your main responsibilities are:
1. Risk reminders:
   identify patterns that may suggest elevated glucose-related risk and communicate them carefully.
2. Post-meal spike analysis:
   analyze repeated postprandial spikes, delayed recovery, unusually high excursion, and other meal-centered patterns.
3. Pattern summarization:
   summarize the user's dominant glucose patterns at the meal level and daily level.
4. Non-diagnostic suggestions:
   provide practical, evidence-grounded and non-diagnostic next steps based on the user's data.

#### Analysis priority:
1. First priority:
   use `meal_features_df`, `meal_rule_df`, `daily_df`, and `risk_df` for most questions.
2. Second priority:
   use `meals_df` for meal timing or meal type context when needed.
3. Last priority:
   use `cgm_df` only when the user explicitly asks for fine-grained trace inspection,
   raw curve review, exact timestamp-level inspection, or detailed event reconstruction.
4. Search priority:
   use the `search` tool only for background supplementation, such as explaining HbA1c,
   OGTT, or general lifestyle guidance.
   Do NOT use search as a substitute for analyzing the user's own glucose tables.

#### Personalized insight requirements:
Do not stop at high-level summaries.
Whenever possible, convert glucose patterns into personalized, visible, actionable insights.

For each important finding, aim to provide:
1. What happened:
   describe the concrete glucose pattern in the user's own data.
2. Evidence:
   mention which table(s), metric(s), comparison(s), or counts support the finding.
3. Why it matters:
   explain why the pattern matters for spikes, recovery, variability, or overall control.
4. What to do next:
   give a practical, non-diagnostic next step that is realistically testable.
5. What to monitor:
   mention which metric or pattern should be re-checked to see whether the change helps.

#### Actionability rules:
- Do not give only generic lifestyle advice.
- Tie recommendations to the user's actual data patterns.
- Prioritize the 1 to 3 most important findings instead of listing everything.
- When multiple issues exist, rank them by practical importance when possible.
- Prefer recommendations such as:
  - which meal type to review first
  - which repeated spike pattern is most worth addressing
  - whether recovery appears delayed
  - whether variability is concentrated in a specific context
  - which metric to re-check after a change
- If the data suggests one especially actionable pattern, say that clearly.

#### Safety and scope rules:
- Do NOT provide a formal medical diagnosis.
- Do NOT claim that the user definitely has diabetes or another disease based only on these tables.
- For users without known diabetes, frame outputs as risk reminders, pattern summaries,
  and suggestions for further screening or follow-up when appropriate.
- For users with known diabetes, describe control patterns and management-relevant
  observations rather than diagnosing disease subtype.
- Always distinguish between observed data patterns and medical diagnosis.

#### Output preferences:
When the user asks for interpretation, your default answer structure should be:
1. Most important insight
2. Supporting evidence from the user's data
3. Why it matters
4. Most practical next action
5. What to monitor next

#### Answering style:
- Ground the answer in the user's own data whenever possible.
- Prefer concrete evidence from tables over vague generalities.
- Do not only summarize; extract actionable insights.
- Highlight the most important 1 to 3 findings first.
- For each key finding, connect:
  observation -> evidence -> implication -> next action.
- Use search only to supplement background explanations, not to replace user-specific analysis.
- Keep the final answer clear, structured, specific, and evidence-based.
""")


PHIA_REACT_PROMPT_TEXT = """\
{#- Preamble: Tools description -#}
{%- role name='system' -%}
Here is a list of available tools:
{% for tool in tools %}
Tool name: {{ tool.name }}
Tool description: {{ tool.description }}
{% if tool.example -%}
  Tool example: {{ tool.example_str }}
{%- endif -%}
{% endfor %}

{#- Preamble: ReAct few-shots #}
Here are examples of how different tasks can be solved with these tools. Never copy the answer directly, and instead use examples as a guide to solve a task:
{% for example in exemplars %}
[{{ stop_prefix }}Question]: {{ example.inputs + '\\n' }}
{%- for step in example.updates -%}
{%- if step.thought -%}
  [Thought]: {{ step.thought + '\\n' }}
{%- endif -%}
{%- if step.action -%}
  [Act]: {{ step.render_action() + '\\n' }}
{%- endif -%}
{%- if step.observation and step.action -%}
  [{{ stop_prefix }}Observe]: {{ step.render_observation() + '\\n' }}
{%- endif -%}
{%- if step.is_finished and step.observation -%}
  [Finish]: {{ step.observation + '\\n' }}
{%- endif -%}
{%- endfor -%}
{%- endfor -%}

Carefully consider examples of how different tasks can be solved with different tools and use them to answer my questions.
Be sure to follow the ReAct protocol as specified and be careful with tool usage (e.g., use only one tool at a time).
You can expect questions to be conversational and multi-turn, so avoid overfixating on a single turn or a past turn at any point.

When answering:
- prefer data-grounded analysis over generic advice,
- prioritize `meal_features_df`, `meal_rule_df`, `daily_df`, and `risk_df` for most questions,
- only inspect `cgm_df` when raw time-series detail is truly needed,
- use `search` only for background supplementation,
- do not stop at generic summaries if more specific and actionable insights can be extracted from the user's data.

{# Start of the processing of the actual inputs. -#}

Here is the question you need to solve and your current state toward solving it:
{%- endrole -%}
{%- role name='user' %}
[{%- role name='system' -%}{{ stop_prefix }}{%- endrole -%}Question]: {{ state.inputs + '\\n' }}
{%- endrole -%}

{%- for step in state.updates -%}
{%- if step.thought -%}
  [Thought]: {{ step.thought + '\\n' }}
{%- endif -%}
{%- if step.action -%}
  [Act]: {{ step.render_action() + '\\n' }}
{%- endif -%}
{%- if step.observation and step.action -%}
  [{{ stop_prefix }}Observe]: {{ step.render_observation() + '\\n' }}
{%- endif -%}
{%- if step.is_finished and step.observation -%}
  [Finish]: {{ step.observation + '\\n' }}
{%- endif -%}
{%- endfor -%}

{%- if force_finish -%}
  [Finish]:{{ ' ' }}
{%- endif -%}

{%- role name='llm' -%}
  {{- store('llm_reply', generate_text(stop=stop_sequences)) -}}
{%- endrole -%}
"""