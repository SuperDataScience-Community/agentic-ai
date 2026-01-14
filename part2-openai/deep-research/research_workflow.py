import os
from agents import Agent, Runner, WebSearchTool, InputGuardrail, GuardrailFunctionOutput
from agents.exceptions import InputGuardrailTripwireTriggered
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import logging
import pprint

load_dotenv()

class ResearchValidation(BaseModel):
    reason: str
    is_valid: bool

class Subtopics(BaseModel):
    subtopics: list[str]

class ResearchResults(BaseModel):
    findings: str

class OptimizationDecision(BaseModel):
    justification: str
    needs_more_research: bool

class SummaryReport(BaseModel):
    report: str

# Step 1: What agents do we need?

input_guardrail_agent = Agent(
    name="InputGuardrailAgent",
    instructions="You are an input validation agent. Ensure the user is not asking for disallowed content or anything inappropriate.",
    model="gpt-4o-mini",
    output_type=ResearchValidation
)

# Input guardrail function
async def input_guardrail(ctx, agent, input_data):
    result = await Runner.run(input_guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered= not final_output.is_valid
    )

topic_splitter_agent = Agent(
    name="TopicSplitterAgent",
    instructions="You are a topic splitting agent. Break down the research topic into manageable subtopics.",
    model="gpt-4o-mini",
    output_type=Subtopics
)

research_agent = Agent(
    name="ResearchAgent",
    instructions="You are a research agent. Use web search tools to gather information on the given subtopic.",
    model="gpt-4.1-mini",
    tools=[WebSearchTool()],
    output_type=ResearchResults
)

optimizer_agent = Agent(
    name="OptimizerAgent",
    instructions="You are an optimizer agent. Determine if the research is sufficient or additional research must be done.",
    model="gpt-4.1-mini",
    output_type=OptimizationDecision
)

synthesizer_agent = Agent(
    name="SynthesizerAgent",
    instructions="You are a synthesizer agent. Combine all optimized research findings into a coherent summary report.",
    model="gpt-4.1-mini",
    output_type=SummaryReport
)


# Step 3: What tools do we need?

# Step 4: What is the workflow?
    # a. How would the research be done?

async def research_workflow(user_query: str):

    # Step 1: Get subtopics
    try:
        subtopics = await Runner.run(topic_splitter_agent, user_query)
    except InputGuardrailTripwireTriggered as e:
        print(f"Input guardrail triggered: {e}")
        return

    subtopics_list = subtopics.final_output.subtopics

    # Step 2: Create a research task for each subtopic
    research_tasks = [
        Runner.run(research_agent, subtopic) for subtopic in subtopics_list
    ]

    # Step 3: Await all research tasks to complete
    research_results = await asyncio.gather(*research_tasks)

    # Step 4: Extract findings from each research result
    findings_list = [result.final_output.findings for result in research_results]

    # Step 5: Consolidate findings
    consolidated_findings = "\n".join(findings_list)

    return consolidated_findings



# b. How would the final check and summarization be done?

async def run_research_pipeline(user_query: str):

    try:
        research = await research_workflow(user_query)

        optimizer_decision = await Runner.run(optimizer_agent, research)

        if optimizer_decision.final_output.needs_more_research:
            print("Additional research is needed based on the optimizer's decision.")
            research = await research_workflow(user_query)
            optimizer_decision = await Runner.run(optimizer_agent, research)

        final_report = await Runner.run(synthesizer_agent, research)
        
        return final_report.final_output.report

    except Exception as e:
        print(f"An error occurred during the research pipeline: {e}")
        return f"An error occurred during research: {str(e)}"


if __name__ == "__main__":
    user_query = "The impact of renewable energy adoption on global economies."

    final_summary = asyncio.run(run_research_pipeline(user_query))

    print("Final Summary Report:")
    pprint.pprint(final_summary)