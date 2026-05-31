import asyncio
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv ()

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike
from github import Github, Auth, GithubException

"""
IMPROVEMENT: Make solution simpler by placing github user and github repo name 
in environment variables (as constant)
"""

# LLM
LLM_MODEL = "gpt-4o-mini"

# Free Tiny LLM to use free LLM client with OpenAI compatible syntax
tiny_api_key = os.getenv ("TINY_API_KEY")
if not tiny_api_key:
    raise RuntimeError ("TINY_API_KEY is not set.")
tiny_base_url: str = "https://litellm.aks-hs-prod.int.hyperskill.org/openai/"

llm = OpenAILike (
    model=LLM_MODEL,
    api_base=tiny_base_url,
    api_key=tiny_api_key,
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=True,
)

# My own public GitHub repository URL if needed.
# repo_url = "https://github.com/fvreys/recipes-api-main.git"
repo_url = os.getenv ("REPOSITORY")
pr_number_environment = os.getenv ("PR_NUMBER")


# TOOLS
def get_repository(file_path: str):
    """ With a given file path, fetch the repository of this file """
    github_token = os.getenv ("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError ("GITHUB_TOKEN is not set.")
    git = Github (auth=Auth.Token (github_token))

    if (not file_path or file_path.strip () in {".", "/"}) \
            or (file_path == "") or (file_path is None):
        final_repo_url = repo_url
    else:
        final_repo_url = file_path.strip ()

    repo_name = final_repo_url.split ('/')[-1].replace ('.git', '')
    username = final_repo_url.split ('/')[-2]
    full_repo_name = f"{username}/{repo_name}"
    repo = git.get_repo (full_repo_name)
    return repo


def get_pr_details(pr_number: int, file_path: str) -> dict[str, int | str | list[Any]]:
    """ Given a pull request number, return details about the pull request such as the author, title, body, commit SHAs, state, and more. """
    repo = get_repository (file_path)
    pr = repo.get_pull (pr_number)

    commit_sha: list[Any] = []
    commits = pr.get_commits ()
    for c in commits:
        commit_sha.append (c.sha)
    pr_details = {'number': pr_number,
                  'author': pr.user.login if pr.user else "",
                  'title': pr.title or "",
                  'body': pr.body or "",
                  'diff_url': pr.diff_url or "",
                  'state': pr.state or "",
                  'pr_commit_SHA': commit_sha
                  }
    return pr_details


def get_pr_commit_details(pr_number: int, file_path: str) -> list[dict[str, Any]]:
    """ Given a pull request number, return details about the commits in the pull request such as the commit SHAs, commit messages, and more. """
    " IMPROVEMENT: Given a commit SHA, return all changed files in the commit "
    " commit = repo.get_commit(head_sha) / for f in commit.files: "
    pr_details = get_pr_details (pr_number, file_path)
    # print (f'*** commit details {pr_details}')
    repo = get_repository (file_path)

    changed_files: list[dict[str, Any]] = []
    commit_sha_all = pr_details['pr_commit_SHA']
    for commit_sha in commit_sha_all:
        commit = repo.get_commit (commit_sha)
        # print (f'*** commit {commit}')

        for f in commit.files:
            changed_files.append ({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            })
    return changed_files


def get_file(file_path: str, file_to_fetch: str) -> str:
    """ With a given file path and file name, fetch the contents of a file from the repository """
    if not file_to_fetch or not file_to_fetch.strip ():
        raise ValueError ("file_to_fetch must be provided.")
    try:
        repo = get_repository (file_path)
        file_content = repo.get_contents (file_to_fetch)
        return file_content.decoded_content.decode ('utf-8')

    except GithubException as error:
        if error.status == 404:
            return f"Error: file '{file_to_fetch}' was not found in the repository."
        return f"Error while fetching '{file_to_fetch}' from GitHub: {error.data}"

    except Exception as error:
        return f"Error while fetching '{file_to_fetch}': {error}"


# STATE and MEMORY - Create a context to store the conversation history/session state
async def add_summary_to_state(ctxt: Context, new_summary: str) -> str:
    """ Add a summary to the state, with as input the summary. """
    async with ctxt.store.edit_state () as ctxt_state:
        summary = ctxt_state.get ("summary", [])
        summary.append (new_summary)
        ctxt_state["state"]["summary"] = summary
    return "Summary added to state."


async def add_comment_to_state(ctxt: Context, draft_comment: str) -> str:
    """Add the draft PR review comment to the state, with as input the draft PR review comment."""
    async with ctxt.store.edit_state () as ctxt_state:
        comment = ctxt_state.get ("comment", [])
        comment.append (draft_comment)
        ctxt_state["state"]["comment"] = comment
        ctxt_state["state"]["review_comment"] = draft_comment
    return "New PR comment added to state."


async def add_final_review_to_state(ctxt: Context, final_review: str) -> str:
    """ A tool to add the final review to the state """
    async with ctxt.store.edit_state () as ctxt_state:
        comment = ctxt_state.get ("review", [])
        comment.append (final_review)
        ctxt_state["state"]["review"] = comment
        ctxt_state["state"]["final_review"] = final_review
    return "Final review added to state."


def post_final_review(pr_number: int, comment: str, file_path: str = "") -> str:
    """ Post the final review comment to a GitHub pull request.
    Takes a PR number and a review comment, fetches the pull request,
    and posts the comment as a pull request review body.
    """
    if not comment or not comment.strip ():
        raise ValueError ("Comment must be provided.")

    repo = get_repository (file_path)
    pr = repo.get_pull (pr_number)
    pr.create_review (body=comment, event="COMMENT")
    return f"Final review added to PR {pr_number}"


tools = [
    FunctionTool.from_defaults (fn=get_pr_details),
    FunctionTool.from_defaults (fn=get_pr_commit_details),
    FunctionTool.from_defaults (fn=get_file),
    FunctionTool.from_defaults (fn=add_summary_to_state),
    FunctionTool.from_defaults (fn=add_comment_to_state),
    FunctionTool.from_defaults (fn=add_final_review_to_state),
    FunctionTool.from_defaults (fn=post_final_review),

]

# AGENTS
system_prompt_contextagent: str = """ 
You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the CommentorAgent. 
"""
system_prompt_commentoragent: str = """ 
You are the CommentorAgent that writes review comments for pull requests as a human reviewer would. \n 

Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n

 After creating the review text:
1. Save the review text to the shared workflow state.
2. Handoff back to ReviewAndPostingAgent, 
   who is responsible for posting the review to GitHub and producing the final response.
3. Do not produce the final answer yourself.

- If you need any additional pr details, you must hand off to the ContextAgent. \n
- You must hand off to the ReviewAndPostingAgent once you finished drafting a review. 
- You should directly address the author. So your comments should sound like: \n
"Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?" 
"""
system_prompt_reviewandpostingagent: str = """ 
You are the Review and Posting agent. 
You must use the CommentorAgent to create a review comment. 
If PR details are not available, you must hand off to the ContextAgent.

WORKFLOW:
1. If PR context is missing, hand off to ContextAgent.
2. If PR review text is missing, hand off to CommentorAgent.
3. If PR review text exists and does meet the criteria (it is no draft), call the GitHub review posting tool.
4. Only produce the final response after the review has been posted.

Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n

 If the draft review does not meet this criteria, 
 you must ask the CommentorAgent to rewrite and address these concerns. \n
 If the review meets the criteria, post the review comment to GitHub.  
 Do not skip the final posting step.
"""

context_agent = FunctionAgent (
    name="ContextAgent",
    description="Gathers all the needed context to generate a summary.",
    system_prompt=system_prompt_contextagent,
    llm=llm,
    tools=[get_pr_details, get_pr_commit_details, get_file, add_summary_to_state],
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent (
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull request review comment.",
    system_prompt=system_prompt_commentoragent,  # etc.
    llm=llm,
    tools=[add_comment_to_state],
    can_handoff_to=["ReviewAndPostingAgent", "ContextAgent"]
)

review_and_posting_agent = FunctionAgent (
    name="ReviewAndPostingAgent",
    description="Uses the final comment gathered by the commentor agent to create the final PR comment.",
    system_prompt=system_prompt_reviewandpostingagent,
    llm=llm,
    tools=[add_final_review_to_state, post_final_review],
    can_handoff_to=["ContextAgent", "CommentorAgent"],
)

# WORKFLOW (ORCHESTRATION of CONTEXT RETRIEVAL) - AgentWorkflow
workflow_agent = AgentWorkflow (
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "summary": "",
        "comment": "",
        "review_comment": "",
        "review": "",
        "final_review": "",
    },
)


async def main():
    query = f"Write a review for PR: {pr_number_environment}"

    ctx = Context (workflow_agent)
    handler = workflow_agent.run (query, ctx=ctx)

    current_agent = None
    async for event in handler.stream_events ():
        if hasattr (event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print (f"Current agent: {current_agent}")
        elif isinstance (event, AgentOutput):
            if event.response.content:
                print ("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print ("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance (event, ToolCallResult):
            print (f"Output from tool: {event.tool_output}")
        elif isinstance (event, ToolCall):
            print (f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    final_result = await handler
    print ("\n\nFinal result:", final_result)


if __name__ == "__main__":
    asyncio.run (main ())