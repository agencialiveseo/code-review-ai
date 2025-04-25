import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from github import Github, GithubException, UnknownObjectException
from github.PullRequest import PullRequest
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from typing import Optional, List, Dict, Union, Any, Tuple
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- Configuration & Initialization ---

# Load environment variables
load_dotenv()

# Configure Flask app and logging
app = Flask(__name__)
# Use Flask's built-in logger
app.logger.setLevel(logging.INFO) 
handler = logging.StreamHandler() # Log to console
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.propagate = False # Prevent duplicate logging if root logger is configured

# GitHub and OpenAI API Keys
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variables
missing_vars = []
if not GITHUB_TOKEN:
    missing_vars.append("GITHUB_TOKEN")
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")

if missing_vars:
    error_msg = f"Error: Missing required environment variables: {', '.join(missing_vars)}. Please set them in your .env file."
    app.logger.error(error_msg)
    exit(1) # Exit if essential config is missing

# Configure LLM client (ensure you have langchain-openai installed)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
except Exception as e:
    app.logger.error(f"Error initializing OpenAI client: {e}")
    exit(1)

# Initialize GitHub API client (ensure PyGithub is installed)
try:
    github_client = Github(GITHUB_TOKEN)
    # Test connection
    github_client.get_user().login 
    app.logger.info("Successfully connected to GitHub API.")
except GithubException as e:
     app.logger.error(f"Error initializing GitHub client or authenticating: {e}")
     exit(1)
except Exception as e:
    app.logger.error(f"An unexpected error occurred during GitHub client initialization: {e}")
    exit(1)

# --- Tool Definitions ---
# Tools are now stateless functions decorated with @tool
# They accept necessary context (like repo/pr info) as arguments

@tool("Get Changed Files Tool")
def get_changed_files_tool(repo_full_name: str, pr_number: int) -> Dict[str, Any]:
    """
    Get relevant changed files and their patches for a specific pull request.
    Filters out binary files, lock files, and removed files.
    Args:
        repo_full_name (str): The full name of the repository (e.g., 'owner/repo').
        pr_number (int): The number of the pull request.
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'files': A list of dicts, each with 'filename' and 'status'.
            - 'patches': A dictionary mapping filenames to their patch content.
            - 'error': An error message string if an error occurred, otherwise None.
            - 'last_commit_sha': The SHA of the last commit in the PR.
    """
    app.logger.info(f"Tool 'get_changed_files_tool' called for {repo_full_name} PR #{pr_number}")
    try:
        repo = github_client.get_repo(repo_full_name)
        pull_request = repo.get_pull(pr_number)
        
        files = pull_request.get_files()
        changed_files_data = []
        file_patches = {}
        excluded_extensions = ('.lock', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.pdf', '.zip', '.gz')
        excluded_files = ('package-lock.json', 'yarn.lock', 'pnpm-lock.yaml')

        for file in files:
            if file.status == 'removed' or \
               file.filename.lower().endswith(excluded_extensions) or \
               os.path.basename(file.filename.lower()) in excluded_files or \
               not file.patch: # Skip files without patches (e.g., large files, submodule changes)
                app.logger.debug(f"Skipping file: {file.filename} (status: {file.status}, excluded: {file.filename.lower().endswith(excluded_extensions) or os.path.basename(file.filename.lower()) in excluded_files or not file.patch})")
                continue
                
            # Truncate large patches to avoid LLM context limits / high costs
            max_patch_length = 15000 # Increased limit slightly
            patch_content = file.patch
            if len(patch_content) > max_patch_length:
                 app.logger.warning(f"Patch for {file.filename} truncated to {max_patch_length} characters.")
                 patch_content = patch_content[:max_patch_length] + "\n[... patch truncated ...]"
                 
            file_patches[file.filename] = patch_content
            changed_files_data.append({
                "filename": file.filename,
                "status": file.status, # Primarily 'added' or 'modified' now
            })
            
        # Get last commit SHA safely
        last_commit_sha = None
        commits = pull_request.get_commits()
        if commits.totalCount > 0:
             # Get the last commit in the sequence returned by the API
             last_commit_sha = commits[commits.totalCount - 1].sha 
        
        if not last_commit_sha:
            app.logger.warning(f"Could not determine last commit SHA for PR #{pr_number}")
            # Decide if this is critical. Let's allow proceeding but commenting might fail.
            
        app.logger.info(f"Found {len(changed_files_data)} relevant changed files for PR #{pr_number}.")
        return {"files": changed_files_data, "patches": file_patches, "error": None, "last_commit_sha": last_commit_sha}
        
    except UnknownObjectException:
        error_msg = f"Error: Pull Request #{pr_number} not found in repository {repo_full_name}."
        app.logger.error(error_msg)
        return {"files": [], "patches": {}, "error": error_msg, "last_commit_sha": None}
    except GithubException as e:
        error_msg = f"GitHub API error getting changed files for PR #{pr_number}: {e.status} - {e.data.get('message', 'No details')}"
        app.logger.error(error_msg, exc_info=True)
        return {"files": [], "patches": {}, "error": error_msg, "last_commit_sha": None}
    except Exception as e:
        error_msg = f"Unexpected error getting changed files for PR #{pr_number}: {e}"
        app.logger.error(error_msg, exc_info=True)
        return {"files": [], "patches": {}, "error": error_msg, "last_commit_sha": None}


@tool("Find Diff Position Tool")
def find_diff_position_tool(patch_content: str, target_line_content: str) -> int:
    """
    Find the position (line number within the diff hunk) for a specific line of added code.
    Args:
        patch_content (str): The git diff patch content for a single file.
        target_line_content (str): The exact content of the line (must start with '+') to find.
    Returns:
        int: The 1-based position in the diff if found, otherwise -1.
    """
    app.logger.debug(f"Tool 'find_diff_position_tool' called. Target line: '{target_line_content}'")
    if not patch_content or not target_line_content or not target_line_content.startswith('+'):
        app.logger.warning(f"Invalid input to find_diff_position_tool: patch_empty={not patch_content}, target_empty={not target_line_content}, target_no_plus={not target_line_content.startswith('+')}")
        return -1 

    # Clean the target line slightly - remove leading '+' and strip surrounding whitespace
    # LLMs might add/remove trailing spaces sometimes. Be a bit lenient.
    clean_target_line = target_line_content[1:].strip() 
    
    lines = patch_content.split('\n')
    position_in_diff = 0 # Overall line number in the patch
    position_in_hunk = 0 # Line number within the current hunk for GitHub comments
    
    for line in lines:
        position_in_diff += 1
        
        if line.startswith('@@'):
            position_in_hunk = 0 # Reset hunk position counter
            continue # Skip hunk header line itself

        position_in_hunk += 1 # Increment position within the current hunk

        if line.startswith('+'):
            line_content = line[1:].strip() # Compare stripped content
            if clean_target_line == line_content:
                app.logger.info(f"Found match for target line at diff position {position_in_hunk}")
                return position_in_hunk 
        elif line.startswith('-'):
            continue # Ignore removed lines for calculating position
        elif line.startswith('\\'): # No newline at end of file marker
             continue
        # else: context line, count it in position_in_hunk
            
    app.logger.warning(f"Could not find exact match for line '{clean_target_line}' in the provided patch.")
    return -1 # Not found

@tool("Post GitHub Review Comment Tool")
def post_review_comment_tool(
    repo_full_name: str,
    pr_number: int,
    commit_sha: str,
    filename: str,
    comment_body: str,
    diff_position: Optional[int] = None
) -> str:
    """
    Posts a review comment on a GitHub pull request diff using the Review API.
    If diff_position is provided and valid (>0), it submits a review containing
    a single inline comment at that position. Otherwise, it posts a general PR comment.
    Args:
        repo_full_name (str): The repository name (e.g., 'owner/repo').
        pr_number (int): The pull request number.
        commit_sha (str): The SHA of the commit the comment refers to.
        filename (str): The path to the file being commented on.
        comment_body (str): The text of the comment.
        diff_position (Optional[int]): The position (line number) in the diff hunk (1-based) to comment on.
                                       If None or <= 0, posts as a general PR comment.
    Returns:
        str: A status message indicating success or failure.
    """
    app.logger.info(f"Tool 'post_review_comment_tool' called for {repo_full_name} PR #{pr_number}, Commit: {commit_sha[:7]}, File: {filename}, Position: {diff_position}")

    if not all([repo_full_name, pr_number, commit_sha, filename, comment_body]):
         return "Error: Missing required arguments (repo_full_name, pr_number, commit_sha, filename, comment_body)."

    try:
        repo = github_client.get_repo(repo_full_name)
        try:
             pull_request = repo.get_pull(pr_number)
        except UnknownObjectException:
             error_msg = f"Error: Pull Request #{pr_number} not found in repository {repo_full_name}."
             app.logger.error(error_msg)
             return error_msg
        except GithubException as e:
             error_msg = f"GitHub API error fetching PR #{pr_number}: {e.status} - {e.data.get('message', 'No details')}"
             app.logger.error(error_msg, exc_info=True)
             return error_msg

        try:
            position_int = int(diff_position) if diff_position is not None else 0
        except (ValueError, TypeError):
            app.logger.warning(f"Invalid diff_position '{diff_position}' received, defaulting to general comment.")
            position_int = 0

        if position_int > 0:
            # --- Use PullRequest.create_review for inline comments ---
            app.logger.info(f"Attempting to submit review with inline comment on: PR #{pr_number}, File: {filename}, Position: {position_int}")

            # Fetch the Commit object - needed for create_review
            try:
                commit_object = repo.get_commit(sha=commit_sha)
                app.logger.debug(f"Successfully fetched commit object for SHA {commit_sha}")
            except GithubException as commit_err:
                 app.logger.error(f"Failed to fetch commit object for SHA {commit_sha}: {commit_err}. Falling back to general comment.")
                 fallback_body = (f"**Review Feedback for `{filename}` (Commit Not Found):**\n\n"
                                  f"{comment_body}\n\n"
                                  f"_(Could not fetch commit {commit_sha} to submit review comment.)_")
                 try:
                      pull_request.create_issue_comment(fallback_body)
                      return f"Posted as general comment regarding {filename} (failed to fetch commit {commit_sha})."
                 except Exception as fallback_err:
                      app.logger.error(f"Failed to post fallback comment after failing to fetch commit: {fallback_err}")
                      return f"Error: Failed to fetch commit {commit_sha} and failed to post fallback comment."

            # Prepare the comment dictionary for the create_review call
            review_comments = [
                {
                    "path": filename,
                    "position": position_int,
                    "body": comment_body
                }
            ]

            try:
                # Submit the review with the single comment
                # Use event='COMMENT' to just submit comments without approving/requesting changes
                pull_request.create_review(
                    commit=commit_object,
                    body="",  # Optional: Add a main review body here if desired
                    event="COMMENT",
                    comments=review_comments
                )
                success_msg = f"Successfully submitted review with inline comment on {filename} at position {position_int}."
                app.logger.info(success_msg)
                return success_msg
            except GithubException as e:
                # Handle potential errors like outdated diff (422)
                is_outdated_diff_error = False
                if e.status == 422:
                    try:
                        error_msg_lower = str(e.data).lower()
                        if 'pull_request_review_thread' in error_msg_lower or 'position is invalid' in error_msg_lower or 'diff is outdated' in error_msg_lower:
                           is_outdated_diff_error = True
                    except Exception: pass

                if is_outdated_diff_error:
                    app.logger.warning(f"Failed to submit review comment (likely outdated diff/invalid position): {e.status} - {e.data}. Posting as general comment.")
                else:
                    app.logger.error(f"GitHub API error submitting review for {filename}: {e.status} - {e.data.get('message', 'No details')}", exc_info=True)

                # Fallback to general PR comment
                fallback_body = (f"**Review Feedback for `{filename}` (Review Submission Failed):**\n\n"
                                 f"{comment_body}\n\n"
                                 f"_(Attempted to submit review comment on position {diff_position} of commit {commit_sha[:7]}, but failed. Error: {e.status} - {e.data.get('message', 'See logs')})_")
                try:
                    pull_request.create_issue_comment(fallback_body)
                    return f"Posted as general PR comment regarding {filename} after review submission failed."
                except Exception as inner_e:
                    error_msg = f"Failed original review submission for {filename} (Error: {e}) AND fallback PR comment also failed (Error: {inner_e})"
                    app.logger.error(error_msg, exc_info=True)
                    return error_msg
            except Exception as e: # Catch unexpected errors during review creation
                 app.logger.error(f"Unexpected error submitting review for {filename}: {e}", exc_info=True)
                 fallback_body = (f"**Review Feedback for `{filename}` (Review Submission Failed - Unexpected Error):**\n\n"
                                  f"{comment_body}\n\n"
                                  f"_(Attempted to submit review comment on position {diff_position} of commit {commit_sha[:7]}, but an unexpected error occurred: {e})_")
                 try:
                     pull_request.create_issue_comment(fallback_body)
                     return f"Posted as general PR comment regarding {filename} after review submission failed unexpectedly."
                 except Exception as inner_e:
                     error_msg = f"Failed original review submission for {filename} (Error: {e}) AND fallback PR comment also failed unexpectedly (Error: {inner_e})"
                     app.logger.error(error_msg, exc_info=True)
                     return error_msg # Report double failure

        else:
            # Post a general PR comment (pull_request object already fetched)
            app.logger.info(f"Posting general PR comment for file {filename} (position was {diff_position}).")
            try:
                fallback_body = (f"**Review Feedback for `{filename}`:**\n\n"
                                 f"{comment_body}\n\n"
                                 f"_(Comment relates to the latest changes in the PR. Could not pinpoint specific line.)_")
                pull_request.create_issue_comment(fallback_body)
                return f"Successfully posted general comment regarding {filename}."
            except GithubException as e:
                 error_msg = f"GitHub API error posting general PR comment regarding {filename}: {e.status} - {e.data.get('message', 'No details')}"
                 app.logger.error(error_msg, exc_info=True)
                 return error_msg
            except Exception as e:
                 error_msg = f"Unexpected error posting general PR comment for {filename}: {e}"
                 app.logger.error(error_msg, exc_info=True)
                 return error_msg

    # Catch errors during initial repo lookup or unexpected errors in the tool's main body
    except GithubException as e:
        error_msg = f"GitHub API error in post_review_comment_tool setup for {filename}: {e.status} - {e.data.get('message', 'No details')}"
        app.logger.error(error_msg, exc_info=True)
        return error_msg
    except Exception as e:
        error_msg = f"Outer unexpected error in post_review_comment_tool for {filename}: {e}"
        app.logger.error(error_msg, exc_info=True)
        return error_msg

# --- CrewAI Setup ---

def setup_crewai_agents_and_tasks(repo_full_name: str, pr_number: int, commit_sha: str):
    """Creates the CrewAI Agents and Tasks for code review."""
    app.logger.info("Setting up CrewAI agents and tasks...")

    # Define Agents
    code_analyzer = Agent(
        role='Code Analyzer',
        goal="Analyze code patches (diffs) for potential issues like bugs, style inconsistencies, "
             "performance concerns, security vulnerabilities, and leftover debugging code (e.g., 'console.log', 'debugger'). "
             "Focus specifically on lines added (lines starting with '+').",
        backstory="You are an expert code reviewer with a keen eye for detail. You meticulously scan code changes "
                  "to ensure high quality, maintainability, and correctness, focusing only on the changes introduced.",
        verbose=True,
        llm=llm,
        allow_delegation=False # This agent shouldn't delegate
    )

    github_commenter = Agent(
        role='GitHub Comment Formatter and Poster',
        goal="Receive code analysis findings, determine the correct diff position for each finding, "
             "and post them as review comments on the specified GitHub Pull Request using the available tools.",
        backstory="You are a meticulous assistant responsible for translating code analysis feedback into actionable "
                  "GitHub comments. You use tools precisely to post comments in the right place on the Pull Request.",
        verbose=True,
        llm=llm,
        tools=[find_diff_position_tool, post_review_comment_tool], # Assign the tools!
        allow_delegation=False # This agent executes the posting
    )

    # Define Tasks
    # Task 1: Analyze a single file's patch
    analyze_file_patch_task = Task(
        description=f"""Analyze the provided code patch for the file '{{file_path}}'.
        The patch content represents the changes made to this file in Pull Request #{pr_number} of repository {repo_full_name}.
        Patch content:
        ```diff
        {{patch_content}}
        ```
        Your goal is to identify potential issues ONLY in the added lines (those starting with '+').
        Look for:
        - Bugs or logical errors
        - Poor coding practices or style issues
        - Performance optimizations opportunities
        - Security vulnerabilities
        - Leftover debug statements (console.log, print, debugger, etc.)

        For EACH issue found, provide the following information clearly:
        1.  `problematic_line_content`: The EXACT, complete line of code (including the leading '+') where the issue is found. Do NOT summarize or rephrase the line.
        2.  `comment_body`: A concise explanation of the issue and a concrete suggestion for improvement. Be specific and constructive.

        If no issues are found in the added lines, state that explicitly.
        """,
        expected_output=f"""A JSON list of dictionaries, where each dictionary represents a single issue found.
        Each dictionary MUST contain the keys 'problematic_line_content' and 'comment_body'.
        Example for one issue found:
        ```json
        [
          {{
            "problematic_line_content": "+ console.log('User data:', userData);",
            "comment_body": "Leftover 'console.log' statement. Please remove this debugging code before merging."
          }}
        ]
        ```
        If no issues are found, return an empty list: `[]`.
        """,
        agent=code_analyzer,
        # This task does not directly use external tools, it analyzes text.
    )

    # Task 2: Post comments based on analysis results for that file
    post_file_comments_task = Task(
        description=f"""Process the list of identified issues for the file '{{file_path}}' received from the Code Analyzer.
        The list contains dictionaries, each with 'problematic_line_content' and 'comment_body'.
        The full patch content for the file is also provided below.
        You must use the available tools ('find_diff_position_tool' and 'post_review_comment_tool') to post these comments to GitHub.

        **Context for Tools:**
        - Repository: {repo_full_name}
        - Pull Request Number: {pr_number}
        - Commit SHA: {commit_sha}
        - File Path: {{file_path}}

        **Patch Content for Position Finding:**
        ```diff
        {{patch_content}}
        ```

        **For each issue in the analysis results list:**
        1.  Extract the `problematic_line_content` and `comment_body`.
        2.  Use the `find_diff_position_tool` with the provided `patch_content` and the exact `problematic_line_content` to get the line position in the diff.
        3.  Use the `post_review_comment_tool`:
            - Provide `repo_full_name`: "{repo_full_name}"
            - Provide `pr_number`: {pr_number}
            - Provide `commit_sha`: "{commit_sha}"
            - Provide `filename`: "{{file_path}}"
            - Provide `comment_body`: The extracted `comment_body`.
            - Provide `diff_position`: The position returned by `find_diff_position_tool`. If the tool returns -1 or an error occurs finding the position, pass `diff_position=None` or `diff_position=-1` to `post_review_comment_tool` so it posts a general comment for that file.

        Execute the posting for ALL identified issues.
        """,
        expected_output=f"""A summary report detailing the outcome of attempting to post each comment for file '{{file_path}}'.
        For each comment, indicate whether it was posted successfully (in-line or general) or if an error occurred.
        Example:
        - Comment 1 ('Leftover console.log...'): Successfully posted in-line comment on {{file_path}} at position 15.
        - Comment 2 ('Potential null pointer...'): Posted as general comment regarding {{file_path}} (position finding failed).
        - Comment 3 ('Inefficient loop...'): Error posting comment: GitHub API error 404 - Not Found.
        """,
        agent=github_commenter,
        context=[analyze_file_patch_task], # Depends on the output of the analysis task
        # Tools are assigned to the agent, this task will utilize them.
    )

    # Create and return the Crew
    return Crew(
        agents=[code_analyzer, github_commenter],
        tasks=[analyze_file_patch_task, post_file_comments_task],
        process=Process.sequential, # Run analysis then posting for each file
        verbose=True # Enable verbose mode for detailed logs
    )

# --- Flask Webhook Handler ---

@app.route("/webhook", methods=["POST"])
def github_webhook():
    """Handles incoming GitHub webhook events for PRs."""
    app.logger.info("Webhook received.")
    signature = request.headers.get('X-Hub-Signature')
    # TODO: Implement webhook signature verification for security
    # webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    # if not verify_signature(request.data, signature, webhook_secret):
    #     app.logger.warning("Invalid webhook signature.")
    #     return jsonify({"error": "Invalid signature"}), 403

    data = request.json
    event = request.headers.get('X-GitHub-Event')
    action = data.get("action")

    app.logger.debug(f"Event: {event}, Action: {action}")

    # Process only relevant PR events
    if event == "pull_request" and action in ["opened", "synchronize", "reopened"]:
        if "pull_request" not in data:
            app.logger.warning("Pull request data missing in webhook payload.")
            return jsonify({"message": "Ignoring event: Missing pull request data"}), 200

        pr_data = data["pull_request"]
        pr_number = pr_data["number"]
        repo_data = data["repository"]
        repo_full_name = repo_data["full_name"]

        # Avoid processing drafts unless explicitly desired
        if pr_data.get("draft", False):
            app.logger.info(f"Ignoring event for draft PR #{pr_number} in {repo_full_name}.")
            return jsonify({"message": "Ignoring draft pull request"}), 200

        app.logger.info(f"Processing {action} event for PR #{pr_number} in {repo_full_name}")

        try:
            # 1. Get changed files and context using the tool
            #    (We call it directly here to get data needed for looping and setup)
            # Access the original function via the .func attribute
            file_context = get_changed_files_tool.func(repo_full_name=repo_full_name, pr_number=pr_number) 

            if file_context["error"]:
                app.logger.error(f"Failed to get changed files: {file_context['error']}")
                # Optionally post a comment about the failure
                try:
                    repo = github_client.get_repo(repo_full_name)
                    pr = repo.get_pull(pr_number)
                    pr.create_issue_comment(f"⚠️ AI Review Error: Could not retrieve file changes for analysis. Error: {file_context['error']}")
                except Exception as comment_err:
                    app.logger.error(f"Failed to post error comment: {comment_err}")
                return jsonify({"error": file_context["error"]}), 500

            changed_files = file_context["files"]
            patches_dict = file_context["patches"]
            last_commit_sha = file_context["last_commit_sha"]

            if not changed_files:
                app.logger.info("No relevant file changes detected in the PR.")
                # Optionally post a comment indicating no files were reviewed
                try:
                    repo = github_client.get_repo(repo_full_name)
                    pr = repo.get_pull(pr_number)
                    pr.create_issue_comment("ℹ️ AI Review: No relevant code files found to review in this update.")
                except Exception as comment_err:
                    app.logger.error(f"Failed to post 'no files' comment: {comment_err}")
                return jsonify({"message": "No relevant files to review"}), 200
                
            if not last_commit_sha:
                 app.logger.error("Cannot proceed with commenting without the last commit SHA.")
                 # Post general comment about failure
                 try:
                    repo = github_client.get_repo(repo_full_name)
                    pr = repo.get_pull(pr_number)
                    pr.create_issue_comment("⚠️ AI Review Error: Could not determine the latest commit SHA. Unable to post comments accurately.")
                 except Exception as comment_err:
                    app.logger.error(f"Failed to post commit SHA error comment: {comment_err}")
                 return jsonify({"error": "Missing commit SHA"}), 500


            # 2. Setup CrewAI (now depends on commit_sha)
            review_crew = setup_crewai_agents_and_tasks(repo_full_name, pr_number, last_commit_sha)

            # 3. Iterate and Run Crew for each file
            app.logger.info(f"Starting analysis of {len(changed_files)} files...")
            all_results = []
            files_analyzed_count = 0

            for file_info in changed_files:
                filename = file_info['filename']
                patch_content = patches_dict.get(filename)

                if not patch_content:
                    app.logger.warning(f"Patch content missing for {filename} in collected data. Skipping.")
                    continue

                app.logger.info(f"\n--- Analyzing file: {filename} ---")

                # Prepare inputs for the Crew kickoff for this specific file
                crew_inputs = {
                    'file_path': filename,
                    'patch_content': patch_content
                    # Context needed by the second task's tools is embedded in the task description
                    # or handled by tools themselves (like fetching PR object).
                }

                try:
                    # Kick off the crew for this file
                    result = review_crew.kickoff(inputs=crew_inputs)
                    app.logger.info(f"--- Finished analyzing file: {filename} ---")
                    app.logger.info(f"Result for {filename}: {result}") # Result is the output of the LAST task
                    all_results.append({filename: result})
                    files_analyzed_count += 1
                except Exception as file_error:
                    app.logger.error(f"Error running CrewAI for file {filename}: {file_error}", exc_info=True)
                    # Attempt to post a general comment about the failure for this file
                    try:
                        error_comment_body = (
                            f"**⚠️ AI Review Error for `{filename}`:**\n\n"
                            f"The AI review process encountered an unexpected error while analyzing this file.\n"
                            f"Error details: `{str(file_error)}`\n\n"
                            f"Please review this file manually."
                        )
                        post_review_comment_tool.func( # Use the tool directly for error reporting
                             repo_full_name=repo_full_name,
                             pr_number=pr_number,
                             commit_sha=last_commit_sha, # Use the SHA we have
                             filename=filename,
                             comment_body=error_comment_body,
                             diff_position=None # Post as general comment
                        )
                    except Exception as comment_error:
                        app.logger.error(f"Failed to post error comment for {filename}: {comment_error}", exc_info=True)


            # 4. Optional: Post a summary comment after all files are processed
            summary_message = f"✅ AI Code Review Completed.\n\nAnalyzed {files_analyzed_count} file(s)."
            # You could potentially aggregate results from `all_results` for a more detailed summary
            # For now, just a completion notice.
            try:
                repo = github_client.get_repo(repo_full_name)
                pr = repo.get_pull(pr_number)
                pr.create_issue_comment(summary_message)
                app.logger.info("Posted overall completion summary comment.")
            except Exception as summary_comment_error:
                app.logger.error(f"Failed to post final summary comment: {summary_comment_error}", exc_info=True)


            return jsonify({"message": "Review process completed", "files_analyzed": files_analyzed_count}), 200

        except GithubException as e:
            error_msg = f"A GitHub API error occurred during processing: {e.status} - {e.data}"
            app.logger.error(error_msg, exc_info=True)
            return jsonify({"error": error_msg}), 500
        except Exception as e:
            error_msg = f"An unexpected error occurred during the review process: {e}"
            app.logger.error(error_msg, exc_info=True)
            # Attempt to post a general error comment if possible
            try:
                 repo = github_client.get_repo(repo_full_name) # Might fail if repo_full_name wasn't set yet
                 pr = repo.get_pull(pr_number) # Might fail if pr_number wasn't set yet
                 pr.create_issue_comment(f"⚠️ AI Review System Error: An unexpected critical error occurred: {e}. Review may be incomplete.")
            except Exception:
                 pass # Ignore error during final error reporting
            return jsonify({"error": error_msg}), 500

    else:
        app.logger.debug(f"Ignoring event type '{event}' with action '{action}'")
        return jsonify({"message": "Ignoring event: Not a relevant PR action"}), 200

# --- Main Execution ---

if __name__ == "__main__":
    # Make sure to set HOST and PORT via environment variables or config if needed
    port = int(os.getenv("PORT", 5000))
    app.logger.info(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port) # Use 0.0.0.0 to be accessible externally