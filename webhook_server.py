# --- Imports ---
import os
import re
import json
import logging
import hashlib # For webhook signature
import hmac    # For webhook signature
from flask import Flask, request, jsonify, abort # Added abort
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
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not app.logger.handlers: # Avoid adding duplicate handlers on reloads
    app.logger.addHandler(handler)
app.logger.propagate = False

# --- Environment Variable Loading and Validation ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET") # <<< ADDED
DEFAULT_OPENAI_MODEL = "gpt-4-turbo" # <<< Use a known valid model as default
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", DEFAULT_OPENAI_MODEL) # <<< ADDED

missing_vars = []
if not GITHUB_TOKEN:
    missing_vars.append("GITHUB_TOKEN")
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not GITHUB_WEBHOOK_SECRET: # <<< ADDED Check for webhook secret
    missing_vars.append("GITHUB_WEBHOOK_SECRET")

if missing_vars:
    error_msg = f"Error: Missing required environment variables: {', '.join(missing_vars)}. Please set them in your .env file or environment."
    # Use print for startup errors as logger might not be fully ready / visible
    print(error_msg)
    exit(1) # Exit if essential config is missing

# --- Initialize LLM Client ---
try:
    app.logger.info(f"Initializing OpenAI client with model: {OPENAI_MODEL_NAME}")
    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.2, api_key=OPENAI_API_KEY)
    # Simple test call (optional, but good for validation)
    # llm.invoke("Test")
    app.logger.info("Successfully initialized OpenAI client.")
except Exception as e:
    print(f"CRITICAL ERROR: Error initializing OpenAI client: {e}") # Use print for critical startup error
    exit(1)

# --- Initialize GitHub API Client ---
try:
    app.logger.info("Initializing GitHub client...")
    github_client = Github(GITHUB_TOKEN)
    # Test connection by getting authenticated user
    user = github_client.get_user()
    app.logger.info(f"Successfully connected to GitHub API as user: {user.login}")
except GithubException as e:
     print(f"CRITICAL ERROR: Error initializing GitHub client or authenticating: {e.status} - {e.data.get('message', 'No details')}") # Use print
     exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: An unexpected error occurred during GitHub client initialization: {e}") # Use print
    exit(1)

# --- Constants ---
IMPROVEMENT_OPPORTUNITY_SCORE = 1
ALERT_SCORE = 3
PROBLEM_SCORE = 5
MAX_PATCH_LENGTH = 15000 # Consider making this an ENV VAR: MAX_PATCH_LENGTH
EXCLUDED_EXTENSIONS = ('.lock', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.pdf', '.zip', '.gz') # Consider making this an ENV VAR: EXCLUDED_EXTENSIONS (comma separated)
EXCLUDED_FILES = ('package-lock.json', 'yarn.lock', 'pnpm-lock.yaml') # Consider making this an ENV VAR: EXCLUDED_FILES (comma separated)

# --- Webhook Signature Verification ---
def verify_signature(payload_body, secret_token, signature_header):
    """Verify that the payload was sent from GitHub by validating SHA256 signature."""
    if not signature_header:
        app.logger.warning("Webhook signature header missing.")
        return False
    # Prefer sha256, GitHub sends it as X-Hub-Signature-256
    hash_object = hmac.new(secret_token.encode('utf-8'), msg=payload_body, digestmod=hashlib.sha256)
    expected_signature = "sha256=" + hash_object.hexdigest()
    if not hmac.compare_digest(expected_signature, signature_header):
        app.logger.warning(f"Webhook signature mismatch. Expected={expected_signature}, Got={signature_header}")
        return False
    app.logger.debug("Webhook signature verified successfully.")
    return True

# --- Tool Definitions ---

@tool("Get Changed Files Tool")
def get_changed_files_tool(repo_full_name: str, pr_number: int) -> Dict[str, Any]:
    """
    Get relevant changed files, their patches, and the PR's head commit SHA.
    Filters out binary files, lock files, and removed files. Truncates large patches.
    Args:
        repo_full_name (str): The full name of the repository (e.g., 'owner/repo').
        pr_number (int): The number of the pull request.
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'files': A list of dicts, each with 'filename' and 'status'.
            - 'patches': A dictionary mapping filenames to their patch content.
            - 'error': An error message string if an error occurred, otherwise None.
            - 'head_commit_sha': The SHA of the head commit of the PR's source branch.
    """
    app.logger.info(f"Tool 'get_changed_files_tool' called for {repo_full_name} PR #{pr_number}")
    try:
        repo = github_client.get_repo(repo_full_name)
        pull_request = repo.get_pull(pr_number)

        # --- Use head.sha for the most recent commit SHA ---
        head_commit_sha = pull_request.head.sha
        if not head_commit_sha:
             app.logger.error(f"Could not determine head commit SHA for PR #{pr_number}")
             # This is critical for posting comments accurately
             return {"files": [], "patches": {}, "error": "Could not determine head commit SHA for the PR.", "head_commit_sha": None}
        app.logger.info(f"Found head commit SHA for PR #{pr_number}: {head_commit_sha}")

        files = pull_request.get_files()
        changed_files_data = []
        file_patches = {}

        for file in files:
            # Skip based on status, extension, name, or lack of patch
            is_excluded_ext = file.filename.lower().endswith(EXCLUDED_EXTENSIONS)
            is_excluded_name = os.path.basename(file.filename.lower()) in EXCLUDED_FILES
            is_removed = file.status == 'removed'
            no_patch = not file.patch

            if is_removed or is_excluded_ext or is_excluded_name or no_patch:
                app.logger.debug(f"Skipping file: {file.filename} (Status: {file.status}, ExclExt: {is_excluded_ext}, ExclName: {is_excluded_name}, NoPatch: {no_patch})")
                continue

            # Truncate large patches
            patch_content = file.patch
            if len(patch_content) > MAX_PATCH_LENGTH:
                 app.logger.warning(f"Patch for {file.filename} truncated to {MAX_PATCH_LENGTH} characters.")
                 patch_content = patch_content[:MAX_PATCH_LENGTH] + "\n[... patch truncated ...]"

            file_patches[file.filename] = patch_content
            changed_files_data.append({
                "filename": file.filename,
                "status": file.status, # Primarily 'added' or 'modified' now
            })

        app.logger.info(f"Found {len(changed_files_data)} relevant changed files for PR #{pr_number}.")
        return {"files": changed_files_data, "patches": file_patches, "error": None, "head_commit_sha": head_commit_sha}

    except UnknownObjectException:
        error_msg = f"Error: Pull Request #{pr_number} not found in repository {repo_full_name}."
        app.logger.error(error_msg)
        return {"files": [], "patches": {}, "error": error_msg, "head_commit_sha": None}
    except GithubException as e:
        error_msg = f"GitHub API error getting changed files for PR #{pr_number}: {e.status} - {e.data.get('message', 'No details')}"
        app.logger.error(error_msg, exc_info=True)
        return {"files": [], "patches": {}, "error": error_msg, "head_commit_sha": None}
    except Exception as e:
        error_msg = f"Unexpected error getting changed files for PR #{pr_number}: {e}"
        app.logger.error(error_msg, exc_info=True)
        return {"files": [], "patches": {}, "error": error_msg, "head_commit_sha": None}

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

    clean_target_line = target_line_content[1:].strip()

    lines = patch_content.split('\n')
    position_in_diff = 0
    position_in_hunk = 0

    for line in lines:
        position_in_diff += 1

        if line.startswith('@@'):
            position_in_hunk = 0
            continue

        # Only count lines within the hunk for position
        # Lines starting with ' ', '+', '-', '\' (no newline marker) count towards position
        # We need the position relative to the start of the hunk
        position_in_hunk += 1

        if line.startswith('+'):
            line_content = line[1:].strip()
            if clean_target_line == line_content:
                app.logger.info(f"Found match for target line '{clean_target_line}' at diff position {position_in_hunk}")
                return position_in_hunk
        # No need to explicitly handle '-', ' ', '\' other than incrementing position_in_hunk

    app.logger.warning(f"Could not find exact match for line '{clean_target_line}' in the provided patch.")
    return -1

@tool("Post GitHub Review Comment Tool")
def post_review_comment_tool(
    repo_full_name: str,
    pr_number: int,
    commit_sha: str, # This should be the head_commit_sha
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
        commit_sha (str): The SHA of the commit the comment refers to (use head_commit_sha).
        filename (str): The path to the file being commented on.
        comment_body (str): The text of the comment.
        diff_position (Optional[int]): The position (line number) in the diff hunk (1-based) to comment on.
                                       If None or <= 0, posts as a general PR comment.
    Returns:
        str: A status message indicating success or failure.
    """
    app.logger.info(f"Tool 'post_review_comment_tool' called for {repo_full_name} PR #{pr_number}, Commit: {commit_sha[:7]}, File: {filename}, Position: {diff_position}")

    if not all([repo_full_name, pr_number, commit_sha, filename, comment_body]):
        error_msg = "Error: Missing required arguments for posting comment (repo_full_name, pr_number, commit_sha, filename, comment_body)."
        app.logger.error(error_msg)
        return error_msg

    try:
        repo = github_client.get_repo(repo_full_name)
        pull_request = repo.get_pull(pr_number) # Fetch PR object

        # --- Validate diff_position ---
        position_int = -1 # Default to invalid/general comment
        if diff_position is not None:
            try:
                position_int = int(diff_position)
                if position_int <= 0:
                     app.logger.warning(f"Received non-positive diff_position '{diff_position}', defaulting to general comment.")
                     position_int = -1 # Treat 0 or negative as general comment trigger
            except (ValueError, TypeError):
                app.logger.warning(f"Invalid diff_position type '{type(diff_position).__name__}' value '{diff_position}', defaulting to general comment.")
                position_int = -1

        # --- Attempt Inline Comment via Review API ---
        if position_int > 0:
            app.logger.info(f"Attempting to submit review with inline comment on: PR #{pr_number}, File: {filename}, Position: {position_int}, Commit: {commit_sha}")

            # Fetch the Commit object - REQUIRED for create_review
            try:
                commit_object = repo.get_commit(sha=commit_sha)
                app.logger.debug(f"Successfully fetched commit object for SHA {commit_sha}")
            except GithubException as commit_err:
                app.logger.error(f"Failed to fetch commit object for SHA {commit_sha}: {commit_err.status} - {commit_err.data}. Falling back to general comment.")
                fallback_body = (f"**Feedback para `{filename}` (Commit Inacessível):**\n\n"
                                 f"{comment_body}\n\n"
                                 f"_(Não foi possível buscar o commit `{commit_sha[:7]}` para postar este comentário precisamente. Postado como comentário geral.)_")
                try:
                    pull_request.create_issue_comment(fallback_body)
                    return f"Posted as general comment regarding {filename} (failed to fetch commit {commit_sha[:7]})."
                except Exception as fallback_err:
                    error_msg = f"Error: Failed to fetch commit {commit_sha[:7]} AND failed to post fallback comment: {fallback_err}"
                    app.logger.error(error_msg)
                    return error_msg

            # Prepare the comment dictionary for the create_review call
            review_comments = [
                {
                    "path": filename,
                    "position": position_int,
                    "body": comment_body # Already translated by the agent
                }
            ]

            try:
                pull_request.create_review(
                    commit=commit_object,
                    body="",  # No main review body needed for individual comments
                    event="COMMENT", # Just comment, don't approve/request changes
                    comments=review_comments
                )
                success_msg = f"Successfully submitted review with inline comment on {filename} at position {position_int}."
                app.logger.info(success_msg)
                return success_msg
            except GithubException as e:
                # Handle common errors like outdated diff / position invalid (422)
                is_outdated_or_invalid_pos = False
                if e.status == 422:
                    try:
                        error_data_str = str(e.data).lower()
                        if 'pull_request_review_thread' in error_data_str or 'position is invalid' in error_data_str or 'diff is outdated' in error_data_str:
                           is_outdated_or_invalid_pos = True
                    except Exception: pass # Ignore potential errors converting data to string

                if is_outdated_or_invalid_pos:
                    app.logger.warning(f"Failed submitting review comment (likely outdated diff/invalid position {position_int} for {filename}): {e.status} - {e.data}. Posting as general comment.")
                else:
                    # Log other GitHub API errors more verbosely
                    app.logger.error(f"GitHub API error submitting review for {filename} at pos {position_int}: {e.status} - {e.data.get('message', 'No details')}", exc_info=True)

                # Fallback to general PR comment for ANY review submission error
                fallback_body = (f"**Feedback para `{filename}` (Falha ao Postar na Linha):**\n\n"
                                 f"{comment_body}\n\n"
                                 f"_(Tentativa de postar na posição {position_int} do commit `{commit_sha[:7]}` falhou. Verifique se a linha ainda existe ou se o diff está atualizado. Erro: {e.status})_")
                try:
                    pull_request.create_issue_comment(fallback_body)
                    return f"Posted as general PR comment regarding {filename} after review submission failed (Pos: {position_int}, Error: {e.status})."
                except Exception as inner_e:
                    error_msg = f"Error: Failed original review submission for {filename} (Pos: {position_int}, Error: {e}) AND fallback PR comment also failed (Error: {inner_e})"
                    app.logger.error(error_msg, exc_info=True)
                    return error_msg # Report double failure
            except Exception as e: # Catch unexpected errors during review creation
                 app.logger.error(f"Unexpected error submitting review for {filename} at pos {position_int}: {e}", exc_info=True)
                 fallback_body = (f"**Feedback para `{filename}` (Erro Inesperado ao Postar na Linha):**\n\n"
                                  f"{comment_body}\n\n"
                                  f"_(Tentativa de postar na posição {position_int} do commit `{commit_sha[:7]}` encontrou um erro inesperado: {e})_")
                 try:
                     pull_request.create_issue_comment(fallback_body)
                     return f"Posted as general PR comment regarding {filename} after unexpected review submission error (Pos: {position_int})."
                 except Exception as inner_e:
                     error_msg = f"Error: Failed original review submission for {filename} (Pos: {position_int}, Unexpected Error: {e}) AND fallback PR comment also failed unexpectedly (Error: {inner_e})"
                     app.logger.error(error_msg, exc_info=True)
                     return error_msg

        # --- Post as General PR Comment (if position invalid or fallback) ---
        else:
            app.logger.info(f"Posting general PR comment for file {filename} (Position was invalid or fallback scenario).")
            try:
                # Enhance general comment body to clarify it's about a file but not a specific line
                general_comment_body = (f"**Feedback sobre `{filename}`:**\n\n"
                                        f"{comment_body}\n\n"
                                        f"_(Comentário referente às últimas alterações neste arquivo no PR. Posição específica não encontrada ou inválida: {diff_position})_")
                pull_request.create_issue_comment(general_comment_body)
                return f"Successfully posted general comment regarding {filename}."
            except GithubException as e:
                 error_msg = f"GitHub API error posting general PR comment regarding {filename}: {e.status} - {e.data.get('message', 'No details')}"
                 app.logger.error(error_msg, exc_info=True)
                 return error_msg
            except Exception as e:
                 error_msg = f"Unexpected error posting general PR comment for {filename}: {e}"
                 app.logger.error(error_msg, exc_info=True)
                 return error_msg

    # Catch errors during initial repo/PR lookup or other setup steps
    except UnknownObjectException:
        error_msg = f"Error: Pull Request #{pr_number} not found in repository {repo_full_name} when trying to post comment."
        app.logger.error(error_msg)
        return error_msg # Cannot post comment if PR not found
    except GithubException as e:
        error_msg = f"GitHub API error in post_review_comment_tool setup (fetching repo/PR {repo_full_name}/#{pr_number}): {e.status} - {e.data.get('message', 'No details')}"
        app.logger.error(error_msg, exc_info=True)
        return error_msg
    except Exception as e:
        error_msg = f"Outer unexpected error in post_review_comment_tool for {filename}: {e}"
        app.logger.error(error_msg, exc_info=True)
        return error_msg


# --- CrewAI Setup ---

def setup_crewai_agents_and_tasks(repo_full_name: str, pr_number: int, commit_sha: str):
    """Creates the CrewAI Agents and Tasks for code review, including severity and Portuguese comments."""
    app.logger.info(f"Setting up CrewAI agents and tasks for PR #{pr_number} (Commit: {commit_sha[:7]})")

    # --- Define Agents (Prompts refined based on previous feedback) ---
    code_analyzer = Agent(
        role='Expert Code Quality and Security Analyst',
        goal=f"""Analyze code patches (diffs) focusing ONLY on added lines (starting with '+').
        Identify issues across multiple dimensions: correctness, style, performance, security, and leftover artifacts.
        For EACH issue found on an added line:
        1. Identify the EXACT problematic line content (starting with '+').
        2. Assign a severity score: {PROBLEM_SCORE} (Problem), {ALERT_SCORE} (Alert), {IMPROVEMENT_OPPORTUNITY_SCORE} (Improvement).
        3. Provide a brief, descriptive English category (e.g., "Security", "Performance", "Style", "Bug Risk", "Debug Code").
        4. Write a clear, concise explanation of the issue AND a concrete suggestion for improvement IN ENGLISH.
        Output the findings as a JSON list of dictionaries, strictly adhering to the format. If no issues are found, output an empty list [].""",
        backstory=f"""You are a meticulous software architect and security expert. Your sole focus is scrutinizing *newly added* code lines within diffs to ensure they meet high standards of quality, security, and maintainability. You provide precise, actionable feedback IN ENGLISH, prioritized by potential impact using a {IMPROVEMENT_OPPORTUNITY_SCORE}/{ALERT_SCORE}/{PROBLEM_SCORE} scoring system.""",
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

    github_commenter = Agent(
        role='GitHub Review Integration Specialist (pt-BR)',
        goal=f"""Receive a JSON list of code analysis findings (line content, English comment, severity, category).
        For each finding:
        1. Determine the correct diff position using 'find_diff_position_tool'.
        2. Translate the English 'comment_body' into clear, constructive Brazilian Portuguese (pt-BR).
        3. Determine the Portuguese severity prefix ([PROBLEMA], [ALERTA], [SUGESTÃO]) and translate the English category to Portuguese.
        4. Construct the final comment string in Portuguese: "[Severity Prefix] (Categoria em Português): Comentário Traduzido".
        5. Use 'post_review_comment_tool' to post the formatted Portuguese comment to the specified GitHub PR file and position (or as a general comment if position finding fails).
        Provide a summary report of the posting actions.""",
        backstory="""Você é um especialista em automação que integra análises de código ao GitHub. Você traduz relatórios técnicos (inglês) para português brasileiro (pt-BR) claro e acionável, posicionando-os corretamente no diff do PR usando ferramentas de API. Sua missão é garantir que o feedback chegue ao desenvolvedor de forma eficiente e compreensível.""",
        verbose=True,
        llm=llm,
        tools=[find_diff_position_tool, post_review_comment_tool],
        allow_delegation=False
    )

    # --- Define Tasks (Prompts refined) ---
    analyze_file_patch_task = Task(
        description=f"""Analyze the provided code patch for the file '{{file_path}}' in PR #{pr_number} of {repo_full_name}.
        Focus *only* on lines starting with '+' (added lines). Ignore context (' ') and removed ('-') lines.

        Patch content for '{{file_path}}':
        ```diff
        {{patch_content}}
        ```

        **Analysis Goals:** Identify issues in ADDED lines related to Correctness, Security (OWASP), Performance, Style, Best Practices, and Leftover Artifacts (debug code, TODOs).

        **Output Format:** Return a JSON list of dictionaries. Each dictionary represents one distinct issue found in an added line and MUST contain:
        1.  `problematic_line_content`: string (The EXACT added line, starting with '+').
        2.  `severity_score`: integer ({PROBLEM_SCORE}, {ALERT_SCORE}, or {IMPROVEMENT_OPPORTUNITY_SCORE}).
        3.  `category`: string (Brief English category, e.g., "Security", "Style", "Debug Code").
        4.  `comment_body`: string (Concise ENGLISH explanation and suggestion).

        If an issue spans multiple added lines, report it against the *first* relevant added line.
        **If no issues are found in added lines, return an empty JSON list: `[]`.**""",
        expected_output=f"""A valid JSON list adhering strictly to the specified format. All string values must be properly escaped within the JSON.
        Example of valid JSON output:
        ```json
        [
          {{
            "problematic_line_content": "+ console.log('User data:', userData); // DEBUG",
            "severity_score": {ALERT_SCORE},
            "category": "Debug Code",
            "comment_body": "This console.log statement appears to be for debugging and should be removed before merging."
          }},
          {{
            "problematic_line_content": "+ result = unsafe_execute(user_input)",
            "severity_score": {PROBLEM_SCORE},
            "category": "Security",
            "comment_body": "Executing commands or queries with unsanitized user input ('user_input') can lead to injection vulnerabilities. Ensure the input is properly validated and sanitized or parameterized."
          }}
        ]
        ```
        Example if no issues are found:
        ```json
        []
        ```
        """,
        agent=code_analyzer,
        # No async execution needed here, runs first
    )

    post_file_comments_task = Task(
        description=f"""Process the JSON list of code issues for file '{{file_path}}' received from the Code Analyzer.
        The input is the JSON list generated by the previous task.
        Use tools to post these findings as comments IN BRAZILIAN PORTUGUESE (pt-BR) on GitHub Pull Request #{pr_number} in repository {repo_full_name}.

        **Input:** JSON list from the analysis task.
        **Context for Tools:**
        - Repository: {repo_full_name}
        - PR Number: {pr_number}
        - Commit SHA: {commit_sha}
        - File Path: {{file_path}}
        - Full Patch Content (for positioning):
          ```diff
          {{patch_content}}
          ```

        **Execution Steps for EACH issue in the input list:**
        1.  Extract `problematic_line_content`, `severity_score`, `category` (Eng), `comment_body` (Eng).
        2.  Translate English `comment_body` to Brazilian Portuguese (pt-BR).
        3.  Translate English `category` to a Portuguese equivalent (e.g., "Security" -> "Segurança", "Style" -> "Estilo").
        4.  Determine severity prefix based on `severity_score`: {PROBLEM_SCORE} -> "[PROBLEMA]", {ALERT_SCORE} -> "[ALERTA]", {IMPROVEMENT_OPPORTUNITY_SCORE} -> "[SUGESTÃO]".
        5.  Construct final comment: `"[Severity Prefix] (Categoria em Português): Comentário Traduzido"`
        6.  Use `find_diff_position_tool` with `patch_content` and `problematic_line_content` to get the position.
        7.  Use `post_review_comment_tool` with repo/PR/commit/file info, the *final Portuguese comment*, and the found position. If position is -1, pass `diff_position=None` or `diff_position=-1` to trigger general comment fallback in the tool.

        Execute posting for ALL issues individually. Report the outcome of each posting attempt.
        """,
        expected_output=f"""A concise summary report IN ENGLISH describing the outcome of posting each comment for file '{{file_path}}'.
        Indicate success (inline/general) or failure for each comment, mentioning the Portuguese severity/category.
        Example:
        - Posting '[SUGESTÃO] (Estilo) Comment about spacing...': Success (inline comment at position 5).
        - Posting '[ALERTA] (Código de Debug) Comment about console.log...': Success (inline comment at position 12).
        - Posting '[PROBLEMA] (Segurança) Comment about input validation...': Success (posted as general comment, position finding failed).
        - Posting '[ALERTA] (Performance) Comment about N+1 query...': Failed (GitHub API error 403).
        """,
        agent=github_commenter,
        context=[analyze_file_patch_task], # Depends on the analysis results
    )

    # Create and return the Crew
    app.logger.info("CrewAI agents and tasks setup complete.")
    return Crew(
        agents=[code_analyzer, github_commenter],
        tasks=[analyze_file_patch_task, post_file_comments_task],
        process=Process.sequential,
        verbose=2 # Use verbose level 2 for more detailed crew execution logs
    )

# --- Flask Webhook Handler ---

@app.route("/webhook", methods=["POST"])
def github_webhook():
    """Handles incoming GitHub webhook events for PRs."""
    app.logger.info("Webhook received.")

    # --- Signature Verification ---
    signature_header = request.headers.get('X-Hub-Signature-256') # Use SHA256
    if not verify_signature(request.data, GITHUB_WEBHOOK_SECRET, signature_header):
        app.logger.error("Invalid webhook signature.")
        abort(403, "Invalid signature.") # Use abort for security failures

    data = request.json
    event = request.headers.get('X-GitHub-Event')
    action = data.get("action") if isinstance(data, dict) else None

    # Ensure data is a dictionary before proceeding
    if not isinstance(data, dict):
         app.logger.warning("Received non-JSON payload or invalid format.")
         return jsonify({"message": "Invalid payload format"}), 400

    app.logger.debug(f"Event: {event}, Action: {action}")

    # Process only relevant PR events
    if event == "pull_request" and action in ["opened", "synchronize", "reopened"]:
        if "pull_request" not in data or "repository" not in data:
            app.logger.warning("Webhook payload missing 'pull_request' or 'repository' data.")
            return jsonify({"message": "Ignoring event: Missing necessary data"}), 200

        pr_data = data["pull_request"]
        pr_number = pr_data["number"]
        repo_data = data["repository"]
        repo_full_name = repo_data["full_name"]

        # Avoid processing drafts
        if pr_data.get("draft", False):
            app.logger.info(f"Ignoring event for draft PR #{pr_number} in {repo_full_name}.")
            return jsonify({"message": "Ignoring draft pull request"}), 200

        app.logger.info(f"Processing '{action}' event for PR #{pr_number} in {repo_full_name}")

        try:
            # 1. Get changed files and context (using the tool's function directly)
            file_context = get_changed_files_tool.func(repo_full_name=repo_full_name, pr_number=pr_number)

            if file_context["error"]:
                error_message = f"Failed to get changed files: {file_context['error']}"
                app.logger.error(error_message)
                # Try to post error comment to PR
                try:
                    repo = github_client.get_repo(repo_full_name)
                    pr = repo.get_pull(pr_number)
                    pr.create_issue_comment(f"⚠️ **Erro na Revisão por IA:**\n\nNão foi possível obter as alterações de arquivos para análise.\nErro: `{file_context['error']}`")
                except Exception as comment_err:
                    app.logger.error(f"Failed to post 'get files error' comment: {comment_err}")
                # Return error to GitHub webhook processor
                return jsonify({"error": error_message}), 500

            changed_files = file_context["files"]
            patches_dict = file_context["patches"]
            head_commit_sha = file_context["head_commit_sha"] # Use the correct key name

            if not changed_files:
                app.logger.info("No relevant file changes detected in the PR.")
                try:
                    repo = github_client.get_repo(repo_full_name)
                    pr = repo.get_pull(pr_number)
                    pr.create_issue_comment("ℹ️ **Revisão por IA:** Nenhum arquivo de código relevante encontrado para revisar nesta atualização.")
                except Exception as comment_err:
                    app.logger.error(f"Failed to post 'no files' comment: {comment_err}")
                return jsonify({"message": "No relevant files to review"}), 200

            # head_commit_sha should have been validated inside get_changed_files_tool already
            if not head_commit_sha:
                 # This case should theoretically be handled by the check inside the tool, but double-check
                 error_message = "Critical Error: Head commit SHA is missing after getting files."
                 app.logger.error(error_message)
                 try:
                    repo = github_client.get_repo(repo_full_name)
                    pr = repo.get_pull(pr_number)
                    pr.create_issue_comment("⚠️ **Erro Crítico na Revisão por IA:**\n\nNão foi possível determinar o SHA do commit principal. Não é possível postar comentários.")
                 except Exception as comment_err:
                    app.logger.error(f"Failed to post commit SHA error comment: {comment_err}")
                 return jsonify({"error": error_message}), 500

            # 2. Setup CrewAI for this specific PR and commit
            review_crew = setup_crewai_agents_and_tasks(repo_full_name, pr_number, head_commit_sha)

            # --- Initialize Summary Data ---
            overall_severity_counts = {'PROBLEMA': 0, 'ALERTA': 0, 'SUGESTÃO': 0}
            files_analyzed_count = 0
            files_with_errors = [] # List to store filenames that caused errors during processing
            # detailed_posting_results = {} # Optional: store commenter summary per file

            app.logger.info(f"Starting analysis of {len(changed_files)} files for commit {head_commit_sha[:7]}...")

            # 3. Iterate and Run Crew for each file
            for file_info in changed_files:
                filename = file_info['filename']
                patch_content = patches_dict.get(filename)

                if not patch_content:
                    app.logger.warning(f"Patch content missing for file {filename}. Skipping analysis.")
                    files_with_errors.append(f"{filename} (Missing Patch)")
                    continue

                app.logger.info(f"\n--- Analyzing file: {filename} ---")
                crew_inputs = {'file_path': filename, 'patch_content': patch_content}
                analysis_task_output_str = None # Store the raw output string

                try:
                    # Kick off the crew for this file
                    posting_result_summary = review_crew.kickoff(inputs=crew_inputs)
                    app.logger.info(f"--- Crew finished for file: {filename} ---")
                    app.logger.debug(f"Commenter Task Output Summary for {filename}: {posting_result_summary}")
                    # detailed_posting_results[filename] = posting_result_summary

                    # --- Access Analyzer Task Output for Summary ---
                    if review_crew.tasks and len(review_crew.tasks) > 0:
                        analyzer_task = review_crew.tasks[0] # Assuming analyzer is the first task
                        if hasattr(analyzer_task, 'output') and analyzer_task.output:
                            # Access raw output which should be the JSON string
                            analysis_task_output_str = analyzer_task.output.raw_output if hasattr(analyzer_task.output, 'raw_output') else str(analyzer_task.output)
                            app.logger.debug(f"Analyzer Task Raw JSON Output for {filename}: {analysis_task_output_str}")
                        else:
                            app.logger.warning(f"Analyzer task output object not found or empty for {filename}.")
                    else:
                         app.logger.warning(f"No tasks found in crew after execution for {filename}.")

                    # --- Process the Analyzer's JSON Findings ---
                    if analysis_task_output_str:
                        findings = []
                        try:
                            # Clean potential markdown code blocks before parsing
                            cleaned_output = re.sub(r'^```json\s*|\s*```$', '', analysis_task_output_str, flags=re.MULTILINE).strip()
                            if cleaned_output:
                                findings = json.loads(cleaned_output)
                            else:
                                app.logger.warning(f"Cleaned analyzer output is empty for {filename}. Original: {analysis_task_output_str}")

                            # Ensure findings is a list before iterating
                            if isinstance(findings, list):
                                file_severity_counts = {'PROBLEMA': 0, 'ALERTA': 0, 'SUGESTÃO': 0}
                                for finding in findings:
                                    if isinstance(finding, dict):
                                        severity = finding.get('severity_score')
                                        if severity == PROBLEM_SCORE:
                                            overall_severity_counts['PROBLEMA'] += 1
                                            file_severity_counts['PROBLEMA'] += 1
                                        elif severity == ALERT_SCORE:
                                            overall_severity_counts['ALERTA'] += 1
                                            file_severity_counts['ALERTA'] += 1
                                        elif severity == IMPROVEMENT_OPPORTUNITY_SCORE:
                                            overall_severity_counts['SUGESTÃO'] += 1
                                            file_severity_counts['SUGESTÃO'] += 1
                                        # else: ignore invalid severity scores silently or log warning
                                    else:
                                         app.logger.warning(f"Finding item is not a dictionary in {filename}: {finding}")
                                app.logger.info(f"Severity counts tallied for {filename}: {file_severity_counts}")
                            else:
                                app.logger.warning(f"Parsed findings for {filename} is not a list: {type(findings)}. Raw output: {analysis_task_output_str}")

                        except json.JSONDecodeError as json_err:
                            app.logger.error(f"Failed to decode JSON from analyzer output for {filename}: {json_err}. Output was: {analysis_task_output_str}")
                            files_with_errors.append(f"{filename} (JSON Decode Error)")
                        except Exception as proc_err:
                            app.logger.error(f"Error processing findings for {filename}: {proc_err}", exc_info=True)
                            files_with_errors.append(f"{filename} (Findings Processing Error)")
                    else:
                         app.logger.warning(f"No valid analysis output string to process for {filename}")

                    files_analyzed_count += 1

                except Exception as file_crew_error:
                    app.logger.error(f"Error running CrewAI workflow for file {filename}: {file_crew_error}", exc_info=True)
                    files_with_errors.append(f"{filename} (Crew Workflow Error)")
                    # Attempt to post a general comment about the failure for this file
                    try:
                        error_comment_body = (
                            f"**⚠️ Erro na Revisão por IA para `{filename}`:**\n\n"
                            f"O processo de revisão encontrou um erro inesperado ao analisar este arquivo.\n"
                            f"Detalhes: `{str(file_crew_error)}`\n\n"
                            f"Por favor, revise este arquivo manualmente."
                        )
                        # Use the tool function directly for robust error reporting
                        post_review_comment_tool( # Call tool directly, not via agent
                             repo_full_name=repo_full_name,
                             pr_number=pr_number,
                             commit_sha=head_commit_sha, # Use the SHA we have
                             filename=filename,
                             comment_body=error_comment_body,
                             diff_position=None # Post as general comment
                        )
                        app.logger.info(f"Posted error comment for failed analysis of {filename}")
                    except Exception as comment_error:
                        app.logger.error(f"CRITICAL: Failed to post error comment for {filename} after workflow error: {comment_error}", exc_info=True)

            # --- Post Final Summary Comment ---
            app.logger.info(f"Analysis loop finished. Overall severity counts: {overall_severity_counts}")
            total_findings_identified = sum(overall_severity_counts.values())

            summary_lines = [
                f"### ✅ Revisão por IA Concluída",
                f"**Commit Analisado:** `{head_commit_sha[:7]}`",
                f"**Arquivos Analisados:** {files_analyzed_count} de {len(changed_files)} relevantes.",
                "", # Separator
                f"**📊 Resumo dos Pontos Identificados:**",
                f"- **{overall_severity_counts['PROBLEMA']}** :octagonal_sign: [PROBLEMA](## \"Questões críticas ou bugs prováveis que exigem atenção imediata.\")",
                f"- **{overall_severity_counts['ALERTA']}** :warning: [ALERTA](## \"Possíveis problemas, otimizações de performance, ou código de debug remanescente.\")",
                f"- **{overall_severity_counts['SUGESTÃO']}** :bulb: [SUGESTÃO](## \"Oportunidades de melhoria de estilo, legibilidade ou boas práticas.\")",
                f"- **Total:** {total_findings_identified}",
                # Add link to documentation or run details if available
            ]

            if files_with_errors:
                 summary_lines.append("\n**⚠️ Atenção:**")
                 summary_lines.append(f"Ocorreram erros ao processar os seguintes arquivos (verifique os logs do sistema e revise-os manualmente):")
                 for f_err in files_with_errors:
                      summary_lines.append(f"- `{f_err}`")

            summary_message = "\n".join(summary_lines)

            try:
                repo = github_client.get_repo(repo_full_name)
                pr = repo.get_pull(pr_number)
                pr.create_issue_comment(summary_message)
                app.logger.info("Successfully posted the final summary comment.")
            except Exception as summary_comment_error:
                app.logger.error(f"Failed to post final summary comment: {summary_comment_error}", exc_info=True)

            # Return success response to GitHub
            return jsonify({
                "message": "Review process completed.",
                "files_analyzed": files_analyzed_count,
                "total_findings": total_findings_identified,
                "severity_counts": overall_severity_counts,
                "files_with_errors": files_with_errors
                }), 200

        # --- Catch top-level errors during webhook processing ---
        except GithubException as e:
            error_msg = f"A GitHub API error occurred during PR processing: {e.status} - {e.data.get('message', 'No details')}"
            app.logger.error(error_msg, exc_info=True)
            # Attempt to post error on PR if possible
            try:
                 repo = github_client.get_repo(repo_full_name)
                 pr = repo.get_pull(pr_number)
                 pr.create_issue_comment(f"⚠️ **Erro Crítico na Revisão por IA:**\n\nOcorreu um erro de API do GitHub ({e.status}) que impediu a conclusão da revisão.\nErro: `{e.data.get('message', 'Detalhes não disponíveis')}`\nConsulte os logs do sistema.")
            except Exception: pass # Avoid further errors if comment posting fails
            return jsonify({"error": error_msg}), 500 # Signal server error
        except Exception as e:
            error_msg = f"An unexpected error occurred during the review process: {e}"
            app.logger.error(error_msg, exc_info=True)
            # Attempt to post a general error comment
            try:
                 # Need repo_full_name and pr_number if available at this point
                 if 'repo_full_name' in locals() and 'pr_number' in locals():
                     repo = github_client.get_repo(repo_full_name)
                     pr = repo.get_pull(pr_number)
                     pr.create_issue_comment(f"⚠️ **Erro Crítico no Sistema de Revisão por IA:**\n\nOcorreu um erro inesperado: `{e}`.\nA revisão pode estar incompleta. Verifique os logs do sistema.")
            except Exception:
                 pass # Ignore error during final error reporting
            return jsonify({"error": error_msg}), 500 # Signal server error

    else:
        # Ignore events other than relevant PR actions
        app.logger.debug(f"Ignoring event type '{event}' with action '{action}'")
        return jsonify({"message": "Ignoring event: Not a relevant PR action"}), 200

# --- Main Execution ---

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001)) # Changed default port slightly to avoid common conflicts
    app.logger.info(f"Starting AI Code Review server on http://0.0.0.0:{port}")
    # Use debug=False for production/staging. Set to True ONLY for local dev.
    # Use a production-ready WSGI server like Gunicorn or Waitress instead of app.run in production.
    app.run(host="0.0.0.0", port=port, debug=False)