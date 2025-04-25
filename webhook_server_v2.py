from flask import Flask, request, jsonify
import os
import re
import requests
from github import Github, GithubException
from crewai import Agent, Task, Crew, Process
from typing import Optional, List, Dict, Union, Any
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from github.PullRequest import PullRequest

# Load environment variables
load_dotenv()

app = Flask(__name__)

# GitHub authentication
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variables
if not all([GITHUB_TOKEN, OPENAI_API_KEY]):
    missing_vars = []
    if not GITHUB_TOKEN:
        missing_vars.append("GITHUB_TOKEN")
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file")
    exit(1)

# Configure LLM client
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, api_key=OPENAI_API_KEY)

# Initialize GitHub API client
github_client = Github(GITHUB_TOKEN)

# Create a class to hold the PR context
class PRContext:
    def __init__(self, pull_request):
        self.pull_request = pull_request
        self.file_patches = {}
        self.last_commit_sha = None
        
        # Initialize last commit SHA
        commits = self.pull_request.get_commits()
        if commits.totalCount > 0:
            self.last_commit_sha = commits[commits.totalCount - 1].sha

# Global PR context for tool functions to access
_PR_CONTEXT = None

# Define simple tool functions
def get_changed_files():
    """Get a list of changed files in the pull request."""
    try:
        files = _PR_CONTEXT.pull_request.get_files()
        changed_files_data = []
        excluded_extensions = ('.lock', '.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico')
        excluded_files = ('package-lock.json', 'yarn.lock', 'pnpm-lock.yaml')

        for file in files:
            if file.status == 'removed' or \
               file.filename.lower().endswith(excluded_extensions) or \
               os.path.basename(file.filename.lower()) in excluded_files:
                print(f"Skipping file: {file.filename} (status: {file.status})")
                continue

            _PR_CONTEXT.file_patches[file.filename] = file.patch
            changed_files_data.append({
                "filename": file.filename,
                "status": file.status,
            })
            
        print(f"Found {len(changed_files_data)} relevant changed files.")
        return changed_files_data
    except Exception as e:
        print(f"Error getting changed files: {e}")
        return [{"error": f"Error getting changed files: {e}"}]

def get_file_patch(filename):
    """Get the patch content for a specific file."""
    if filename in _PR_CONTEXT.file_patches:
        patch_content = _PR_CONTEXT.file_patches[filename]
        max_patch_length = 10000
        if len(patch_content) > max_patch_length:
            print(f"Warning: Patch for {filename} truncated to {max_patch_length} characters.")
            return patch_content[:max_patch_length]
        return patch_content
    else:
        return f"Error: Patch for file '{filename}' not found in cache."

def post_review_comment(filename, comment_body, diff_position):
    """Post a review comment on a specific position in a file."""
    try:
        if not filename or not comment_body or diff_position is None:
            return "Error: filename, comment_body, and diff_position are required."
        
        # Convert diff_position to int if it's a string
        if isinstance(diff_position, str) and diff_position.isdigit():
            diff_position = int(diff_position)
        
        if not _PR_CONTEXT.last_commit_sha:
            return "Error: Could not determine the last commit SHA."

        print(f"Attempting to post comment on: {filename}, Position: {diff_position}")
        print(f"Comment: {comment_body}")

        _PR_CONTEXT.pull_request.create_review_comment(
            body=comment_body,
            commit_id=_PR_CONTEXT.last_commit_sha,
            path=filename,
            position=diff_position
        )
        return f"Successfully posted comment on {filename} at position {diff_position}."
    except Exception as e:
        try:
            fallback_body = (f"**Review Feedback (Failed to post on specific line for `{filename}`):**\n\n"
                         f"{comment_body}")
            _PR_CONTEXT.pull_request.create_issue_comment(fallback_body)
            return f"Failed to post comment at position {diff_position} for {filename}. Posted as general PR comment. Error: {str(e)}"
        except Exception as inner_e:
            print(f"Failed to post general comment as fallback: {inner_e}")
            return f"Failed to post comment for {filename}. Error: {str(e)}"

def find_diff_position(filename, target_line_content):
    """Find the position in the diff for a specific line content."""
    patch = _PR_CONTEXT.file_patches.get(filename)
    if not patch:
        print(f"Error: Patch not found for {filename} in find_diff_position.")
        return -1  # Use -1 to indicate "not found"

    lines = patch.split('\n')
    hunk_header_re = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@')

    current_file_line_number = 0
    position_in_diff = 0

    for line in lines:
        position_in_diff += 1
        match = hunk_header_re.match(line)
        if match:
            current_file_line_number = int(match.group(1))
            continue

        if line.startswith('-') or line.startswith('\\') or not line.strip():
            continue
        elif line.startswith('+'):
            line_content = line[1:]
            if target_line_content.strip() == line_content.strip():
                print(f"Found match for '{target_line_content}' at diff position {position_in_diff}")
                return position_in_diff
            current_file_line_number += 1
        else:
            current_file_line_number += 1

    print(f"Warning: Could not find exact match for line '{target_line_content}' in patch for {filename}.")
    return -1  # Use -1 to indicate "not found"

def setup_crew(pull_request):
    # Create and set global PR context
    global _PR_CONTEXT
    _PR_CONTEXT = PRContext(pull_request)
    
    # Define Agents
    code_analyzer = Agent(
        role='Code Analyzer',
        goal="Analyze code patches for issues and improvements",
        backstory="You are a detail-oriented code reviewer who finds bugs, style issues, and leftover debugging code.",
        verbose=True,
        llm=llm
    )

    github_commenter = Agent(
        role='GitHub Commenter',
        goal="Post review comments on GitHub pull requests",
        backstory="You carefully analyze code review feedback and post it as comments on GitHub.",
        verbose=True,
        llm=llm
    )

    # Define Tasks
    analyze_file_patch_task = Task(
        description="""Analyze the provided code patch for file '{file_path}'.
        The patch content is:
        ```diff
        {patch_content}
        ```
        Identify issues based on your role (bugs, quality, performance, 'console.log', 'debugger').
        Focus ONLY on lines starting with '+'.
        For each issue found, clearly state:
        1. The exact line content (starting with '+') where the issue occurs.
        2. A concise description of the issue and a suggestion for improvement.""",
        expected_output="""A list of dictionaries, where each dictionary represents an issue found.
        Each dictionary must contain:
        - 'file_path': The path of the file being analyzed.
        - 'problematic_line_content': The exact content of the line (including the leading '+') where the issue is.
        - 'comment_body': The review comment text explaining the issue and suggestion.""",
        agent=code_analyzer
    )

    post_file_comments_task = Task(
        description="""Process the list of identified issues for the file '{file_path}' from the analysis step.
        For each issue:
        1. Extract the file_path, problematic_line_content (remove the leading '+'), and comment_body.
        2. For each issue, call these functions in sequence:
           a. diff_position = find_diff_position(file_path, problematic_line_content without the '+')
           b. If diff_position is greater than 0:
              post_review_comment(file_path, comment_body, diff_position)
           c. If diff_position is -1 or less, post a general comment.""",
        expected_output="A summary report confirming which comments were successfully posted for file '{file_path}'.",
        agent=github_commenter
    )

    return Crew(
        agents=[code_analyzer, github_commenter],
        tasks=[analyze_file_patch_task, post_file_comments_task],
        process=Process.sequential,
        verbose=1
    )

@app.route("/webhook", methods=["POST"])
def github_webhook():
    data = request.json
    print("Received webhook data:", data)
    print("Action:", data.get("action"))
    print("Has pull_request:", "pull_request" in data)

    if data.get("action") in ["opened", "synchronize"] and "pull_request" in data:
        pr_number = data["pull_request"]["number"]
        repo_full_name = data["repository"]["full_name"]
        
        try:
            # Get the repository and pull request
            repo = github_client.get_repo(repo_full_name)
            pull_request = repo.get_pull(pr_number)
            
            # Setup the crew
            crew = setup_crew(pull_request)
            
            # Create monkeypatched functions for the LLM to use
            def analyze_file(filename, patch_content):
                return crew.kickoff(inputs={'file_path': filename, 'patch_content': patch_content})
            
            # Get changed files directly using the function
            changed_files = get_changed_files()
            
            if not changed_files or (isinstance(changed_files, list) and changed_files and "error" in changed_files[0]):
                error_msg = changed_files[0]["error"] if changed_files and isinstance(changed_files[0], dict) and "error" in changed_files[0] else "No relevant files to review"
                print(f"Could not retrieve changed files or no relevant files changed: {error_msg}")
                return jsonify({"message": error_msg}), 200
            
            print(f"\nAnalyzing {len(changed_files)} files...")
            all_results = []
            
            for file_info in changed_files:
                filename = file_info['filename']
                print(f"\n--- Analyzing file: {filename} ---")
                
                patch_content = get_file_patch(filename)
                
                if isinstance(patch_content, str) and (patch_content.startswith("Error:") or not patch_content.strip()):
                    print(f"Could not get patch or patch is empty for {filename}. Skipping.")
                    continue
                
                # Run the crew for this file
                try:
                    result = analyze_file(filename, patch_content)
                    print(f"--- Finished analyzing file: {filename} ---")
                    print(f"Result: {result}")
                    all_results.append({filename: result})
                except Exception as file_error:
                    print(f"Error analyzing file {filename}: {file_error}")
                    # Post a generic comment about the error
                    try:
                        pull_request.create_issue_comment(
                            f"**Error reviewing file `{filename}`:**\n\n"
                            f"The AI review system encountered an error while analyzing this file. "
                            f"Please check manually for any issues."
                        )
                    except Exception as comment_error:
                        print(f"Error posting comment: {comment_error}")
            
            # Post a summary comment if needed
            if not all_results:
                try:
                    pull_request.create_issue_comment(
                        "**AI Code Review Summary:**\n\n"
                        "The automated review process completed, but no specific issues were found in the analyzed files."
                    )
                except Exception as comment_error:
                    print(f"Error posting summary comment: {comment_error}")
            
            return jsonify({"message": "Review completed", "files_reviewed": len(all_results)}), 200
            
        except Exception as e:
            print(f"Error during review process: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": "Not a PR event", 
        "received_data": data,
        "action": data.get("action"),
        "has_pr": "pull_request" in data
    }), 400

# Expose functions for CrewAI to use
globals()['get_changed_files'] = get_changed_files
globals()['get_file_patch'] = get_file_patch
globals()['post_review_comment'] = post_review_comment
globals()['find_diff_position'] = find_diff_position

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 