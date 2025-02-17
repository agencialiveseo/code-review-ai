from flask import Flask, request, jsonify
import os
import requests
from github import Github
from crew_setup import crew  # Import the Crew setup
from dotenv import load_dotenv

load_dotenv()  # Adicione esta linha no início

app = Flask(__name__)

# GitHub authentication (use GitHub Actions secret or a .env file)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
print(GITHUB_TOKEN)
print(GITHUB_REPO)

# Initialize GitHub API client
github_client = Github(GITHUB_TOKEN)
repo = None

try:
    print(f"Attempting to access repository: {GITHUB_REPO}")
    print(f"Using token: {GITHUB_TOKEN[:10]}...")  # Mostra apenas os primeiros 10 caracteres do token
    
    # Teste a conexão com o GitHub
    user = github_client.get_user()
    print(f"Authenticated as: {user.login}")
    
    repo = github_client.get_repo(GITHUB_REPO)
    print(f"Successfully accessed the repository: {repo.full_name}")
except Exception as e:
    print(f"Error accessing the repository: {e}")
    print(f"Type of error: {type(e)}")
    raise e

@app.route("/webhook", methods=["POST"])
def github_webhook():
    data = request.json
    print("Received webhook data:", data)  # Debug
    print("Action:", data.get("action"))   # Debug
    print("Has pull_request:", "pull_request" in data)  # Debug

    # Ensure the event is for a pull request
    if data.get("action") in ["opened", "synchronize"] and "pull_request" in data:
        pr_number = data["pull_request"]["number"]
        pr_diff_url = data["pull_request"]["diff_url"]
        
        # Run Crew AI for review
        review_results = crew.kickoff(inputs={"diff": pr_diff_url})
        
        # Converte os resultados para string se necessário
        if isinstance(review_results, tuple):
            review_results = [str(item) for item in review_results]
        elif not isinstance(review_results, list):
            review_results = [str(review_results)]
            
        # Post the review as a comment on the PR
        pr = repo.get_pull(pr_number)
        comment = "\n".join(review_results)
        print(f"Posting comment: {comment}")  # Debug
        pr.create_issue_comment(comment)

        return jsonify({"message": "Review completed"}), 200

    return jsonify({
        "message": "Not a PR event", 
        "received_data": data,
        "action": data.get("action"),
        "has_pr": "pull_request" in data
    }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
