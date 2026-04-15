import json
import urllib.request
import os

def lambda_handler(event, context):
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPO"]
    workflow = os.environ["GITHUB_WORKFLOW"]

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    
    data = json.dumps({"ref": "main"}).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    
    with urllib.request.urlopen(req) as response:
        print(f"GitHub API response: {response.status}")
    
    return {"statusCode": 200, "body": "Training triggered"}