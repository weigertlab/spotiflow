import os

def is_github_actions_running() -> bool:
    return os.getenv('GITHUB_ACTIONS') == 'true'
