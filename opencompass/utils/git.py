import subprocess


def get_git_root() -> str:
    cmd = ['git', 'rev-parse', '--show-toplevel']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    return result.stdout.decode('utf-8').strip()


def get_latest_commit(branch: str) -> str:
    cmd = ['git', 'rev-parse', branch]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    return result.stdout.decode('utf-8').strip()
