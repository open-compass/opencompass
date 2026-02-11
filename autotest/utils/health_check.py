from time import sleep

import fire
import requests


def health_check(url: str = 'http://0.0.0.0:23333', timeout: int = 300):
    for i in range(int(timeout / 5)):
        try:
            sleep(5)
            response = requests.get(f'{url}/health', timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            continue
    raise TimeoutError(f'Health check failed after {timeout} seconds.')


if __name__ == '__main__':
    fire.Fire(health_check)
