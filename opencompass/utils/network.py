import os
import platform
import socket
import subprocess
from typing import Dict, Optional, Tuple

import requests

from .logging import get_logger

logger = get_logger()


def setup_proxies(proxy_env_name):
    try:
        proxy_url = os.environ[proxy_env_name]
        if not proxy_url:
            raise ValueError('Proxy URL environment variable is empty')

        # Validate proxy URL format
        if not proxy_url.startswith(('http://', 'https://')):
            raise ValueError('Proxy URL must start with http:// or https://')

        proxies = {'http://': proxy_url, 'https://': proxy_url}
        return proxies

    except KeyError:
        # Handle the proxy_env_name environment is nonexistent
        logger.warning(f'{proxy_env_name} environment variable not found')
        return {}

    except ValueError as e:
        # Handle invalid proxy URL format
        logger.error(f'Invalid proxy configuration: {str(e)}')
        return None

    except Exception as e:
        # Handle any unexpected errors
        logger.error(f'Unexpected error while setting up proxies: {str(e)}')
        return None


def check_network_connectivity(
        host: str = '8.8.8.8',
        port: int = 53,
        timeout: float = 3,
        proxies: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """Check network connectivity using multiple methods with optional proxy
    support.

    Args:
        host: str, target host to check (default: Google DNS "8.8.8.8")
        port: int, target port to check (default: 53 for DNS)
        timeout: float, timeout in seconds (default: 3)
        proxies: Optional[Dict[str, str]], proxy configuration (default: None)
            Example: {
                'http': 'http://proxy:8080',
                'https': 'https://proxy:8080'
            }

    Returns:
        Tuple[bool, str]: (is_connected, message)
    """

    # Method 1: Socket connection test (direct connection, no proxy)
    def check_socket() -> bool:
        try:
            socket.create_connection((host, port), timeout=timeout)
            return True
        except OSError:
            return False

    # Method 2: HTTP request test (supports proxy)
    def check_http() -> bool:
        try:
            response = requests.get('http://www.google.com',
                                    timeout=timeout,
                                    proxies=proxies)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # Method 3: Ping test (direct connection, no proxy)
    def check_ping() -> bool:
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '1', host]
        try:
            return subprocess.call(command,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL) == 0
        except subprocess.SubprocessError:
            return False

    # Try all methods
    is_socket_connected = check_socket()
    is_http_connected = check_http()
    is_ping_connected = check_ping()

    # Generate detailed message including proxy information
    status_msg = (
        f'Network Status:\n'
        f"Socket Test: {'Success' if is_socket_connected else 'Failed'}\n"
        f"HTTP Test (via {'Proxy' if proxies else 'Direct'}): "
        f"{'Success' if is_http_connected else 'Failed'}\n"
        f"Ping Test: {'Success' if is_ping_connected else 'Failed'}")

    # If using proxy, add proxy details to message
    if proxies:
        status_msg += '\nProxy Configuration:'
        for protocol, proxy in proxies.items():
            status_msg += f'\n  {protocol}: {proxy}'

    is_connected = any(
        [is_socket_connected, is_http_connected, is_ping_connected])
    logger.info(status_msg)
    return is_connected, status_msg


def check_url_accessibility(
        url: str,
        timeout: float = 3,
        proxies: Optional[Dict[str,
                               str]] = None) -> Tuple[bool, Optional[int]]:
    """Check if a specific URL is accessible through optional proxy.

    Args:
        url: str, target URL to check
        timeout: float, timeout in seconds (default: 3)
        proxies: Optional[Dict[str, str]], proxy configuration (default: None)
            Example: {
                'http': 'http://proxy:8080',
                'https': 'https://proxy:8080'}

    Returns:
        Tuple[bool, Optional[int]]: (is_accessible, status_code)
    """
    try:
        response = requests.get(url, timeout=timeout, proxies=proxies)
        return True, response.status_code
    except requests.RequestException as e:
        logger.error(f'Failed to access URL {url}: {str(e)}')
        return False, None
