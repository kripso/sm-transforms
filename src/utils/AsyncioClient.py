import asyncio
import aiohttp
from typing import Callable
from dataclasses import dataclass
from utils.file_handling import load_asyncio_client_config

CONFIG = load_asyncio_client_config()
TIMEOUT_CONFIG = CONFIG.get('ASYNC_TIMEOUT', {})
RETRYABLE_ERROR_DESCRIPTIONS = ("Client Timeout Error", "Client Connector Error")
ASYNC_TIMEOUT = aiohttp.ClientTimeout(
    total=TIMEOUT_CONFIG.get("total"),
    sock_connect=TIMEOUT_CONFIG.get("sock_connect", 5),
    sock_read=TIMEOUT_CONFIG.get("sock_read", 5),
)


@dataclass
class ClientResponse:
    passed: bool = None
    redirected: bool = None
    response: dict = None
    error: str = None
    status_code: int = None

    def __post_init__(self):
        self.passed = True if self.error is None else False

        if self.passed:
            if 200 <= self.status_code < 300:
                self.redirected = False
            if 300 <= self.status_code < 400:
                self.redirected = True


def retry(func):
    async def _retry_function(*args, **kwargs):
        for _ in range(CONFIG.get('ASYNC_RETRY_COUNT', 5)):
            try:
                response = await func(*args, **kwargs)
                if response.error is not None and response.error.startswith(RETRYABLE_ERROR_DESCRIPTIONS):
                    await asyncio.sleep(CONFIG.get('ASYNC_BACKOFF_FACTOR', 1))
                    continue
            except BaseException as err:
                return ClientResponse(error=f"Unknown error: {err}")
            else:
                return response
        return ClientResponse(error=f"Client has reached the maximum number of {CONFIG.get('ASYNC_RETRY_COUNT', 5)} retries")

    return _retry_function


class AsyncioClient:
    REQUEST_KWARGS = {
        "timeout": CONFIG.get('ASYNC_TIMEOUT'),
        "data": None,
        "json": None,
        "content_type": "application/json",
        "accept": "application/json",
    }

    def __init__(self, authorization):
        self.auth_header = authorization

    async def create_session(self, limit: int):
        self.my_conn = aiohttp.TCPConnector(limit=limit)
        self.session = aiohttp.ClientSession(connector=self.my_conn)

    async def close_session(self):
        await self.session.close()
        await self.my_conn.close()

    async def with_session(self, limit: int = CONFIG.get('ASYNC_TCP_LIMIT', 25), func: Callable = None):
        await self.create_session(limit)
        result = await func()
        await self.close_session()

        return result

    @retry
    async def _request(self, method, path, **requested_kwargs):
        kwargs = {
            "method": method,
            "url": path,
            "headers": {
                "authorization": self.auth_header,
                "content-type": requested_kwargs["content_type"],
                "accept": requested_kwargs["accept"],
            },
            "timeout": requested_kwargs["timeout"],
            "data": requested_kwargs["data"],
            "json": requested_kwargs["json"],
            "raise_for_status": True,
        }
        try:
            async with self.session.request(**kwargs) as r:
                result = ClientResponse(response=await r.json(), status_code=r.status)
        except aiohttp.ClientResponseError as err:
            return ClientResponse(error=f"Client Response error: {err}")
        except aiohttp.ClientConnectorError as err:
            return ClientResponse(error=f"Client Connector Error: {err}")
        except aiohttp.ServerDisconnectedError as err:
            return ClientResponse(error=f"Server Disconnected Error: {err}")
        except asyncio.TimeoutError as err:
            return ClientResponse(error=f"Client Timeout Error: {err}")
        except BaseException as err:
            return ClientResponse(error=f"Unknown error: {err}")

        return result

    def post(self, path, **requested_kwargs):
        return self._request("POST", path, **(AsyncioClient.REQUEST_KWARGS | requested_kwargs))

    def put(self, path, **requested_kwargs):
        return self._request("PUT", path, **(AsyncioClient.REQUEST_KWARGS | requested_kwargs))

    def get(self, path, **requested_kwargs):
        return self._request("GET", path, **(AsyncioClient.REQUEST_KWARGS | requested_kwargs))

    def delete(self, path, **requested_kwargs):
        return self._request("DELETE", path, **(AsyncioClient.REQUEST_KWARGS | requested_kwargs))
