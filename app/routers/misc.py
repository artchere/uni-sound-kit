import asyncio
from asyncio import Future
from fastapi import APIRouter
from fastapi.routing import APIRoute
from typing import Optional, List, Union, Any, Coroutine
from starlette.responses import PlainTextResponse, RedirectResponse


class Startup:
    """Class that handles /startup endpoint."""
    status: bool = True

    @classmethod
    async def endpoint(cls):
        """Server startup."""
        status_code: int = 200 if cls.status else 425
        return PlainTextResponse(content=str(status_code), status_code=status_code)


class Readiness:
    """Class that handles /readiness endpoint."""
    urls: Optional[List[str]] = None
    tasks: Optional[List[Union[Future, Coroutine]]] = None
    logger: Any = None
    client: Any = None

    status: bool = False

    def __init__(
        self,
        urls: List[str],
        tasks: List[Union[Future, Coroutine]],
        logger: Any,
        client: Any = None,
    ) -> None:
        """
        :param urls: list of service urls to check.
        :param tasks: list of futures or coroutines
        :param logger: Logger object.
        :param client: HTTPClient object.
        """
        Readiness.urls = urls or []
        Readiness.tasks = tasks or []
        Readiness.logger = logger
        Readiness.client = client

        Readiness.status = False

    @classmethod
    async def _make_request(cls, url: str) -> None:
        """Check readiness of the specified service."""
        while True:
            cls.logger.info(
                f"Trying to connect to '{url}'",
            )
            try:
                response = await cls.client.get(url=f"{url}", timeout=30)
                if response.status_code == 200:
                    cls.logger.info(
                        f"Successfully connected to '{url}'",
                    )
                    break

                cls.logger.warning(
                    f"Failed to connect to '{url}'",
                )
            except Exception as e:
                cls.logger.warning(
                    f"Failed to connect to '{url}': {str(e)}",
                )

            await asyncio.sleep(10)

    @classmethod
    async def _check_readiness(cls) -> None:
        """Check readiness of all services."""
        cls.logger.info(
            f"Running readiness checks.",
        )
        await asyncio.gather(*(cls._make_request(url) for url in cls.urls or []))

        for task in cls.tasks or []:
            try:
                await task
            except Exception as e:
                cls.logger.info(e)

        cls.logger.info(
            f"Successfully finished readiness checks.",
        )

        cls.status = True

    @classmethod
    def run(cls) -> None:
        """Create an asyncio task to check all passed urls."""
        loop = asyncio.get_event_loop()
        loop.create_task(cls._check_readiness())

    @classmethod
    async def endpoint(cls):
        """Server readiness."""
        status_code: int = 200 if (cls.status and Startup.status) else 425

        return PlainTextResponse(content=str(status_code), status_code=status_code)


async def health_check():
    """Server status."""
    status_code: int = 200
    return PlainTextResponse(content=str(status_code), status_code=status_code)


async def redirect_doc():
    """Documentation redirection."""
    return RedirectResponse("/doc")


class ServiceReadiness(Readiness):
    @classmethod
    async def _make_request(cls, url: str) -> None:
        """Check readiness of the specified service."""
        while True:
            cls.logger.info(
                f"Trying to connect to '{url}'",
            )

            try:
                async with cls.client as client:
                    response = await client.get(url=url)
                    if response.status_code == 200:
                        cls.logger.info(
                            f"Successfully connected to '{url}'",
                        )
                    else:
                        cls.logger.info(
                            f"Failed to connect to '{url}'",
                        )
            except Exception as e:
                cls.logger.info(e)
            await asyncio.sleep(10)

    @classmethod
    async def _check_readiness(cls) -> None:
        """Check readiness of all services."""
        cls.logger.info(
            f"Running readiness checks.",
        )
        await asyncio.gather(*(cls._make_request(url) for url in cls.urls or []))

        for task in cls.tasks or []:
            await task

        cls.logger.info(
            f"Successfully finished readiness checks.",
        )

        cls.status = True

    @classmethod
    def run(cls) -> None:
        """Create an asyncio task to check all passed urls."""
        loop = asyncio.get_event_loop()
        loop.create_task(cls._check_readiness())


class Liveness:
    """Class that handles /liveness endpoint."""
    @classmethod
    async def endpoint(cls):
        """Server liveness."""
        status_code: int = 200 if (Startup.status and Readiness.status) else 425

        return PlainTextResponse(content=str(status_code), status_code=status_code)


router = APIRouter(
    routes=[
        APIRoute(path="/health_check", endpoint=health_check, tags=["Misc"]),
        APIRoute(path="/startup", endpoint=Startup.endpoint, tags=["Misc"]),
        APIRoute(path="/readiness", endpoint=Readiness.endpoint, tags=["Misc"]),
        APIRoute(path="/liveness", endpoint=Liveness.endpoint, tags=["Misc"]),
        APIRoute(path="/", endpoint=redirect_doc, tags=["Misc"])
    ],
)
