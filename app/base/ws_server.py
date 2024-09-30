import os
import sys
import json
import yaml
import logging
import asyncio
from enum import Enum
from typing import Union
from aiohttp import web
from aiohttp_middlewares.cors import cors_middleware

import torch
import statistics
import numpy as np

sys.path.append(os.getcwd())


class StatusCode(Enum):
    OK = 0
    INVALID_ARGUMENT = 1
    UNAUTHENTICATED = 2
    INTERNAL = 3


class SessionError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message


class Server:
    def __init__(self, logger=None, session_limit=0):
        self.logger = logger
        self.dbpool = None

        self.session_count, self.session_limit = 0, session_limit

    @staticmethod
    def request_remote_info(request):
        remote = "---"

        if request.transport is not None:
            remote = "%s:%s" % request.transport.get_extra_info("peername")

        return remote

    def log_user(self, remote, msg, user=None, session_data=None):
        username = "---" if user is None else user.name
        session = "---" if session_data is None else session_data.session.session_id

        self.logger.info("%s | %s:%s | %s", remote, username, session, msg)

    def get_middlewares(self):
        middlewares = [self.error_middleware]

        if self.session_limit > 0:
            middlewares.insert(0, self.session_limit_middleware)

        return middlewares

    @web.middleware
    async def session_limit_middleware(self, request, handler):
        if self.session_count >= self.session_limit:
            return web.json_response(
                {"response_code": StatusCode.INTERNAL.value, "msg": "Session count limit!"}, status=200
            )

        self.session_count += 1

        try:
            return await handler(request)

        finally:
            self.session_count -= 1

    @web.middleware
    async def error_middleware(self, request, handler):
        remote = self.request_remote_info(request)

        try:
            return await handler(request)

        except SessionError as e:
            self.log_user(remote, f"CONNECTION ABORTING | Status: {e.code}, message: {e.message}")
            return web.json_response({"response_code": e.code.value, "msg": e.message}, status=200)

        except (web.HTTPNotFound, web.HTTPFound, web.HTTPForbidden):
            raise

        except Exception as e:
            self.log_user(remote, f"CONNECTION TERMINATING ABNORMALLY")
            self.logger.exception(e)

            return web.json_response({"response_code": StatusCode.INTERNAL.value, "msg": str(e)}, status=200)

    @staticmethod
    async def parse_json_config(request, ws):
        try:
            return await (ws.receive_json() if request is None else request.json())

        except (TypeError, json.JSONDecodeError):
            raise SessionError(code=StatusCode.INVALID_ARGUMENT, message="Configuration parsing failure.")


class KWSServer(Server):
    def __init__(self, cfg: Union[str, dict] = None, logger: logging.Logger = None):
        super(KWSServer, self).__init__()

        if isinstance(cfg, dict):
            config = cfg
        else:
            with open(cfg, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.logger = logger

        self.device = torch.device(config["general"]["device"] if torch.cuda.is_available() else "cpu")

        model_path = config["general"]["inference_model_path"]

        assert os.path.exists(model_path), f"Can`t find the model at the path {model_path}"

        self.nn_model = torch.load(model_path, map_location=self.device).to(self.device)
        self.nn_model.eval()

        if self.logger is not None:
            self.logger.info(f"Model instantiated to: {self.device}")

        self.datatype = np.int16 if config["stream"]["use_int16"] else np.float32
        self.target_sr = config["general"]["sample_rate"]
        self.chunk_samples = int(self.target_sr * config["stream"]["buffer_size"])
        self.window_samples = int(self.target_sr * config["stream"]["window_size"])
        self.kw_target = int(config["stream"]["keyword_label"])
        self.threshold = float(config["stream"]["threshold"])
        self.use_argmax = config["stream"]["use_argmax"]

    async def ping(self, request):
        self.log_user(self.request_remote_info(request), "PING REQUEST")

        return web.json_response({
            "response_code": StatusCode.OK.value,
            "r": "Server is alive"
        })

    @staticmethod
    def relu(x):
        return 0 if x < 0 else x

    async def ws(self, request):
        remote = self.request_remote_info(request)
        self.log_user(remote, f"WS REQUEST")

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        user, session_data = None, None
        t = 0

        try:
            collecting_count = int(self.window_samples / self.chunk_samples)

            audio_buffer = [
                np.zeros(self.chunk_samples, dtype=self.datatype)
                for _ in range(collecting_count)
            ]
            result_buffer = [1 for _ in range(collecting_count - 1)]

            audio_timer = 0

            self.log_user(remote, f"AUTHORIZED: {user}", user=user)

            await ws.send_json({"response": StatusCode.OK.value})

            async for msg in ws:
                eos = False
                if msg.type == web.WSMsgType.BINARY:
                    decoded_audio = np.frombuffer(msg.data, dtype=self.datatype)

                    audio_buffer.append(decoded_audio)
                    audio_timer += decoded_audio.shape[0]

                    if len(audio_buffer) == collecting_count:
                        result = await self.ws_recognize(audio_buffer)
                        audio_buffer.pop(0)
                        result_buffer.append(result)

                        if len(result_buffer) > collecting_count:
                            result_buffer.pop(0)

                        await ws.send_json({"response": StatusCode.OK.value,
                                            "results": statistics.mode(result_buffer)})

                elif msg.type == web.WSMsgType.TEXT:
                    if msg.data == "/EOP" or msg.data == "/EOS":
                        logtext = "EOP COMMAND"

                        if msg.data == "/EOS":
                            logtext = "EOS COMMAND"
                            eos = True

                        self.log_user(remote, logtext, user=user, session_data=session_data)

                        await ws.send_json(
                            {"response": StatusCode.OK.value, "results": [{"result": "FILE TRANSMISSION FINISHED"}]})

                    else:
                        raise SessionError(
                            code=StatusCode.INVALID_ARGUMENT, message=f"Unrecognized command '{msg.data}'!"
                        )

                else:
                    raise SessionError(code=StatusCode.INVALID_ARGUMENT, message="Received invalid ws message!")

                if eos:
                    break

        except SessionError as e:
            self.log_user(
                remote, f"CONNECTION ABORTING | Status: {e.code}, message: {e.message}",
                user=user, session_data=session_data
            )

            if not ws.closed:
                await ws.send_json({"response": e.code.value, "msg": e.message})

        except Exception as e:
            self.log_user(remote, f"CONNECTION TERMINATING ABNORMALLY", user=user, session_data=session_data)
            self.logger.exception(e)

            if not ws.closed:
                await ws.send_json({"response": StatusCode.INTERNAL.value, "msg": str(e)})

        except asyncio.CancelledError:
            if session_data is not None:
                await self.flush_session(session_data)

            raise

        finally:
            if len(result_buffer) > 0:
                await ws.send_json({"response": StatusCode.OK.value,
                                    "results": statistics.mode(result_buffer)})
                result_buffer.clear()
            self.log_user(remote, "WS SESSION CLOSED", user=user, session_data=session_data)

        return ws

    async def ws_recognize(self, data):
        samples = torch.from_numpy(np.concatenate(data))

        with torch.no_grad():
            samples = samples.to(self.device).unsqueeze(0)
            predictions = self.nn_model(samples).detach().cpu().numpy()
            if not self.use_argmax:
                if predictions.squeeze()[self.kw_target] >= self.threshold:
                    prediction = self.kw_target
                elif 0.5 <= predictions.squeeze()[self.kw_target] < self.threshold:
                    prediction = int(np.flip(np.argsort(predictions.squeeze()))[1].item())
                else:
                    prediction = int(predictions.argmax(1).item())
            else:
                prediction = int(predictions.argmax(1))

        return prediction


async def create_app_async(cfg=None, enable_online_asr=True, use_cors=False):
    server = KWSServer(cfg)

    middlewares = [cors_middleware(allow_all=True)] if use_cors else []
    middlewares.extend(server.get_middlewares())

    app = web.Application(middlewares=middlewares, client_max_size=32 * 1024 ** 2)
    server.logger = app.logger

    routes = [
        web.get("/ping", server.ping),
    ]

    if enable_online_asr:
        routes.append(web.get("/ws", server.ws))

    app.add_routes(routes)
    app.logger.info("Running server...")

    return app


async def create_full_app_async():
    return await create_app_async()


def create_app(cfg=None):
    return asyncio.get_event_loop().run_until_complete(create_app_async(cfg=cfg, use_cors=True))


def main(cfg=None):
    web.run_app(create_app(cfg), host="127.0.0.1", port=5000)  # TODO: to .env, config or docker?


if __name__ == "__main__":
    main("config.yaml")
