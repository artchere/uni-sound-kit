import os
import rel
import sys
import yaml
import asyncio
import argparse
from typing import Union
from websocket import ABNF
from websocket import create_connection

sys.path.append(os.getcwd())
from app.base.io_stream import stream_mic_async, FileStream


class WebsocketClient:
    @staticmethod
    def get_stream(cfg: Union[str, dict] = None, path_to_audio: str = None):
        if isinstance(cfg, dict):
            config = cfg
        else:
            with open(cfg, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)

        if path_to_audio is None:
            stream = stream_mic_async(config, return_bytes=True)
            print("Listening from mic...")
        else:
            stream = FileStream(config, path_to_audio)

        return stream

    async def run(self, args):
        stream = self.get_stream(args)
        ws = create_connection(args.address,
                               skip_utf8_validation=True,
                               enable_multithread=False,
                               suppress_origin=True)

        async for frames in stream:
            ws.send(frames, opcode=ABNF.OPCODE_BINARY)
            response = eval(ws.recv_data()[-1].decode())

            if "results" in response:
                print(response["results"])

        rel.signal(2, rel.abort)  # Keyboard Interrupt
        rel.dispatch()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        help="Use input file or microphone",
        default=None
    )

    parser.add_argument(
        "--address",
        help="Connection address",
        default="ws://127.0.0.1:5000/ws"  # TODO: to .env, config or docker?
    )

    args = parser.parse_args()

    client = WebsocketClient()
    task = client.run(args)

    try:
        asyncio.get_event_loop().run_until_complete(task)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

