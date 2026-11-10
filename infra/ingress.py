"""Loopback port forwarder with fixed destinations into the internal network."""
import asyncio
from contextlib import suppress

TARGETS={8080:('app',8080),8081:('llm-gateway',8081),9090:('prometheus',9090),
         9093:('alertmanager',9093),3000:('grafana',3000),3200:('tempo',3200)}

async def pipe(reader,writer):
    try:
        while chunk:=await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        with suppress(OSError,RuntimeError):writer.write_eof()

async def handle(reader,writer,host,port):
    remote=None
    try:
        remote_reader,remote=await asyncio.wait_for(asyncio.open_connection(host,port),3)
        async with asyncio.timeout(300):
            await asyncio.gather(pipe(reader,remote),pipe(remote_reader,writer))
    except (OSError,TimeoutError):
        pass
    finally:
        for stream in (remote,writer):
            if stream is not None:
                stream.close()
                with suppress(OSError):await stream.wait_closed()

async def main():
    servers=[]
    for listen,(host,port) in TARGETS.items():
        server=await asyncio.start_server(lambda r,w,h=host,p=port:handle(r,w,h,p),'0.0.0.0',listen)
        servers.append(server)
    await asyncio.gather(*(s.serve_forever() for s in servers))

if __name__=='__main__':asyncio.run(main())
