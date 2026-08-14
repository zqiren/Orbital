import asyncio, os, sys, logging
sys.path.insert(0, "/Users/keanezhou/Desktop/orbital-test")
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
S = "/private/tmp/claude-501/-Users-keanezhou-Desktop-orbital-test/99fe8248-cd0f-4398-bb85-d36f7c99d161/scratchpad"
WS = f"{S}/ws2"; os.makedirs(WS, exist_ok=True)
CMD, ARGV = "/bin/sh", ["-c", f"cd {S}/acptest && exec ./node_modules/.bin/dsh-acp-demo"]

async def main():
    t = ACPSDKTransport(permission_mode="auto")
    await asyncio.wait_for(t.start(CMD, ARGV, WS, env=dict(os.environ)), timeout=90)
    sid = t.session_id
    print("session:", sid)
    r1 = await asyncio.wait_for(t.send("Remember the number 8317. Reply with just OK."), timeout=240)
    print("turn1:", repr(r1))
    r2 = await asyncio.wait_for(t.send("What number did I ask you to remember?"), timeout=240)
    print("turn2:", repr(r2))
    await t.stop()
    print("MULTI-TURN MEMORY:", "PASS" if "8317" in (r2 or "") else "FAIL")

asyncio.run(main())
