"""Live end-to-end Path-A turn: real DeepSeek model through Orbital's ACPSDKTransport."""
import asyncio, os, sys, json, glob, logging

sys.path.insert(0, "/Users/keanezhou/Desktop/orbital-test")
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport

SCRATCH = "/private/tmp/claude-501/-Users-keanezhou-Desktop-orbital-test/99fe8248-cd0f-4398-bb85-d36f7c99d161/scratchpad"
WS = f"{SCRATCH}/ws"
os.makedirs(WS, exist_ok=True)
CMD, ARGV = "/bin/sh", ["-c", f"cd {SCRATCH}/acptest && exec ./node_modules/.bin/dsh-acp-demo"]

PROMPT = ("Use your bash tool to run exactly: echo hi > hello.txt  "
          "Then read the file back and tell me its contents. Be brief.")


async def main():
    env = dict(os.environ)
    assert env.get("DEEPSEEK_API_KEY"), "no key in env"

    t = ACPSDKTransport(permission_mode="auto")
    await asyncio.wait_for(t.start(CMD, ARGV, WS, env=env), timeout=90)
    print("session_id:", t.session_id, "| resume_outcome:", t._resume_outcome)

    events = []

    async def drain():
        try:
            async for ev in t.read_stream():
                events.append(ev)
        except Exception as e:
            print("stream ended:", type(e).__name__)

    drain_task = asyncio.create_task(drain())

    print(f"\n>>> PROMPT: {PROMPT}\n")
    try:
        reply = await asyncio.wait_for(t.send(PROMPT), timeout=300)
        print(f"<<< send() RETURNED ({len(reply or '')} chars):\n{reply}\n")
    except Exception as e:
        print(f"!!! send() FAILED: {type(e).__name__}: {e}")

    await asyncio.sleep(1.0)
    drain_task.cancel()

    print(f"=== {len(events)} transport events ===")
    for ev in events:
        d = ev.data if isinstance(ev.data, dict) else {}
        detail = d.get("tool_name") or (str(d.get("text", ""))[:80])
        print(f"  [{ev.event_type}] {detail}")

    print("\n=== workspace after turn ===")
    for f in sorted(glob.glob(f"{WS}/*")):
        print(" ", f, os.path.getsize(f), "bytes")
        if f.endswith(".txt"):
            print("   content:", repr(open(f).read()))

    print("\n=== session log on disk ===")
    for f in sorted(glob.glob(f"{SCRATCH}/acptest/.sessions/**/*", recursive=True)):
        if os.path.isfile(f):
            print(" ", f.replace(SCRATCH, "…"), os.path.getsize(f), "bytes")

    await t.stop()
    print("\nstopped. session_id was:", t.session_id)


asyncio.run(main())
