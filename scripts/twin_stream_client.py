"""
twin_stream_client.py
---------------------
Reference client for the UE5 stream (/ws/field). Use it to check what UE5
will receive, without UE5:

    python digital_twin_api.py                       # terminal 1
    python scripts/twin_stream_client.py --heatmap   # terminal 2
    curl -X POST localhost:8008/demo/start -H 'content-type: application/json' -d '{}'

Prints one line per message and, with --heatmap, an ASCII stress map of the
20 x 20 grid (row 0 = north edge, as in the UE5 HISM grid).
"""

from __future__ import annotations

import argparse
import asyncio
import json

SHADES = " .:-=+*#%@"


def heatmap(values, rows, cols) -> str:
    lines = []
    for r in range(rows):
        row = values[r * cols:(r + 1) * cols]
        lines.append("".join(SHADES[min(len(SHADES) - 1, int(v * len(SHADES)))] * 2 for v in row))
    return "\n".join(lines)


def describe(msg: dict, show_heatmap: bool) -> str:
    if msg["type"] == "field.snapshot":
        n = len(msg["stress"])
        mean = lambda k: sum(msg[k]) / n  # noqa: E731
        out = (f"[snapshot #{msg['seq']}] day {msg['sim_day']} {msg['growth_stage']} "
               f"(GDD {msg['gdd']}) ndvi {mean('ndvi'):.3f} stress {mean('stress'):.3f} "
               f"fungus {mean('fungus'):.3f} water {mean('water_mm'):.0f} mm "
               f"anomalies {len(msg['anomalies'])} pending sprays {msg['pending_sprays']}")
        if show_heatmap:
            out += "\n" + heatmap(msg["stress"], msg["grid"]["rows"], msg["grid"]["cols"])
        return out
    if msg["type"] == "twin.status":
        err = msg.get("error") or {}
        e = f"  [twin vs truth RMSE: fungus {err['fungus']:.3f}]" if err else ""
        return f"[status:{msg['phase']}] {msg['text']}{e}"
    if msg["type"] == "swarm.positions":
        parts = [f"{d} r{p[0]:.1f} c{p[1]:.1f} {p[2]:.0f}m{' SPRAY' if p[3] else ''}"
                 for d, p in msg["drones"].items()]
        return f"[drones t={msg['t_s']:.0f}s] " + " | ".join(parts)
    if msg["type"] == "swarm.track":
        parts = [f"{d}: {len(p)} samples, {sum(s[4] for s in p)} spraying"
                 for d, p in msg["drones"].items()]
        return f"[track] {msg['kind']} mission, day {msg['sim_day']}: " + "; ".join(parts)
    return f"[{msg['type']}] {json.dumps(msg)[:200]}"


async def main(url: str, show_heatmap: bool, count: int) -> None:
    import websockets

    async with websockets.connect(url, max_size=2 ** 22) as ws:
        print(f"connected to {url}")
        n, last_pos = 0, 0.0
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "swarm.positions":   # 10 Hz: print about once a second
                now = asyncio.get_running_loop().time()
                if now - last_pos < 1.0:
                    continue
                last_pos = now
            print(describe(msg, show_heatmap), flush=True)
            n += 1
            if count and n >= count:
                break


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8008/ws/field")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--count", type=int, default=0, help="exit after N messages (0 = run forever)")
    args = ap.parse_args()
    asyncio.run(main(args.url, args.heatmap, args.count))
