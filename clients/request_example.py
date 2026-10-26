"""Example client for a running SWOT-Bot server (python -m swotbot serve)."""

import requests

BASE = "http://127.0.0.1:8088"
TEXTS = [
    "As informed, the Group has competitive edge over peers due to its robust customer support service.",
    "Manpower attrition risk along with increasing salary cost.",
    "The group has forayed into multiple new countries through wholly owned subsidiaries.",
    "Indian automobile industry is currently in a slowdown, with demand growing at the slowest rate in a decade.",
]

if __name__ == "__main__":
    for item in requests.post(f"{BASE}/predict", json={"texts": TEXTS}, timeout=30).json():
        print(f"{item['label']:13s} {item['confidence']:6.1%}  {item['text'][:70]}")

    # Same call the Word add-in makes.
    print(requests.post(f"{BASE}/api/", params={"TEXT": f"['{TEXTS[1]}']"}, timeout=30).json())
