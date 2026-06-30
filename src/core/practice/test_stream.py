import time

import httpx


URL = "http://127.0.0.1:8000/llm/stream_chat"


def main() -> None:
    payload = {
        "message": (
            "this is a test"
            "hello world"
        )
        
    }

    start = time.perf_counter()
    with httpx.stream("POST", URL, json=payload, timeout=120) as response:
        print("status:", response.status_code)
        print("content-type:", response.headers.get("content-type"))
        response.raise_for_status()

        for chunk in response.iter_text():
            if not chunk:
                continue

            elapsed = time.perf_counter() - start
            #print(f"\n[{elapsed:0.2f}s chunk]")
            print(chunk, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
