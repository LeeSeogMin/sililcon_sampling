from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from clova_client import clova_chat_completions_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="CLOVA Studio Chat Completions v3 smoke test")
    parser.add_argument("--model", type=str, default=None, help="Override CLOVA model (or use CLOVA_STUDIO_MODEL)")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--json", dest="print_json", action="store_true", help="Print full JSON response")
    args = parser.parse_args()

    load_dotenv()

    resp = clova_chat_completions_v3(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.print_json:
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        return

    text = None
    try:
        # Best-effort extraction (response schema may vary by product/version)
        candidates = resp.get("choices") or []
        if candidates:
            msg = candidates[0].get("message") or {}
            text = msg.get("content")
    except Exception:
        text = None

    if not text:
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        return
    print(text)


if __name__ == "__main__":
    main()

