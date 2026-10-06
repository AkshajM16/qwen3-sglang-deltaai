import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "Qwen/Qwen3-4B"

PROMPTS = [
    "Explain KV caching in LLM inference in two concise sentences.",

    "Calculate 37 x 19. Briefly explain how you got the answer.",

    "Write a Python function that removes duplicates from a list while preserving the original order.",

    "Explain the difference between a process and a thread to a computer science student in 4-5 sentences.",

    (
        "An inference server handles requests individually at 3 requests/second, "
        "but with batching handles 10 requests/second. "
        "What is the approximate throughput improvement factor?"
    ),

    (
        "Explain continuous batching in an LLM inference server "
        "and why it can improve GPU utilization."
    ),

    (
        "Find the bug in this Python code and explain the fix:\n"
        "numbers = [1, 2, 3]\n"
        "for i in range(len(numbers) + 1):\n"
        "    print(numbers[i])"
    ),

    (
        "In three sentences, compare supervised fine-tuning "
        "and reinforcement learning for training an LLM."
    ),
]


def query(
    prompt,
    thinking,
    max_tokens,
    temperature,
    top_p,
    top_k,
):
    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "chat_template_kwargs": {
                "enable_thinking": thinking
            },
        },
        timeout=180,
    )
    
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]


def main():
    parser = argparse.ArgumentParser(
        description="Send a batch of concurrent prompts to an SGLang server."
    )

    parser.add_argument(
        "-n",
        "--num-prompts",
        type=int,
        default=4,
        help=f"Number of prompts to send concurrently (1-{len(PROMPTS)})",
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of generated tokens. "
            "Defaults to 512 normally and 2048 with --thinking."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling value (default: 0.8)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling value (default: 20)",
    )

    args = parser.parse_args()

    if args.num_prompts < 1 or args.num_prompts > len(PROMPTS):
        parser.error(
            f"--num-prompts must be between 1 and {len(PROMPTS)}"
        )

    if args.max_tokens is None:
        max_tokens = 2048 if args.thinking else 512
    else:
        max_tokens = args.max_tokens

    prompts = PROMPTS[:args.num_prompts]

    print(
        f"Sending {len(prompts)} concurrent requests "
        f"to {MODEL} at {URL}"
    )

    print(
        f"Thinking: {args.thinking} | "
        f"max_tokens: {max_tokens} | "
        f"temperature: {args.temperature} | "
        f"top_p: {args.top_p} | "
        f"top_k: {args.top_k}"
    )

    with ThreadPoolExecutor(max_workers=args.num_prompts) as executor:
        futures = {
            executor.submit(
                query,
                prompt,
                args.thinking,
                max_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
            ): i
            for i, prompt in enumerate(prompts, start=1)
        }

        for future in as_completed(futures):
            i = futures[future]
            prompt = prompts[i - 1]

            print(f"\n--- Response {i} ---")
            print("Prompt:")
            print(prompt)

            try:
                message = future.result()

                reasoning = message.get("reasoning_content")
                content = message.get("content")

                if reasoning:
                    print("\nReasoning:")
                    print(reasoning)

                print("\nAnswer:")
                print(content)

            except Exception as e:
                print("\nRequest failed:")
                print(e)


if __name__ == "__main__":
    main()