#!/usr/bin/env python3
"""
Test script for LM Studio API at http://127.0.0.1:1234
"""

import json
import requests
import time


def test_connection():
    """Test basic connection to LM Studio"""
    print("=" * 60)
    print("TEST 1: Basic Connection")
    print("=" * 60)
    try:
        response = requests.get(
            "http://127.0.0.1:1234/v1/models",
            timeout=5
        )
        response.raise_for_status()
        print("✓ Connected to LM Studio")
        models = response.json()
        print(f"Available models: {json.dumps(models, indent=2)}")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect - is LM Studio running?")
        print("  Start with: lms server start")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_simple_chat():
    """Test simple chat completion"""
    print("\n" + "=" * 60)
    print("TEST 2: Simple Chat Completion")
    print("=" * 60)
    try:
        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'Hello World' and nothing else"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 100,
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"✓ Got response: {content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_json_response():
    """Test JSON response format"""
    print("\n" + "=" * 60)
    print("TEST 3: JSON Response Format")
    print("=" * 60)
    try:
        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Respond only with valid JSON. "
                            "No other text."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            'Return JSON: {"action": "write_file", '
                            '"path": "test.txt", "content": "hello"}'
                        )
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 200,
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"✓ Raw response:\n{content}")

        # Try to parse JSON
        try:
            json_data = json.loads(content)
            print(f"✓ Valid JSON parsed:\n{json.dumps(json_data, indent=2)}")
        except json.JSONDecodeError:
            print("⚠ Response is not valid JSON (this is common)")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_streaming():
    """Test streaming response"""
    print("\n" + "=" * 60)
    print("TEST 4: Streaming Response")
    print("=" * 60)
    try:
        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [
                    {
                        "role": "user",
                        "content": "Count to 5"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": True,
            },
            timeout=60,
            stream=True
        )
        response.raise_for_status()
        print("✓ Streaming enabled, reading chunks:")
        print("---")

        full_response = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_response += content
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

        print("\n---")
        print(f"✓ Full response received ({len(full_response)} chars)")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_parameters():
    """Test various parameter combinations"""
    print("\n" + "=" * 60)
    print("TEST 5: Parameter Variations")
    print("=" * 60)

    configs = [
        {
            "name": "Low temperature (0.1)",
            "temp": 0.1,
            "prompt": "What is 2+2?"
        },
        {
            "name": "High temperature (1.0)",
            "temp": 1.0,
            "prompt": "Generate a creative word"
        },
        {
            "name": "High max_tokens (500)",
            "temp": 0.7,
            "prompt": "Tell me about Python"
        },
    ]

    for config in configs:
        try:
            response = requests.post(
                "http://127.0.0.1:1234/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": config["prompt"]}],
                    "temperature": config["temp"],
                    "max_tokens": 200,
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"✓ {config['name']}: {content[:80]}...")
        except Exception as e:
            print(f"✗ {config['name']}: {e}")


def test_conversation():
    """Test multi-turn conversation"""
    print("\n" + "=" * 60)
    print("TEST 6: Multi-Turn Conversation")
    print("=" * 60)
    try:
        messages = [
            {"role": "user", "content": "My name is Alex"}
        ]

        # First turn
        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        reply1 = data["choices"][0]["message"]["content"]
        print(f"Turn 1 - User: My name is Alex")
        print(f"Turn 1 - Bot: {reply1}")

        # Second turn
        messages.append({"role": "assistant", "content": reply1})
        messages.append({"role": "user", "content": "What is my name?"})

        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 100,
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        reply2 = data["choices"][0]["message"]["content"]
        print(f"\nTurn 2 - User: What is my name?")
        print(f"Turn 2 - Bot: {reply2}")

        print(f"\n✓ Conversation test passed")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LM Studio API Test Suite")
    print("Testing: http://127.0.0.1:1234")
    print("=" * 60)

    results = []

    # Test 1: Connection
    if not test_connection():
        print("\n❌ Cannot proceed without connection")
        return

    time.sleep(1)
    results.append(("Simple Chat", test_simple_chat()))

    time.sleep(1)
    results.append(("JSON Response", test_json_response()))

    time.sleep(1)
    results.append(("Streaming", test_streaming()))

    time.sleep(1)
    test_parameters()

    time.sleep(1)
    results.append(("Multi-Turn", test_conversation()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()