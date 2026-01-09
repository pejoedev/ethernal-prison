#!/usr/bin/env python3
"""
Autonomous AI Agent Loop with Sandboxed File/Command Execution
Uses LM Studio's OpenAI-compatible endpoints for local model queries
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests


class AgentConfig:
    """Load and manage agent configuration from settings.json"""

    def __init__(self, config_path: str = "settings.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load settings.json with defaults"""
        if not self.config_path.exists():
            default = {
                "api_endpoint": "http://localhost:1234/v1",
                "model": "openai/gpt-oss-20b",
                "root_directory": ".",
                "max_file_size_mb": 10,
                "temperature": 0.7,
                "max_tokens": 2000,
                "allowed_commands": {
                    "nest": "nest generate module {}",
                    "test": "npm test -- {}",
                    "lint": "npm run lint -- {}",
                    "build": "npm run build",
                    "git_add": "git add {}",
                    "git_commit": "git commit -m '{}'",
                    "git_status": "git status",
                },
                "allowed_command_prefixes": [
                    "npm",
                    "git",
                    "npx",
                ],
                "forbidden_patterns": [
                    "rm -rf /",
                    "sudo",
                    "||",
                    "&&",
                    ";",
                    "`",
                    "$(",
                ],
            }
            self._save_config(default)
            return default
        return json.loads(self.config_path.read_text())

    def _save_config(self, config: dict):
        """Save config to file"""
        self.config_path.write_text(json.dumps(config, indent=2))

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        return self.config.get(key, default)


class FileManager:
    """Secure file operations within root directory"""

    def __init__(self, root: str, max_size_mb: int = 10):
        self.root = Path(root).resolve()
        self.max_bytes = max_size_mb * 1024 * 1024

    def _sanitize_path(self, filepath: str) -> str:
        """Remove dangerous path components"""
        # Remove leading slashes
        filepath = filepath.lstrip("/")
        # Remove ./ and ../
        filepath = filepath.replace("./", "").replace("../", "")
        # Remove standalone . and ..
        filepath = filepath.replace("\\", "/")
        parts = filepath.split("/")
        parts = [p for p in parts if p and p != "." and p != ".."]
        return "/".join(parts) if parts else "."

    def _validate_path(self, filepath: str) -> Path:
        """Ensure path is within root directory"""
        # Sanitize the path first
        sanitized = self._sanitize_path(filepath)
        target = (self.root / sanitized).resolve()

        # Final check: ensure target is within root
        if not str(target).startswith(str(self.root)):
            raise ValueError(
                f"Path escape attempt detected: {filepath} "
                f"(sanitized to {sanitized})"
            )
        return target

    def read_file(self, filepath: str) -> str:
        """Read file content"""
        try:
            path = self._validate_path(filepath)
        except ValueError as e:
            return f"Error: {e}"

        if not path.exists():
            return f"Error: File not found - {filepath}"
        if not path.is_file():
            return f"Error: Not a file - {filepath}"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, filepath: str, content: str) -> str:
        """Write content to file"""
        try:
            path = self._validate_path(filepath)
        except ValueError as e:
            return f"Error: {e}"

        if len(content.encode()) > self.max_bytes:
            return f"Error: File too large (max {self.max_bytes} bytes)"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"File written: {filepath}"
        except Exception as e:
            return f"Error writing file: {e}"

    def show_directory(self, dirpath: str = ".") -> str:
        """Show directory contents with details"""
        try:
            path = self._validate_path(dirpath)
        except ValueError as e:
            return f"Error: {e}"

        if not path.exists():
            return f"Error: Directory not found - {dirpath}"
        if not path.is_dir():
            return f"Error: Not a directory - {dirpath}"

        try:
            items = sorted(path.iterdir())
            if not items:
                return f"[empty directory]"

            # Build formatted listing with file sizes
            listing = []
            for item in items:
                if item.is_dir():
                    listing.append(f"[DIR]  {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    listing.append(f"[FILE] {item.name} ({size_str})")

            return "\n".join(listing)
        except Exception as e:
            return f"Error listing directory: {e}"

    def list_directory(self, dirpath: str = ".") -> str:
        """List directory contents (simple format)"""
        try:
            path = self._validate_path(dirpath)
        except ValueError as e:
            return f"Error: {e}"

        if not path.is_dir():
            return f"Error: Not a directory - {dirpath}"
        try:
            items = sorted(path.iterdir())
            listing = "\n".join(
                f"{'[DIR] ' if item.is_dir() else ''}{item.name}"
                for item in items
            )
            return listing if listing else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"


class CommandExecutor:
    """Execute whitelisted commands safely"""

    def __init__(self, config: AgentConfig, root: str):
        self.config = config
        self.root = Path(root).resolve()
        self.forbidden = config.get("forbidden_patterns", [])
        self.custom_commands = config.get("allowed_commands", {})
        self.allowed_prefixes = config.get(
            "allowed_command_prefixes", []
        )

    def _check_forbidden_patterns(self, command: str) -> bool:
        """Check if command contains forbidden patterns"""
        for pattern in self.forbidden:
            if pattern in command:
                return True
        return False

    def _check_allowed_prefix(self, command: str) -> bool:
        """Check if command starts with allowed prefix"""
        command_lower = command.lower().strip()
        for prefix in self.allowed_prefixes:
            if command_lower.startswith(prefix):
                # Ensure it's a complete command word
                # (e.g., "npm" matches "npm install" but not "npmd")
                remainder = command_lower[len(prefix) :]
                if not remainder or remainder[0] in (" ", "\t"):
                    return True
        return False

    def execute(self, command: str) -> str:
        """Execute command with safety checks"""
        if not command or not command.strip():
            return "Error: Empty command"

        # Check for forbidden patterns
        if self._check_forbidden_patterns(command):
            return (
                "Error: Forbidden pattern detected. "
                "Command chaining (&&, ||, ;) and subshells "
                "(``, $()) are not allowed."
            )

        # Check if it's a custom command (explicit aliases)
        for alias, template in self.custom_commands.items():
            if command.startswith(alias):
                remainder = command[len(alias) :].strip()
                # Only match if it's a complete command word
                if not command[len(alias) :] or command[len(alias)] in (
                    " ",
                    "\t",
                ):
                    command = (
                        template.format(remainder)
                        if remainder
                        else template
                    )
                    break

        # Check if it's an allowed prefix command
        if not self._check_allowed_prefix(command):
            # Not a custom command and not an allowed prefix
            available = ", ".join(
                list(self.custom_commands.keys())
                + self.allowed_prefixes
            )
            return (
                f"Error: Command not allowed. "
                f"Available: {available}"
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            return output[:2000] if output else "(command completed)"
        except subprocess.TimeoutExpired:
            return "Error: Command timeout (60s)"
        except Exception as e:
            return f"Error executing command: {e}"


class LMStudioClient:
    """Client for LM Studio's OpenAI-compatible API"""

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """Send chat completion request"""
        try:
            response = requests.post(
                f"{self.endpoint}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to reach LM Studio - {e}")
            print(
                "Make sure LM Studio is running. "
                "Start it with: lms server start"
            )
            return None


class AgentLoop:
    """Main autonomous agent loop"""

    def __init__(self, config_path: str = "settings.json"):
        self.config = AgentConfig(config_path)
        self.files = FileManager(
            self.config.get("root_directory", "."),
            self.config.get("max_file_size_mb", 10),
        )
        self.executor = CommandExecutor(self.config, ".")
        self.client = LMStudioClient(
            self.config.get("api_endpoint", "http://localhost:1234/v1"),
            self.config.get("model", "openai/gpt-oss-20b"),
        )
        self.history_file = Path(".agent_history")
        self.last_message = self._load_last_message()

    def _load_last_message(self) -> Optional[str]:
        """Load last message from history file"""
        if self.history_file.exists():
            return self.history_file.read_text().strip()
        return None

    def _save_message(self, message: str):
        """Save current message to history file"""
        self.history_file.write_text(message)

    def _build_context(self) -> str:
        """Build context string with current state"""
        context = "=== AGENT CONTEXT ===\n\n"

        # Add last message if available
        if self.last_message:
            context += f"Last message:\n{self.last_message}\n\n"

        # Add README
        readme = self.files.read_file("README.md")
        if not readme.startswith("Error"):
            context += f"README.md:\n{readme}\n\n"

        # Add TO DO
        todo = self.files.read_file("TODO.md")
        if not todo.startswith("Error"):
            context += f"TODO.md:\n{todo}\n\n"

        # Add directory listing
        context += f"Current directory:\n{self.files.show_directory()}\n"

        return context

    def _parse_action(self, response: str) -> dict:
        """Parse JSON action from response"""
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        return {}

    def _format_help(self) -> str:
        """Generate help message"""
        custom_cmds = self.config.get("allowed_commands", {})
        prefixes = self.config.get("allowed_command_prefixes", [])

        help_text = """
=== AGENT HELP ===
You are an autonomous agent with access to file and command execution.
Always respond with valid JSON containing an 'action' field.
If you do not know what to do, respond with {"action": "help"}.

AVAILABLE ACTIONS:
1. read_file
   {"action": "read_file", "path": "path/to/file"}

2. write_file
   {"action": "write_file", "path": "path/to/file", "content": "..."}

3. show_dir
   {"action": "show_dir", "path": "." (default: current dir)}
   Shows directory contents with file sizes and types.

4. list_dir
   {"action": "list_dir", "path": "." (default: current dir)}
   Shows simple directory listing.

5. execute
   {"action": "execute", "command": "command to run"}

6. think
   {"action": "think", "message": "your reasoning"}

7. help
   {"action": "help"}

CUSTOM COMMAND ALIASES:
"""
        for alias, template in custom_cmds.items():
            help_text += f"  {alias}: {template}\n"

        help_text += f"\nALLOWED COMMAND PREFIXES:\n"
        for prefix in prefixes:
            help_text += f"  {prefix} (e.g., {prefix} install)\n"

        help_text += """
SECURITY NOTES:
- All paths are sandboxed to the root directory
- Path sanitization: /, ./, and ../ are automatically removed
- Command chaining (&&, ||, ;) is forbidden
- Shell subshells (``, $()) are forbidden
- Only npm, git, npx, and custom commands are allowed

EXAMPLE PATHS (all safely resolve within root):
  "src/index.js"     -> ./src/index.js
  "/src/index.js"    -> ./src/index.js (leading / removed)
  "src/../index.js"  -> ./src/index.js (../ removed)
  "./src/index.js"   -> ./src/index.js (./ removed)
"""
        return help_text

    def run(self):
        """Main agent loop"""
        print("Autonomous AI Agent Loop")
        print("Press Ctrl+C to exit\n")

        context = self._build_context()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous AI agent working on a project. "
                    "Respond only with valid JSON containing an 'action' "
                    "and parameters. Always include your reasoning in a "
                    "'reasoning' field. If you do not know what to do, "
                    "respond with {\"action\": \"help\"}."
                ),
            },
            {"role": "user", "content": context},
        ]

        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")

                # Get response from LM Studio
                response = self.client.chat(
                    messages,
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 2000),
                )

                if response is None:
                    print("Failed to get response from LM Studio")
                    break

                print(f"Response:\n{response}\n")
                self._save_message(response)

                # Parse and execute action
                action = self._parse_action(response)

                if not action:
                    # Didn't get JSON, ask for help
                    action = {"action": "help"}

                action_type = action.get("action", "help").lower()

                if action_type == "read_file":
                    result = self.files.read_file(action.get("path", ""))
                elif action_type == "write_file":
                    result = self.files.write_file(
                        action.get("path", ""),
                        action.get("content", ""),
                    )
                elif action_type == "show_dir":
                    result = self.files.show_directory(
                        action.get("path", ".")
                    )
                elif action_type == "list_dir":
                    result = self.files.list_directory(
                        action.get("path", ".")
                    )
                elif action_type == "execute":
                    result = self.executor.execute(
                        action.get("command", "")
                    )
                elif action_type == "think":
                    result = f"Thought: {action.get('message', '')}"
                elif action_type == "help":
                    result = self._format_help()
                else:
                    result = f"Unknown action: {action_type}"

                print(f"Result:\n{result}\n")

                # Add to message history for next iteration
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": result})

                # Keep history reasonable (last 6 exchanges = 12 messages)
                if len(messages) > 14:
                    messages = messages[:2] + messages[-12:]

                time.sleep(1)  # Prevent hammering the API

            except KeyboardInterrupt:
                print("\nExiting agent loop")
                break
            except Exception as e:
                print(f"Error in loop: {e}")
                time.sleep(2)


if __name__ == "__main__":
    agent = AgentLoop()
    agent.run()