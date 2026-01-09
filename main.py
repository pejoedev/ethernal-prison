import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests


class Logger:
    """Log agent actions to daily log files with organized structure"""

    def __init__(self, logs_dir: str = "./logs"):
        self.logs_dir = Path(logs_dir)
        self.actions_dir = self.logs_dir / "actions"
        self.responses_dir = self.logs_dir / "responses"
        self.errors_dir = self.logs_dir / "errors"
        self.reviews_dir = self.logs_dir / "reviews"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self, subdir: Path) -> Path:
        """Get today's log file path"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return subdir / f"{date_str}.log"

    def log_action(self, action_type: str, details: dict, result: str = ""):
        """Log an action with details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "details": details,
            "result_preview": result[:200] if result else "",
        }
        self._write_log(log_entry, self.actions_dir)

    def log_response(self, response: str, reasoning: str = ""):
        """Log AI response and reasoning"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "response": response[:500],
            "reasoning": reasoning[:500],
            "full_length": len(response),
        }
        self._write_log(log_entry, self.responses_dir)

    def log_error(self, error_type: str, message: str, context: dict):
        """Log errors with context"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": message,
            "context": context,
        }
        self._write_log(log_entry, self.errors_dir)

    def log_review(self, original_action: str, review: str):
        """Log command reviews"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "original_action": original_action,
            "review": review,
        }
        self._write_log(log_entry, self.reviews_dir)

    def _write_log(self, entry: dict, subdir: Path):
        """Write log entry to file"""
        try:
            log_file = self._get_log_file(subdir)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Logging error: {e}")


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
                "sandbox_root": "./sandbox",
                "max_file_size_mb": 10,
                "temperature": 0.7,
                "max_tokens": 2000,
                "max_iterations": None,
                "credits_name": "Unknown Developer",
                "context_window": 4096,
                "max_prompt_tokens": 3000,
                "allowed_commands": {
                    "nest": "npx @nestjs/cli@latest new {}",
                    "nest_gen": "nest generate module {}",
                    "react": "npx create-react-app {}",
                    "svelte": "npm create svelte@latest {}",
                    "install": "npm install {}",
                    "dev": "npm run dev",
                    "build": "npm run build",
                    "test": "npm test",
                    "lint": "npm run lint",
                    "git_status": "git status",
                    "git_add": "git add {}",
                    "git_commit": "git commit -m '{}'",
                    "git_branch": "git branch {}",
                    "git_checkout": "git checkout {}",
                    "git_pull": "git pull",
                    "git_push": "git push",
                    "git_log": "git log --oneline -n {}",
                    "git_diff": "git diff {}",
                    "git_merge": "git merge {}",
                    "git_stash": "git stash",
                    "git_stash_pop": "git stash pop",
                },
                "allowed_command_prefixes": ["npm", "git", "npx"],
                "forbidden_patterns": [
                    "rm -rf /",
                    "sudo",
                    "||",
                    "&&",
                    ";",
                    "`",
                    "$(", ],
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
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_size_mb * 1024 * 1024

    def _sanitize_path(self, filepath: str) -> str:
        """Remove dangerous path components"""
        filepath = filepath.lstrip("/")
        filepath = filepath.replace("./", "").replace("../", "")
        filepath = filepath.replace("\\", "/")
        parts = filepath.split("/")
        parts = [p for p in parts if p and p != "." and p != ".."]
        return "/".join(parts) if parts else "."

    def _validate_path(self, filepath: str) -> Path:
        """Ensure path is within root directory"""
        sanitized = self._sanitize_path(filepath)
        target = (self.root / sanitized).resolve()

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
        if not filepath or not filepath.strip():
            return "Error: File path cannot be empty"
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
                return "[empty directory]"

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
                    listing.append(
                        f"[FILE] {item.name} ({size_str})"
                    )

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
                remainder = command_lower[len(prefix) :]
                if not remainder or remainder[0] in (" ", "\t"):
                    return True
        return False

    def execute(self, command: str) -> str:
        """Execute command with safety checks"""
        if not command or not command.strip():
            return "Error: Empty command"

        if self._check_forbidden_patterns(command):
            return (
                "Error: Forbidden pattern detected. "
                "Command chaining (&&, ||, ;) and subshells "
                "(``, $()) are not allowed."
            )

        for alias, template in self.custom_commands.items():
            if command.startswith(alias):
                remainder = command[len(alias) :].strip()
                if (
                    not command[len(alias) :]
                    or command[len(alias)] in (" ", "\t")
                ):
                    command = (
                        template.format(remainder)
                        if remainder
                        else template
                    )
                    break

        if not self._check_allowed_prefix(command):
            available = ", ".join(
                list(self.custom_commands.keys())
                + self.allowed_prefixes
            )
            return (
                f"Error: Command not allowed. Available: {available}"
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
    """Client for LM Studio's OpenAI-compatible API with tool use"""

    def __init__(
        self,
        endpoint: str,
        model: str,
        context_window: int = 4096,
        logger: Optional[Logger] = None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.context_window = context_window
        self.logger = logger
        self.retry_count = 0
        self.max_retries = 3
        self.retry_delay = 5

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation: ~1 token per 4 characters"""
        return len(text) // 4

    def _count_messages_tokens(self, messages: list) -> int:
        """Estimate total tokens in messages"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
        return total

    def _trim_messages(
        self, messages: list, max_prompt_tokens: int
    ) -> list:
        """Trim messages to stay under token limit"""
        tokens = self._count_messages_tokens(messages)

        if tokens <= max_prompt_tokens:
            return messages

        trimmed = messages[:2]
        for msg in messages[2:]:
            new_tokens = self._count_messages_tokens(
                trimmed
            ) + self._estimate_tokens(msg.get("content", ""))
            if new_tokens <= max_prompt_tokens:
                trimmed.append(msg)

        return trimmed

    def get_tools(self) -> list:
        """Define available tools for the AI"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the content of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "show_dir",
                    "description": "Show directory contents with details",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Directory path (default: current "
                                    "directory)"
                                ),
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List directory contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Directory path (default: current "
                                    "directory)"
                                ),
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute",
                    "description": (
                        "Execute whitelisted commands (npm, git, npx)"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command to execute",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "think",
                    "description": "Record your reasoning and thoughts",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Your thoughts",
                            }
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_feedback",
                    "description": (
                        "Submit feedback for improvements"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Feedback text",
                            }
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "request_command",
                    "description": "Request a new command to be added",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command_name": {
                                "type": "string",
                                "description": "Alias for the command",
                            },
                            "command_template": {
                                "type": "string",
                                "description": (
                                    "Command template with {} for args"
                                ),
                            },
                        },
                        "required": [
                            "command_name",
                            "command_template",
                        ],
                    },
                },
            },
        ]

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_prompt_tokens: int = 3000,
    ) -> Optional[dict]:
        """Send chat completion request with tool use support"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        messages = self._trim_messages(messages, max_prompt_tokens)
        self.retry_count = 0

        while self.retry_count < self.max_retries:
            try:
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "tools": self.get_tools(),
                        "tool_choice": "auto",
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                self.retry_count = 0
                return data["choices"][0]["message"]

            except requests.exceptions.HTTPError as e:
                error_msg = str(e)
                context_overflow = (
                    "context length" in error_msg.lower()
                    or "context overflow" in error_msg.lower()
                )

                if context_overflow:
                    print(
                        f"\n[{current_time}] Context overflow detected"
                    )
                    print(
                        "Trimming message history and retrying..."
                    )

                    if self.logger:
                        self.logger.log_error(
                            "context_overflow",
                            "Context window exceeded",
                            {
                                "prompt_tokens": (
                                    self._count_messages_tokens(messages)
                                ),
                                "context_window": self.context_window,
                                "retry_count": self.retry_count,
                            },
                        )

                    max_prompt_tokens = int(max_prompt_tokens * 0.7)
                    messages = self._trim_messages(
                        messages, max_prompt_tokens
                    )
                    self.retry_count += 1
                    time.sleep(self.retry_delay)
                    continue

                print(
                    f"\n[{current_time}] Error: Failed to reach "
                    f"LM Studio - {e}"
                )
                print(
                    "Make sure LM Studio is running. "
                    "Start it with: lms server start"
                )

                if self.logger:
                    self.logger.log_error(
                        "http_error",
                        str(e),
                        {"retry_count": self.retry_count},
                    )

                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    print(
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                break

            except requests.exceptions.Timeout:
                print(
                    f"\n[{current_time}] Error: Request timeout (120s)"
                )

                if self.logger:
                    self.logger.log_error(
                        "timeout",
                        "Request timeout after 120 seconds",
                        {"retry_count": self.retry_count},
                    )

                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    print(
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                break

            except requests.exceptions.ConnectionError:
                print(
                    f"\n[{current_time}] Error: Cannot connect to "
                    f"LM Studio"
                )
                print(
                    "Make sure LM Studio is running. "
                    "Start it with: lms server start"
                )

                if self.logger:
                    self.logger.log_error(
                        "connection_error",
                        "Failed to connect to LM Studio",
                        {"endpoint": self.endpoint},
                    )

                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    print(
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                break

            except Exception as e:
                print(f"\n[{current_time}] Unexpected error: {e}")

                if self.logger:
                    self.logger.log_error(
                        "unexpected_error",
                        str(e),
                        {"retry_count": self.retry_count},
                    )

                self.retry_count += 1
                break

        print(
            f"\n[{current_time}] Failed to get response from "
            f"LM Studio after {self.retry_count} attempts"
        )
        return None


class CommandReviewer:
    """Reviews unknown tool calls and provides suggestions"""

    def __init__(
        self,
        client: LMStudioClient,
        logger: Optional[Logger] = None,
        sandbox_root: str = "./sandbox",
    ):
        self.client = client
        self.logger = logger
        self.reviewer_instructions = (
            Path(sandbox_root) / "reviewer.txt"
        )

    def _get_reviewer_instructions(self) -> str:
        """Load reviewer instructions from file"""
        if self.reviewer_instructions.exists():
            try:
                return self.reviewer_instructions.read_text(
                    encoding="utf-8"
                )
            except Exception as e:
                return f"Error reading reviewer.txt: {e}"
        return (
            "Review the unknown tool call and explain what went wrong. "
            "Provide the correct tool name and parameters format."
        )

    def review_command(
        self, tool_call: dict, full_response: str
    ) -> str:
        """Review an unknown tool call"""
        instructions = self._get_reviewer_instructions()

        review_prompt = f"""You are a command reviewer AI. A main AI agent tried to use a tool that failed.

=== MAIN AGENT'S RESPONSE ===
{full_response}

=== FAILED TOOL CALL ===
{json.dumps(tool_call, indent=2)}

=== REVIEWER INSTRUCTIONS ===
{instructions}

=== YOUR TASK ===
Analyze what went wrong and provide a clear, actionable suggestion for fixing it.
Be concise and helpful."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful command reviewer. "
                    "Analyze failed tool calls and provide clear, "
                    "actionable suggestions."
                ),
            },
            {"role": "user", "content": review_prompt},
        ]

        try:
            response = self.client.chat(
                messages,
                temperature=0.3,
                max_tokens=800,
                max_prompt_tokens=8000,
            )

            if response and self.logger:
                content = response.get("content", "")
                self.logger.log_review(
                    json.dumps(tool_call), content
                )

            return response.get("content", "Unable to generate review")
        except Exception as e:
            return f"Error generating review: {e}"


class FeedbackManager:
    """Manage AI feedback and command requests"""

    def __init__(self, sandbox_root: str):
        self.sandbox_root = Path(sandbox_root).resolve()
        self.feedback_dir = self.sandbox_root / ".feedback"
        self.requests_dir = self.sandbox_root / ".requests"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir.mkdir(parents=True, exist_ok=True)

    def submit_feedback(self, feedback_text: str) -> str:
        """Submit feedback from AI"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feedback_file = (
                self.feedback_dir / f"feedback_{timestamp}.txt"
            )
            feedback_file.write_text(feedback_text, encoding="utf-8")
            return (
                f"Feedback submitted: {feedback_file.name}. "
                f"Note: This feedback may take hundreds of "
                f"iterations to be reviewed."
            )
        except Exception as e:
            return f"Error submitting feedback: {e}"

    def request_command(
        self, command_name: str, command_template: str
    ) -> str:
        """Request a new command to be added"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_file = (
                self.requests_dir / f"request_{timestamp}.json"
            )
            request_data = {
                "timestamp": datetime.now().isoformat(),
                "command_name": command_name,
                "command_template": command_template,
            }
            request_file.write_text(
                json.dumps(request_data, indent=2),
                encoding="utf-8",
            )
            return (
                f"Command request submitted: {request_file.name}. "
                f"Requested command: '{command_name}' = "
                f"'{command_template}'. "
                f"Note: This request may take hundreds of "
                f"iterations to be processed."
            )
        except Exception as e:
            return f"Error submitting command request: {e}"

    def list_feedback(self) -> str:
        """List all submitted feedback"""
        try:
            files = sorted(self.feedback_dir.glob("feedback_*.txt"))
            if not files:
                return "No feedback submitted yet."
            return "\n".join(f.name for f in files)
        except Exception as e:
            return f"Error listing feedback: {e}"

    def list_requests(self) -> str:
        """List all submitted command requests"""
        try:
            files = sorted(self.requests_dir.glob("request_*.json"))
            if not files:
                return "No command requests submitted yet."
            result = []
            for f in files:
                try:
                    data = json.loads(f.read_text())
                    result.append(
                        f"{f.name}: {data.get('command_name')} = "
                        f"{data.get('command_template')}"
                    )
                except Exception:
                    result.append(f"{f.name}: (error reading file)")
            return "\n".join(result)
        except Exception as e:
            return f"Error listing requests: {e}"


class AgentLoop:
    """Main autonomous agent loop"""

    def __init__(self, config_path: str = "settings.json"):
        self.config = AgentConfig(config_path)
        sandbox_root = self.config.get("sandbox_root", "./sandbox")
        self.files = FileManager(
            sandbox_root, self.config.get("max_file_size_mb", 10)
        )
        self.executor = CommandExecutor(self.config, sandbox_root)
        self.logger = Logger()
        self.client = LMStudioClient(
            self.config.get("api_endpoint", "http://localhost:1234/v1"),
            self.config.get("model", "openai/gpt-oss-20b"),
            context_window=self.config.get("context_window", 4096),
            logger=self.logger,
        )
        self.reviewer = CommandReviewer(
            self.client, self.logger, sandbox_root
        )
        self.feedback_manager = FeedbackManager(sandbox_root)
        self.history_file = Path(sandbox_root) / ".agent_history"
        self.last_message = self._load_last_message()
        self.max_iterations = self.config.get("max_iterations")
        self.credits_name = self.config.get(
            "credits_name", "Unknown Developer"
        )

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

        if self.last_message:
            context += f"Last message:\n{self.last_message}\n\n"

        readme = self.files.read_file("README.md")
        if not readme.startswith("Error"):
            context += f"README.md:\n{readme}\n\n"

        todo = self.files.read_file("TODO.md")
        if not todo.startswith("Error"):
            context += f"TODO.md:\n{todo}\n\n"

        context += f"Current directory:\n{self.files.show_directory()}\n"

        return context

    def _serialize_response(self, response: dict) -> str:
        """Serialize response dict to string for logging"""
        return json.dumps(response, indent=2)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if tool_name == "read_file":
            path = tool_input.get("path", "")
            result = self.files.read_file(path)
            self.logger.log_action(
                tool_name, {"path": path}, result
            )
            return result

        elif tool_name == "write_file":
            path = tool_input.get("path", "")
            content = tool_input.get("content", "")
            result = self.files.write_file(path, content)
            self.logger.log_action(
                tool_name,
                {"path": path, "content_length": len(content)},
                result,
            )
            return result

        elif tool_name == "show_dir":
            path = tool_input.get("path", ".")
            result = self.files.show_directory(path)
            self.logger.log_action(
                tool_name, {"path": path}, result
            )
            return result

        elif tool_name == "list_dir":
            path = tool_input.get("path", ".")
            result = self.files.list_directory(path)
            self.logger.log_action(
                tool_name, {"path": path}, result
            )
            return result

        elif tool_name == "execute":
            command = tool_input.get("command", "")
            result = self.executor.execute(command)
            self.logger.log_action(
                tool_name, {"command": command}, result
            )
            return result

        elif tool_name == "think":
            message = tool_input.get("message", "")
            result = f"Noted: {message}"
            self.logger.log_action(
                tool_name, {"message": message}, result
            )
            return result

        elif tool_name == "submit_feedback":
            message = tool_input.get("message", "")
            result = self.feedback_manager.submit_feedback(message)
            self.logger.log_action(
                tool_name, {"message": message}, result
            )
            return result

        elif tool_name == "request_command":
            cmd_name = tool_input.get("command_name", "")
            cmd_template = tool_input.get("command_template", "")
            result = self.feedback_manager.request_command(
                cmd_name, cmd_template
            )
            self.logger.log_action(
                tool_name,
                {
                    "command_name": cmd_name,
                    "command_template": cmd_template,
                },
                result,
            )
            return result

        else:
            result = (
                f"❌ UNKNOWN TOOL: '{tool_name}'\n\n"
                f"Available tools:\n"
                f"- read_file\n"
                f"- write_file\n"
                f"- show_dir\n"
                f"- list_dir\n"
                f"- execute\n"
                f"- think\n"
                f"- submit_feedback\n"
                f"- request_command"
            )

            review = self.reviewer.review_command(
                {"tool": tool_name, "input": tool_input},
                f"Tool call attempt: {tool_name}",
            )

            result += f"\n\n=== AI REVIEWER ANALYSIS ===\n{review}"

            self.logger.log_action(
                "unknown_tool",
                {"requested_tool": tool_name},
                result,
            )
            return result

    def run(self):
        """Main agent loop"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("Autonomous AI Agent Loop")
        print(f"Started: {current_time}")
        print(f"Sandbox root: {self.files.root}")
        print(f"Credits: {self.credits_name}")
        print("Press Ctrl+C to exit\n")

        context = self._build_context()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous AI agent with access to tools. "
                    "CRITICAL: You are NEVER done. Always look for "
                    "improvements. The task description is the MINIMUM "
                    "requirement. After completing it, refactor, optimize, "
                    "add features, improve documentation, add tests, "
                    "enhance security. Use git to track ALL progress with "
                    "descriptive commits. Use the available tools to "
                    "accomplish your goals. Be ambitious and make the "
                    "project excellent."
                ),
            },
            {"role": "user", "content": context},
        ]

        iteration = 0
        while True:
            try:
                iteration += 1
                current_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                if (
                    self.max_iterations
                    and iteration > self.max_iterations
                ):
                    print(
                        f"\n[{current_time}] Reached max iterations. "
                        f"Exiting."
                    )
                    break

                print(f"\n[{current_time}] === Iteration {iteration} ===")

                response = self.client.chat(
                    messages,
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 2000),
                    max_prompt_tokens=self.config.get(
                        "max_prompt_tokens", 3000
                    ),
                )

                if response is None:
                    print(
                        f"[{current_time}] Failed to get response. "
                        f"Retrying..."
                    )
                    time.sleep(5)
                    continue

                response_str = self._serialize_response(response)
                print(f"Response:\n{response_str}\n")
                self._save_message(response_str)

                content = response.get("content", "")
                self.logger.log_response(content, "")

                messages.append({"role": "assistant", "content": response})

                tool_calls = response.get("tool_calls", [])

                if tool_calls:
                    for tool_call in tool_calls:
                        func = tool_call.get("function", {})
                        tool_name = func.get("name", "")
                        tool_input = {}
                        try:
                            tool_input = json.loads(
                                func.get("arguments", "{}")
                            )
                        except json.JSONDecodeError:
                            pass

                        result = self._execute_tool(
                            tool_name, tool_input
                        )
                        print(f"Tool: {tool_name}")
                        print(f"Input: {json.dumps(tool_input)}")
                        print(f"Result:\n{result}\n")

                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Tool '{tool_name}' result:\n{result}"
                                ),
                            }
                        )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please use the available tools to "
                                "accomplish your goals. Call a tool to "
                                "proceed."
                            ),
                        }
                    )

                if len(messages) > 20:
                    messages = messages[:2] + messages[-18:]

                time.sleep(1)

            except KeyboardInterrupt:
                current_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"\n[{current_time}] Exiting agent loop")
                break
            except Exception as e:
                current_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"[{current_time}] Error: {e}")
                self.logger.log_error(
                    "loop_error", str(e), {"iteration": iteration}
                )
                time.sleep(2)


if __name__ == "__main__":
    agent = AgentLoop()
    agent.run()