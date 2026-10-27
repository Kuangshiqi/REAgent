# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

# TODO: remove these annotations by defining fine-grained types
# pyright: reportExplicitAny=false
# pyright: reportArgumentType=false
# pyright: reportAny=false

"""Trajectory recording functionality for Trae Agent."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trae_agent.tools.base import ToolCall, ToolResult
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse


class TrajectoryRecorder:
    """Records trajectory data for agent execution and LLM interactions."""

    def __init__(self, trajectory_path: str | None = None):
        """Initialize trajectory recorder.

        Args:
            trajectory_path: Path to save trajectory file. If None, generates default path.
        """
        if trajectory_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_path = f"trajectories/trajectory_{timestamp}.json"

        self.trajectory_path: Path = Path(trajectory_path).resolve()
        try:
            self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            print("Error creating trajectory directory. Trajectories may not be properly saved.")

        self.trajectory_data: dict[str, Any] = {
            "task": "",
            "start_time": "",
            "end_time": "",
            "provider": "",
            "model": "",
            "max_steps": 0,
            "llm_interactions": [],
            "agent_steps": [],
            "success": False,
            "final_result": None,
            "execution_time": 0.0,
        }
        self._start_time: datetime | None = None

    def start_recording(self, task: str, provider: str, model: str, max_steps: int) -> None:
        """Start recording a new trajectory.

        Args:
            task: The task being executed
            provider: LLM provider being used
            model: Model name being used
            max_steps: Maximum number of steps allowed
        """
        self._start_time = datetime.now()
        self.trajectory_data.update(
            {
                "task": task,
                "start_time": self._start_time.isoformat(),
                "provider": provider,
                "model": model,
                "max_steps": max_steps,
                "llm_interactions": [],
                "agent_steps": [],
            }
        )
        self.save_trajectory()

    def record_llm_interaction(
        self,
        messages: list[LLMMessage],
        response: LLMResponse,
        provider: str,
        model: str,
        tools: list[Any] | None = None,
    ) -> None:
        """Record an LLM interaction.

        Args:
            messages: Input messages to the LLM
            response: Response from the LLM
            provider: LLM provider used
            model: Model used
            tools: Tools available during the interaction
        """
        # The per-step agent trajectory already captures the minimal information we need.
        # Keep this method as a no-op for backward compatibility with existing callers.
        return

    def record_agent_step(
        self,
        step_number: int,
        state: str,
        llm_messages: list[LLMMessage] | None = None,
        llm_response: LLMResponse | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        reflection: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record an agent execution step.

        Args:
            step_number: Step number in the execution
            state: Current state of the agent
            llm_messages: Messages sent to LLM in this step
            llm_response: Response from LLM in this step
            tool_calls: Tool calls made in this step
            tool_results: Results from tool execution
            reflection: Agent reflection on the step
            error: Error message if step failed
        """
        # Minimal schema per user request:
        # - model input
        # - model output
        # - tools called in that round (no tool result content)
        step_data: dict[str, Any] = {
            "step_number": step_number,
            "state": state,
            "model_input": [self._serialize_message_min(msg) for msg in llm_messages]
            if llm_messages
            else None,
            "model_output": llm_response.content if llm_response else None,
            "tool_calls": [self._serialize_tool_call_min(tc) for tc in tool_calls]
            if tool_calls
            else None,
        }

        # Keep errors in a minimal way; helps debug without exploding the log size.
        if error:
            step_data["error"] = error

        self.trajectory_data["agent_steps"].append(step_data)
        self.save_trajectory()

    def update_lakeview(self, step_number: int, lakeview_summary: str):
        for step_data in self.trajectory_data["agent_steps"]:
            if step_data["step_number"] == step_number:
                step_data["lakeview_summary"] = lakeview_summary
                break
        self.save_trajectory()

    def finalize_recording(self, success: bool, final_result: str | None = None) -> None:
        """Finalize the trajectory recording.

        Args:
            success: Whether the task completed successfully
            final_result: Final result or output of the task
        """
        end_time = datetime.now()
        self.trajectory_data.update(
            {
                "end_time": end_time.isoformat(),
                "success": success,
                "final_result": final_result,
                "execution_time": (end_time - self._start_time).total_seconds()
                if self._start_time
                else 0.0,
            }
        )

        # Save to file
        self.save_trajectory()

    def save_trajectory(self) -> None:
        """Save the current trajectory data to file."""
        try:
            # Ensure directory exists
            self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.trajectory_path, "w", encoding="utf-8") as f:
                json.dump(self.trajectory_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Failed to save trajectory to {self.trajectory_path}: {e}")

    def _serialize_message(self, message: LLMMessage) -> dict[str, Any]:
        """Serialize an LLM message to a dictionary."""
        data: dict[str, Any] = {"role": message.role, "content": message.content}

        if message.tool_call:
            data["tool_call"] = self._serialize_tool_call(message.tool_call)

        if message.tool_result:
            data["tool_result"] = self._serialize_tool_result(message.tool_result)

        return data

    def _serialize_message_min(self, message: LLMMessage) -> dict[str, Any]:
        """Serialize an LLM message to a minimal dictionary.

        Only keeps role/content. Tool call/result payloads are intentionally omitted
        to keep trajectories compact.
        """

        return {"role": message.role, "content": message.content}

    def _serialize_tool_call_min(self, tool_call: ToolCall) -> dict[str, Any]:
        """Serialize a tool call to a minimal dictionary."""
        return {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        }

    def _serialize_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Serialize a tool call to a dictionary."""
        return {
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "id": getattr(tool_call, "id", None),
        }

    def _serialize_tool_result(self, tool_result: ToolResult) -> dict[str, Any]:
        """Serialize a tool result to a dictionary."""
        return {
            "call_id": tool_result.call_id,
            "success": tool_result.success,
            "result": tool_result.result,
            "error": tool_result.error,
            "id": getattr(tool_result, "id", None),
        }

    def get_trajectory_path(self) -> str:
        """Get the path where trajectory is being saved."""
        return str(self.trajectory_path)
