"""Loop 2: On-policy distillation modules.

Collects Qwen rollouts, performs Opus-guided trace surgery on failures,
accumulates corrected traces, and triggers QLoRA SFT.
"""

from src.loop2_distill.trace_collector import TraceCollector
from src.loop2_distill.trace_surgeon import TraceSurgeon
from src.loop2_distill.pending_buffer import PendingBuffer
from src.loop2_distill.sft_launcher import SFTLauncher
from src.loop2_distill.mentor_session import MentorSession
from src.loop2_distill.sdpo_reevaluator import SDPOReevaluator

__all__ = [
    "TraceCollector",
    "TraceSurgeon",
    "PendingBuffer",
    "SFTLauncher",
    "MentorSession",
    "SDPOReevaluator",
]
