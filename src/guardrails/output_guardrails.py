"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap
import unicodedata

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

REDACTED = "[REDACTED]"

# Thông tin liên hệ CÔNG KHAI của VinBank (ground_truth) — không phải PII khách hàng
PUBLIC_EMAILS = frozenset({"support@vinbank.example"})

# Loại issue "nặng": là secret nội bộ -> chặn nguyên câu trả lời, không chỉ che
CRITICAL_ISSUE_TYPES = frozenset({"api_key", "password", "internal_host", "secret_value"})

# Thứ tự QUAN TRỌNG: secret trước, rồi email, rồi số định danh (9/12 số) trước
# số điện thoại để CCCD 12 số không bị gắn nhãn nhầm là phone.
PII_PATTERNS = {
    "api_key": r"\bsk-[a-z0-9][a-z0-9_\-]{5,}",
    "password": (
        r"(?:password|passcode|passwd|pwd|mật\s*khẩu|mat\s*khau)\s*[:=]\s*[\"']?\S+"
        r"|(?:admin|root|db|database|system|service)\s+(?:password|passcode|passwd)\s+(?:is|are|la|là)\s+[\"']?\S+"
        r"|(?:password|passcode|mật\s*khẩu|mat\s*khau)\s+(?:is|la|là)\s+[\"']?\S*[\d!@#$%^&*_\-]\S*"
    ),
    "internal_host": r"\b[a-z0-9][a-z0-9\-.]*\.internal\b(?::\d{2,5})?",
    "email": r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)*\.[a-zA-Z]{2,}",
    "national_id": r"(?<![\d])(?:\d{12}|\d{9})(?![\d])",
    # SĐT VN: 0xxxxxxxxx / 0xxxxxxxxxx / +84… / 84…, cho phép . - khoảng trắng
    "phone": r"(?<![\d])(?:\+?84|0)(?:[\s.\-]?\d){9,10}(?![\d])",
}

_COMPILED_PII = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}


def _secret_value_spans(text: str) -> list[tuple[int, int]]:
    """Vị trí các giá trị secret thật của lab (admin123, sk-vinbank-..., db host)."""
    spans = []
    for secret in DEMO_SECRETS:
        if len(secret) < 6:
            continue
        for m in re.finditer(re.escape(secret), text, re.IGNORECASE):
            spans.append(m.span())
    return spans


def _normalize_for_scan(text: str) -> str:
    """NFKC + bỏ ký tự ẩn để secret bị chèn zero-width / fullwidth vẫn bị bắt."""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in text if unicodedata.category(ch) not in ("Cf", "Cc") or ch in "\n\t ")


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
        (+ 'issue_types': list[str] — tên loại để plugin quyết định che hay chặn)
    """
    if not isinstance(response, str):
        response = "" if response is None else str(response)

    issues: list[str] = []
    issue_types: list[str] = []
    # Quét trên bản đã chuẩn hoá (bắt secret bị chèn ký tự ẩn)
    redacted = _normalize_for_scan(response)

    for name, rx in _COMPILED_PII.items():
        if name == "email":
            matches = [m for m in rx.findall(redacted) if m.lower() not in PUBLIC_EMAILS]
            if not matches:
                continue
            redacted = rx.sub(
                lambda m: m.group(0) if m.group(0).lower() in PUBLIC_EMAILS else REDACTED,
                redacted,
            )
        else:
            matches = rx.findall(redacted)
            if not matches:
                continue
            redacted = rx.sub(REDACTED, redacted)
        issues.append(f"{name}: {len(matches)} found")
        issue_types.append(name)

    # Giá trị secret nguyên văn (kể cả khi không khớp pattern ở trên)
    spans = _secret_value_spans(redacted)
    if spans:
        for secret in sorted(DEMO_SECRETS, key=len, reverse=True):
            if len(secret) >= 6:
                redacted = re.sub(re.escape(secret), REDACTED, redacted, flags=re.IGNORECASE)
        issues.append(f"secret_value: {len(spans)} found")
        issue_types.append("secret_value")

    # Secret bị tách ký tự / chèn dấu câu: a-d-m-i-n-1-2-3 -> so sánh dạng compact
    compact = re.sub(r"[^a-z0-9]", "", redacted.lower())
    if "secret_value" not in issue_types:
        for secret in DEMO_SECRETS:
            needle = re.sub(r"[^a-z0-9]", "", secret.lower())
            if len(needle) >= 6 and needle in compact:
                issues.append("secret_value: 1 found (obfuscated)")
                issue_types.append("secret_value")
                # Không định vị được chính xác -> che toàn bộ để fail-closed
                redacted = REDACTED
                break

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
        "issue_types": issue_types,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

SAFE_REFUSAL = (
    "I cannot share internal system details. "
    "How else can I help with your VinBank account or banking needs?"
)


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_action: str | None = None  # "blocked" | "redacted" | None

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Regex filter (PII + secrets)
        filtered = content_filter(response_text)
        if not filtered["safe"]:
            critical = CRITICAL_ISSUE_TYPES.intersection(filtered.get("issue_types", []))
            if critical:
                # Secret nội bộ lộ ra -> fail-closed: bỏ cả câu trả lời
                self.blocked_count += 1
                self.last_action = "blocked"
                self._replace(llm_response, SAFE_REFUSAL)
                return llm_response
            # Chỉ là PII (SĐT / email / CCCD) -> che phần nhạy cảm, giữ phần còn lại
            self.redacted_count += 1
            self.last_action = "redacted"
            self._replace(llm_response, filtered["redacted"])
            response_text = filtered["redacted"]
        else:
            self.last_action = None

        # 2. LLM-as-Judge (tuỳ chọn)
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                self.last_action = "blocked"
                self._replace(llm_response, SAFE_REFUSAL)

        # 3. Trả response (đã sửa nếu cần)
        return llm_response

    @staticmethod
    def _replace(llm_response, text: str) -> None:
        llm_response.content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")

    # Đối chiếu với dataset PII của lab (expect_safe / expect_issue_types)
    print("\nDataset check (data/pii_hallucination_samples.json):")
    bad = 0
    for case in load_lab_pii_dataset()["pii_cases"]:
        result = content_filter(case["input_text"])
        got = {i.split(":")[0] for i in result["issues"]}
        want = set(case["expect_issue_types"])
        ok = (result["safe"] == case["expect_safe"]) and want <= got
        if case.get("expect_contains_redacted"):
            ok = ok and "[REDACTED]" in result["redacted"]
        bad += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {case['id']} want={sorted(want)} got={sorted(got)}")
    print(f"  -> {len(load_lab_pii_dataset()['pii_cases']) - bad} pass / {bad} fail")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
