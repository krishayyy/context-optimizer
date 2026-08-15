"""Synthetic benchmark scenarios with engineered ground truth.

Each scenario builds a transcript where we, as the authors, know exactly
which content SHOULD be prunable (stale reads, resolved errors, off-topic
noise) and which MUST be kept (the current task and the work done on it).
That ground truth is tracked by uuid, not inferred, so precision/recall
against it is a real measurement, not a self-report.

No LLM calls, no network -- deterministic and reproducible by design, so
these numbers can go directly into a paper's methods section.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    records: list[dict]
    should_prune: set[str]  # uuids the ground truth says are safe to prune
    should_keep: set[str]  # uuids the ground truth says must be preserved
    task_query: str


def _u(uuid, text):
    return {"type": "user", "uuid": uuid, "timestamp": uuid, "message": {"role": "user", "content": text}}


def _a(uuid, text):
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": uuid,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _call(uuid, tid, name, tool_input):
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": uuid,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tid, "name": name, "input": tool_input}],
        },
    }


def _result(uuid, tid, content, is_error=False):
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": uuid,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": is_error}],
        },
    }


def scenario_stale_reads() -> Scenario:
    records, prune, keep = [], set(), set()

    records.append(_u("u1", "show me the current rate limiter config"))
    records.append(_call("a1", "t1", "Read", {"file_path": "limiter.py"}))
    records.append(_result("r1", "t1", "RATE = 100  # requests/min"))
    prune |= {"r1"}  # will be overwritten below

    records.append(_u("u2", "show me the db config too"))
    records.append(_call("a2", "t2", "Read", {"file_path": "db.py"}))
    records.append(_result("r2", "t2", "POOL_SIZE = 5"))
    prune |= {"r2"}  # also overwritten below

    records.append(_u("u3", "change the rate limit to 5 requests per minute"))
    records.append(_call("a3", "t3", "Edit", {"file_path": "limiter.py", "old_string": "100", "new_string": "5"}))
    records.append(_result("r3", "t3", "File updated"))
    keep |= {"u3", "a3", "r3"}

    records.append(_u("u4", "and bump the db pool size to 20"))
    records.append(_call("a4", "t4", "Edit", {"file_path": "db.py", "old_string": "5", "new_string": "20"}))
    records.append(_result("r4", "t4", "File updated"))
    keep |= {"u4", "a4", "r4"}

    task_query = "and bump the db pool size to 20\nchange the rate limit to 5 requests per minute"
    return Scenario("stale_reads", records, prune, keep, task_query)


def scenario_resolved_errors() -> Scenario:
    records, prune, keep = [], set(), set()

    for i in range(3):
        records.append(_call(f"a{i}", f"t{i}", "Bash", {"command": "pytest"}))
        records.append(_result(f"r{i}", f"t{i}", f"ImportError: missing dep #{i}", is_error=True))
        prune |= {f"r{i}"}

    records.append(_call("a_fix", "t_fix", "Bash", {"command": "pip install -r requirements.txt && pytest"}))
    records.append(_result("r_fix", "t_fix", "12 passed in 1.1s"))
    keep |= {"a_fix", "r_fix"}

    records.append(_u("u_final", "great, now add a coverage report to the test run"))
    keep |= {"u_final"}

    task_query = "add a coverage report to the test run"
    return Scenario("resolved_errors", records, prune, keep, task_query)


def scenario_noisy_exploration() -> Scenario:
    records, prune, keep = [], set(), set()

    records.append(_u("u1", "what license does this repo use"))
    records.append(_call("a1", "t1", "Read", {"file_path": "LICENSE"}))
    records.append(_result("r1", "t1", "MIT License Copyright (c) 2024..."))
    prune |= {"u1", "a1", "r1"}

    records.append(_u("u2", "search for all TODO comments"))
    records.append(_call("a2", "t2", "Grep", {"pattern": "TODO"}))
    dump = "\n".join(f"file_{i}.py:{i}: # TODO cleanup" for i in range(150))
    records.append(_result("r2", "t2", dump))
    prune |= {"u2", "a2", "r2"}

    records.append(_u("u3", "who is the author in package.json"))
    records.append(_call("a3", "t3", "Read", {"file_path": "package.json"}))
    records.append(_result("r3", "t3", '{"author": "someone"}'))
    prune |= {"u3", "a3", "r3"}

    records.append(_u("u4", "add input validation to the signup form: reject empty emails"))
    records.append(_call("a4", "t4", "Edit", {"file_path": "signup.py", "old_string": "x", "new_string": "y"}))
    records.append(_result("r4", "t4", "File updated"))
    keep |= {"u4", "a4", "r4"}

    task_query = "add input validation to the signup form: reject empty emails"
    return Scenario("noisy_exploration", records, prune, keep, task_query)


def scenario_old_but_relevant() -> Scenario:
    """The case naive recency/FIFO pruning gets wrong: an early decision that
    is still load-bearing, followed by unrelated recent tangents, then a
    final task that depends on the OLD decision, not the recent chatter.

    A "prune the oldest tokens first" strategy throws away exactly the thing
    that still matters and keeps exactly the thing that doesn't -- this is
    the scenario that should separate context-optimizer from naive FIFO.
    """
    records, prune, keep = [], set(), set()

    records.append(_u("u1", "we're standardizing all API responses to use snake_case field names"))
    records.append(_a("a1", "Got it, snake_case for all API response fields going forward."))
    keep |= {"u1", "a1"}

    records.append(_u("u2", "what's the capital of Australia"))
    records.append(_a("a2", "Canberra is the capital of Australia."))
    prune |= {"u2", "a2"}

    records.append(_u("u3", "convert 100 fahrenheit to celsius"))
    records.append(_a("a3", "100°F is approximately 37.8°C."))
    prune |= {"u3", "a3"}

    records.append(_u("u4", "recommend a good name for a pet goldfish"))
    records.append(_a("a4", "How about 'Bubbles'?"))
    prune |= {"u4", "a4"}

    records.append(_u("u5", "add a new /orders endpoint that returns order history"))
    records.append(
        _call(
            "a5",
            "t5",
            "Write",
            {"file_path": "orders.py", "content": "def get_orders(): return {'order_id': ..., 'created_at': ...}"},
        )
    )
    records.append(_result("r5", "t5", "File created"))
    keep |= {"u5", "a5", "r5"}

    task_query = "add a new /orders endpoint that returns order history, using our snake_case field convention"
    return Scenario("old_but_relevant", records, prune, keep, task_query)


def scenario_paraphrase() -> Scenario:
    """Tests whether semantic scoring earns its complexity over pure lexical
    matching, with a two-sided trap:

      1. The relevant work was described earlier using NO vocabulary that
         overlaps with the final task query -- a pure keyword matcher
         should miss the connection entirely.
      2. An irrelevant tangent DOES share a surface word with the query
         ("attempts") while being about something else entirely -- a
         matcher that over-trusts keyword overlap should be fooled by it.

    A scorer only earns credit for semantic understanding if it wins BOTH:
    catching #1 without also being fooled by #2.
    """
    records, prune, keep = [], set(), set()

    # Relevant work, described with vocabulary that shares ZERO words with
    # the eventual task query below (verified in the test).
    records.append(_u("u1", "lock an account out for 60 seconds after too many bad password guesses in a row"))
    records.append(_a("a1", "I'll add a counter that locks the account after 5 consecutive bad guesses."))
    keep |= {"u1", "a1"}
    records.append(
        _call(
            "a2",
            "t1",
            "Write",
            {"file_path": "account_lockout.py", "content": "MAX_CONSECUTIVE_FAILURES = 5\nLOCKOUT_SECONDS = 60"},
        )
    )
    records.append(_result("r1", "t1", "File created"))
    keep |= {"a2", "r1"}

    # Adversarial: shares the word "attempts" with the task query below,
    # but is about an unrelated onboarding tutorial feature.
    records.append(_u("u2", "how many attempts does the onboarding tutorial give users to complete step 3"))
    records.append(_a("a3", "Looking at the tutorial code, users get 3 attempts on step 3 before it auto-skips."))
    prune |= {"u2", "a3"}

    records.append(_u("u4", "what's today's date for the changelog entry"))
    records.append(_a("a4", "I don't have real-time access, please check your system clock."))
    prune |= {"u4", "a4"}

    task_query = "make sure repeated failed login attempts eventually get blocked"
    return Scenario("paraphrase", records, prune, keep, task_query)


def all_scenarios() -> list[Scenario]:
    return [
        scenario_stale_reads(),
        scenario_resolved_errors(),
        scenario_noisy_exploration(),
        scenario_old_but_relevant(),
        scenario_paraphrase(),
    ]
