
from ai_rag.agent.workflow import run


def main():
    question = (
        "The certificate has 10 days remaining "
        "and is connected to CONN-1002. "
        "What should we do?"
    )

    state = run(question)

    # 1. Verify required tools were called.
    called_tools = {
        item["tool"]
        for item in state["tool_results"]
    }

    required_tools = {
        "evaluate_certificate",
        "get_connection_status",
    }

    missing_tools = required_tools - called_tools
    tools_passed = not missing_tools

    print("\n--- EVALUATION ---")

    print(
        f"[{'PASS' if tools_passed else 'FAIL'}] "
        f"Required tools called: "
        f"{sorted(required_tools & called_tools)}"
    )

    if missing_tools:
        print(f"Missing tools: {sorted(missing_tools)}")

    # 2. Verify the deterministic business decision.
    actual_action = (
        state.get("decision") or {}
    ).get("action")

    expected_action = "human_approval"
    decision_passed = actual_action == expected_action

    print(
        f"[{'PASS' if decision_passed else 'FAIL'}] "
        f"Decision: expected={expected_action}, "
        f"actual={actual_action}"
    )

    # 3. Verify that a final answer was generated.
    answer_passed = bool(
        state.get("final_answer", "").strip()
    )

    print(
        f"[{'PASS' if answer_passed else 'FAIL'}] "
        "Final answer generated"
    )

    passed = sum([
        tools_passed,
        decision_passed,
        answer_passed,
    ])

    print(f"\nEvaluation: {passed}/3 checks passed")

    if passed != 3:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
