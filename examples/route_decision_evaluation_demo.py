
from ai_rag.agent.decisioning import route_after_tools


def certificate_result(days):
    return {
        "success": True,
        "data": {
            "remaining_days": days,
            "at_risk": days < 30,
            "notify_operations": days < 15,
            "emergency_renewal": days < 7,
        },
    }


def connection_result(status):
    return {
        "success": True,
        "data": {
            "status": status,
            "trading_partner": "Partner B",
        },
    }


def run_scenario(name, days, connection, expected_action):
    tool_results = [
        {
            "tool": "evaluate_certificate",
            "result": certificate_result(days),
        }
    ]

    if connection is not None:
        tool_results.append({
            "tool": "get_connection_status",
            "result": connection,
        })

    state = {
        "tool_results": tool_results,
        "decision": None,
        "steps": [],
        "next_node": None,
    }

    route_after_tools(state)

    actual_action = state["decision"]["action"]
    passed = actual_action == expected_action

    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}: "
        f"expected={expected_action}, actual={actual_action}"
    )

    if not passed:
        print(f"       Decision: {state['decision']}")

    return passed


def main():
    scenarios = [
        (
            "Emergency renewal takes priority",
            5,
            connection_result("failed"),
            "emergency_renewal",
        ),
        (
            "Failed connection requires approval",
            10,
            connection_result("failed"),
            "human_approval",
        ),
        (
            "Healthy connection requires notification",
            10,
            connection_result("healthy"),
            "notify_operations",
        ),
        (
            "Certificate with 20 days remaining",
            20,
            connection_result("failed"),
            "no_action",
        ),
        (
            "Connection lookup failed",
            10,
            {
                "success": False,
                "error": "CONNECTION_LOOKUP_FAILED",
            },
            "unable_to_evaluate",
        ),
        (
            "Connection result missing",
            10,
            None,
            "unable_to_evaluate",
        ),
        (
            "Connection status is unrecognized",
            10,
            connection_result("unknown"),
            "unable_to_evaluate",
        ),
    ]

    passed = 0

    for scenario in scenarios:
        if run_scenario(*scenario):
            passed += 1

    total = len(scenarios)
    print(f"\nEvaluation: {passed}/{total} scenarios passed")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
