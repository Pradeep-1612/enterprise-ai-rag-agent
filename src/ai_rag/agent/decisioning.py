def route_after_tools(state):
    print("\n[NODE] ROUTE_AFTER_TOOLS")

    tool_results = state.get("tool_results", [])

    certificate_entry = next(
        (
            item for item in tool_results
            if item.get("tool") == "evaluate_certificate"
        ),
        None,
    )

    # No usable certificate evaluation: do not make a decision.
    if (
        certificate_entry is None
        or not certificate_entry.get("result", {}).get("success")
    ):
        decision = {
            "action": "unable_to_evaluate",
            "reason": "Certificate evaluation is missing or failed.",
        }
        state["decision"] = decision
        state["next_node"] = "final_agent"
        state["steps"].append("route_after_tools")
        return state

    certificate = certificate_entry["result"].get("data", {})
    days = certificate.get("remaining_days")

    if days is None:
        decision = {
            "action": "unable_to_evaluate",
            "reason": "Remaining certificate validity is unknown.",
        }

    # Preserve emergency priority: fewer than 7 days remaining.
    elif days < 7:
        decision = {
            "action": "emergency_renewal",
            "reason": f"Certificate expires in {days} days.",
        }

    elif days < 15:
        connection_entry = next(
            (
                item for item in tool_results
                if item.get("tool") == "get_connection_status"
            ),
            None,
        )

        connection_result = (
            connection_entry.get("result", {})
            if connection_entry else {}
        )

        # Fail safely when connection status cannot be established.
        if (
            not connection_result.get("success")
            or not isinstance(connection_result.get("data"), dict)
            or not connection_result["data"].get("status")
            or connection_result["data"]["status"] == "not_found"
        ):
            decision = {
                "action": "unable_to_evaluate",
                "reason": (
                    "Certificate is at risk, but connection status "
                    "is unavailable or unknown."
                ),
            }

        elif connection_result["data"]["status"] == "failed":
            decision = {
                "action": "human_approval",
                "reason": (
                    f"Certificate expires in {days} days and its "
                    "connection has failed."
                ),
            }

        elif connection_result["data"]["status"] == "healthy":
            decision = {
                "action": "notify_operations",
                "reason": (
                    f"Certificate expires in {days} days; "
                    "notify operations."
                ),
            }

        else:
            decision = {
                "action": "unable_to_evaluate",
                "reason": "Connection status is not recognized.",
            }

    else:
        decision = {
            "action": "no_action",
            "reason": f"Certificate has {days} days remaining.",
        }

    state["decision"] = decision
    state["next_node"] = "final_agent"
    state["steps"].append("route_after_tools")
    return state