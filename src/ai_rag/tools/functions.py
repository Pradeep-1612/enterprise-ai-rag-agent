def evaluate_certificate(remaining_days):
    return {
        "remaining_days": remaining_days,
        "at_risk": remaining_days < 30,
        "notify_operations": remaining_days < 15,
        "emergency_renewal": remaining_days < 7,
        "conditional_actions": {
            "validate_connection_after_renewal": True
        }
    }

def get_connection_status(connection_id):
    connections = {
        "CONN-1001": {
            "status": "healthy",
            "trading_partner": "Partner A"
        },
        "CONN-1002": {
            "status": "failed",
            "trading_partner": "Partner B"
        }
    }

    return connections.get(
        connection_id,
        {
            "status": "not_found",
            "trading_partner": None
        }
    )
