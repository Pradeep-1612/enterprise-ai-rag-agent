tools_schema = [{
    "type": "function",
    "function": {
        "name": "evaluate_certificate",
        "description": "Evaluate a production certificate based on the number of days remaining.",
        "parameters": {
            "type": "object",
            "properties": {
                "remaining_days": {
                    "type": "integer",
                    "description": "Number of days remaining before certificate expiration."
                }
            },
            "required": ["remaining_days"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "get_connection_status",
        "description": "Get the status of a trading partner integration connection.",
        "parameters": {
            "type": "object",
            "properties": {
                "connection_id": {
                    "type": "string",
                    "description": "The trading partner connection ID."
                }
            },
            "required": ["connection_id"]
        }
    }
}]