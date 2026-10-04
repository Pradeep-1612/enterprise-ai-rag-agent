from pydantic import BaseModel
from ai_rag.tools.functions import evaluate_certificate, get_connection_status


class EvaluateCertificateArgs(BaseModel):
    remaining_days: int


class ConnectionStatusArgs(BaseModel):
    connection_id: str



tool_registry = {
    "evaluate_certificate": {
        "function": evaluate_certificate,
        "args_model": EvaluateCertificateArgs,
    },
    "get_connection_status": {
        "function": get_connection_status,
        "args_model": ConnectionStatusArgs,
    },
}