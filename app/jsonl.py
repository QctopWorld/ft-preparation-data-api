import pandas as pd
import json


def dataframe_to_jsonl(df: pd.DataFrame, system: str, user: str, assistant: str) -> str:
    """
    Génère un JSONL à partir d’un DataFrame et de 3 templates
    (chaîne avec {{colonne}} remplacées ligne par ligne)
    """
    lines = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        msg_system = system
        msg_user = user
        msg_assistant = assistant
        for col, val in row_dict.items():
            placeholder = "{{" + col + "}}"
            val_str = str(val)
            msg_system = msg_system.replace(placeholder, val_str)
            msg_user = msg_user.replace(placeholder, val_str)
            msg_assistant = msg_assistant.replace(placeholder, val_str)
        messages = [
            {"role": "system", "content": msg_system},
            {"role": "user", "content": msg_user},
            {"role": "assistant", "content": msg_assistant},
        ]
        lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
    return "\n".join(lines)
