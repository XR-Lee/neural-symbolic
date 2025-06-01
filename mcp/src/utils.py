def format_message(agent_name, message_type, content):
    return {
        "agent": agent_name,
        "type": message_type,
        "content": content
    }

def log_message(message):
    print(f"[LOG] {message}")

def parse_message(message):
    agent_name = message.get("agent")
    message_type = message.get("type")
    content = message.get("content")
    return agent_name, message_type, content