COMMAND_MAP = {
    "forward": ["forward", "go forward", "move forward"],
    "backward": ["backward", "go back", "move back"],
    "left": ["left", "turn left"],
    "right": ["right", "turn right"],
    "stop": ["stop", "halt", "pause"],
}


def detect_command(text: str):
    text = text.lower()

    for command, phrases in COMMAND_MAP.items():
        for phrase in phrases:
            if phrase in text:
                return command

    return None