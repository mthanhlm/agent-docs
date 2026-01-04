"""Short-term memory management for conversational AI agents."""


class ShortTermMemory:
    """Manages conversation history with a sliding window approach."""
    
    def __init__(self, max_messages: int = 10):
        """
        Initialize the memory buffer.
        
        Args:
            max_messages: Maximum number of messages to retain (including system prompt).
        """
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, message: dict):
        """Append a message and enforce the size limit."""
        self.messages.append(message)
        self._trim()

    def _trim(self):
        """Preserve the system prompt while trimming older messages."""
        if len(self.messages) <= self.max_messages:
            return

        system_message = None
        if self.messages[0]["role"] == "system":
            system_message = self.messages[0]
            other_messages = self.messages[1:]
        else:
            other_messages = self.messages

        keep_count = self.max_messages - (1 if system_message else 0)
        trimmed_messages = other_messages[-keep_count:]
        self.messages = ([system_message] if system_message else []) + trimmed_messages

    def get_messages(self) -> list:
        """Return the current message history."""
        return self.messages

    def clear(self):
        """Reset memory, preserving only the system prompt if present."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
