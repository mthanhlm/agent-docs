class ShortTermMemory:
    def __init__(self, max_messages=10):
        """
        Initialize short term memory.
        :param max_messages: Maximum number of messages to keep (including system prompt).
        """
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, message):
        """
        Add a new message to memory and trim if it exceeds the limit.
        """
        self.messages.append(message)
        self._trim()

    def _trim(self):
        """
        Keep the system prompt and the most recent messages.
        """
        if len(self.messages) <= self.max_messages:
            return

        system_message = None
        if self.messages[0]["role"] == "system":
            system_message = self.messages[0]
            other_messages = self.messages[1:]
        else:
            other_messages = self.messages

        # Calculate number of messages to keep (excluding system prompt)
        keep_count = self.max_messages - (1 if system_message else 0)
        trimmed_messages = other_messages[-keep_count:]

        self.messages = ([system_message] if system_message else []) + trimmed_messages

    def get_messages(self):
        """
        Return the current list of messages.
        """
        return self.messages

    def clear(self):
        """
        Clear all memory except for the system prompt.
        """
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
