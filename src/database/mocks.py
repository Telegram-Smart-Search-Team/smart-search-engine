# mocks.py


class MockBot:
    """
    A mock implementation of the Bot to simulate user interaction
    for testing the Retriever.
    """

    def __init__(self):
        self.tracked_chats = set()
        print("MockBot initialized.")

    def add_tracked_chat(self, chat_id: int):
        """Simulates adding a chat to the tracking list."""
        self.tracked_chats.add(chat_id)
        print(f"Bot: Now tracking chat {chat_id}. Current list: {self.tracked_chats}")

    def remove_tracked_chat(self, chat_id: int):
        """Simulates removing a chat from the tracking list."""
        self.tracked_chats.discard(chat_id)
        print(f"Bot: Stopped tracking chat {chat_id}. Current list: {self.tracked_chats}")

    def receive_search_query(self, query: str):
        """
        Simulates a user sending a '/search' query.
        This is the entry point for testing the Retriever.
        """
        print(f"\nBot: Received search query: '{query}'")
        # In a real application, this would trigger the Retriever.
        # retriever.process_query(query)
        print("Bot: Forwarding query to Retriever...")

    def send_answer_to_user(self, answer: str):
        """Simulates sending the final generated answer back to the user."""
        print("\n----------- FINAL ANSWER -----------")
        print(f"Bot to User: {answer}")
        print("------------------------------------\n")


class MockClient:
    """
    A mock implementation of the Client to simulate new messages
    for testing the Dumper.
    """

    def __init__(self):
        self.is_logging_active = True
        print("MockClient initialized.")

    def get_status(self) -> str:
        """Simulates the '/status' command."""
        if self.is_logging_active:
            return "You are connected. System is up."
        else:
            return "Logging is paused. System is up."

    def stop_logging(self):
        """Simulates the '/stop_logging' command."""
        self.is_logging_active = False
        print("Client: Logging has been stopped.")

    def resume_logging(self):
        """Simulates the '/resume_logging' command."""
        self.is_logging_active = True
        print("Client: Logging has been resumed.")

    def simulate_new_message(self, message_data: dict):
        """
        Simulates a new message arriving from Telegram.
        This is the entry point for testing the Dumper.
        """
        if not self.is_logging_active:
            print(f"Client: Ignoring new message (ID: {message_data.get('message_id')}) because logging is off.")
            return

        print(f"\nClient: New message received (ID: {message_data.get('message_id')}).")
        # In a real application, this would trigger the Dumper.
        # dumper.process_message(message_data)
        print("Client: Forwarding message to Dumper...")
