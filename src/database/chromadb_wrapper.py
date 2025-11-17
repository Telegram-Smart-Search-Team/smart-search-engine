import chromadb
from chromadb.types import Collection
from typing import List, Dict, Any, Optional
from logging import getLogger

logger = getLogger("chromadb_wrapper")


class VectorDB:
    """
    API for interacting with ChromaDB.

    This class abstracts the underlying ChromaDB client and provides
    a clean, high-level interface for adding and searching messages.
    It is designed to be easily used by the Retriever component with
    minimal configuration.

    The database schema for each entry is as follows:
    - id (str): The unique message ID ('chat_id-message_id').
    - embedding (list[float]): The vector embedding of the message content.
    - metadata (dict): A dictionary containing all other information:
        - message_id (int): Original message ID from Telegram.
        - chat_id (int): The chat where the message originated.
        - author_id (int): The ID of the message author.
        - content (str): The raw text content or description of media.
        - message_type (str): The type of message (e.g., 'text', 'image').
        - retrieved_text_if_image (str, optional): Text extracted from image.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "telegram_chats"):
        """
        Initializes the VectorDB client and connects to the ChromaDB instance.

        Args:
            host (str): The hostname of the ChromaDB server.
            port (int): The port of the ChromaDB server.
            collection_name (str): The name of the collection to use.
        """
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            self._collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(
                f"Successfully connected to ChromaDB at {host}:{port} and loaded collection '{collection_name}'."
            )
        except Exception:
            logger.exception("Error connecting to ChromaDB")
            raise

    @property
    def collection(self) -> Collection:
        """Provides access to the underlying ChromaDB collection object."""
        return self._collection

    def add_message(
        self,
        message_id: int,
        chat_id: int,
        author_id: int,
        content: str,
        embedding: List[float],
        message_type: str,
        retrieved_text_if_image: Optional[str] = None,
    ) -> None:
        """
        Adds or updates a single message in the vector database.

        This method structures the data according to the defined schema
        and upserts it into the ChromaDB collection.

        Args:
            message_id (int): The unique ID of the message.
            chat_id (int): The ID of the chat.
            author_id (int): The ID of the message author.
            content (str): The text content or a description of the media.
            embedding (List[float]): The vector embedding of the content.
            message_type (str): Type of message, e.g., 'text', 'image'.
            retrieved_text_if_image (Optional[str]): Text extracted from an image.
        """
        doc_id = f"{chat_id}-{message_id}"
        metadata = {
            "message_id": message_id,
            "chat_id": chat_id,
            "author_id": author_id,
            "content": content,
            "message_type": message_type,
        }
        if retrieved_text_if_image:
            metadata["retrieved_text_if_image"] = retrieved_text_if_image

        try:
            self._collection.add(ids=[doc_id], embeddings=[embedding], metadatas=[metadata])
            logger.info(f"Added/updated message with ID {doc_id}.")
        except Exception:
            logger.exception(f"Error adding message {doc_id}")
            raise

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5, allowed_chat_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for the top_k most similar messages to a given query embedding.

        This is the primary method to be used by the Retriever. It can filter
        results to only include messages from a specific list of chats.

        Args:
            query_embedding (List[float]): The vector embedding of the user's search query.
            top_k (int): The number of similar messages to retrieve.
            allowed_chat_ids (Optional[List[int]]): A list of chat IDs to restrict
                                                   the search to. If None, searches all chats.

        Returns:
            List[Dict[str, Any]]: A list of candidate messages. Each message is a
                                  dictionary containing its metadata and the similarity score.
                                  Returns an empty list if no results are found.
        """
        where_filter = {}
        if allowed_chat_ids:
            # ChromaDB filter to search only in specific chats
            where_filter = {"chat_id": {"$in": allowed_chat_ids}}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where_filter if where_filter else None
            )

            # Format the output to be clean and easy to use
            formatted_results = []
            if results and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    candidate = {
                        "id": doc_id,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],  # cos_distance
                    }
                    formatted_results.append(candidate)
            logger.info(f"Found {len(formatted_results)} candidates for the query.")
            return formatted_results
        except Exception as e:
            logger.warning(f"Failure during search: {e}")
            return []


# --- Example Usage ---
if __name__ == "__main__":
    # This demonstrates how the components would interact in a test scenario.
    from mocks import MockBot, MockClient

    # 1. Initialize mocks and the DB API
    # Assumes a ChromaDB instance is running on localhost:8000
    try:
        db_api = VectorDB(collection_name="test_collection")
        bot = MockBot()
        client = MockClient()

        # 2. Simulate client receiving messages (would trigger Dumper)
        client.simulate_new_message(
            {
                "message_id": 101,
                "chat_id": -1001,
                "author_id": 123,
                "content": "Can someone explain the new ruff style guide?",
                "type": "text",
            }
        )
        # Dumper would process this and call db_api.add_message()
        # Let's simulate that call:
        db_api.add_message(
            message_id=101,
            chat_id=-1001,
            author_id=123,
            content="Can someone explain the new ruff style guide?",
            embedding=[0.1, 0.2, 0.9] * 512,  # Dummy embedding
            message_type="text",
        )

        client.simulate_new_message(
            {
                "message_id": 102,
                "chat_id": -1001,
                "author_id": 456,
                "content": "I think the new QWEN model is very powerful.",
                "type": "text",
            }
        )
        db_api.add_message(
            message_id=102,
            chat_id=-1001,
            author_id=456,
            content="I think the new QWEN model is very powerful.",
            embedding=[0.8, 0.7, 0.1] * 512,  # Dummy embedding
            message_type="text",
        )

        # 3. Simulate a user sending a search query to the bot (triggers Retriever)
        bot.add_tracked_chat(-1001)
        user_query = "what are people saying about the models?"
        bot.receive_search_query(user_query)

        # The Retriever would get the query, create an embedding, and use the DB API
        query_embedding = [0.7, 0.6, 0.2] * 512  # Dummy embedding for the query
        logger.info(f"Searching DB for query: '{user_query}'")
        candidates = db_api.search_similar(
            query_embedding=query_embedding, top_k=1, allowed_chat_ids=list(bot.tracked_chats)
        )
        logger.info(f"Received candidates: {candidates}")

        # 4. Retriever processes candidates and sends to Generator, then to Bot
        if candidates:
            final_summary = (
                f"Based on the conversation, the key point about models is: '{candidates[0]['metadata']['content']}'"
            )
            bot.send_answer_to_user(final_summary)

    except Exception:
        logger.exception("\nCould not run example. Please ensure ChromaDB is running and accessible.")
        raise
