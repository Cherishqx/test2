from typing import List, Mapping

class ConversationHistory:
    def __init__(self, max_history_size: int = 10):
        """
        初始化对话历史记录。
        :param max_history_size: 最大历史记录条目数，默认为10。
        """
        self.history: List[Mapping[str, str]] = []
        self.max_history_size = max_history_size

    def add_history(self, role: str, content: str) -> None:
        """
        添加一条对话记录到历史中。
        :param role: 角色（如 "user" 或 "assistant"）。
        :param content: 对话内容。
        """
        self.history.append({"role": role, "content": content})
        self.manage_history()

    def manage_history(self) -> None:
        """
        管理历史记录，确保不超过最大限制。
        """
        while len(self.history) > self.max_history_size:
            self.history.pop(0)

    def get_history(self) -> List[Mapping[str, str]]:
        """
        获取当前的对话历史记录。
        :return: 对话历史记录列表。
        """
        return self.history


# 示例用法
if __name__ == '__main__':
    conversation = ConversationHistory(max_history_size=5)
    conversation.add_history("user", "Hello!")
    conversation.add_history("assistant", "Hi there!")
    conversation.add_history("user", "How are you?")
    conversation.add_history("assistant", "I'm good, thanks!")

    print("Current History:", conversation.get_history())
