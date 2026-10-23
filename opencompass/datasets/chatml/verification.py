# flake8: noqa
from typing import List, Literal, Union

from pydantic import BaseModel, model_validator


class TextItem(BaseModel):
    type: Literal['text']
    text: str


class ImageItem(BaseModel):
    type: Literal['image']
    image_url: str


MultimodalItem = Union[TextItem, ImageItem]


class SystemMessage(BaseModel):
    role: Literal['system']
    content: str


class AssistantMessage(BaseModel):
    role: Literal['assistant']
    content: str


class UserMessage(BaseModel):
    role: Literal['user']
    content: Union[str, List[MultimodalItem]]


Message = Union[SystemMessage, UserMessage, AssistantMessage]


class VerifyDataset(BaseModel):
    question: List[Message]
    answer: List[str]

    @model_validator(mode='after')
    def validate_answer_length(self) -> 'VerifyDataset':
        user_count = sum(1 for item in self.question
                         if hasattr(item, 'role') and item.role == 'user')

        if len(self.answer) != 1 and len(self.answer) != user_count:
            raise ValueError(
                f'Answer must have length 1 or {user_count} (number of user messages),'
                f'but got {len(self.answer)}')
        return self
