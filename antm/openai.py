import pandas as pd
from typing import Mapping, List, Any


DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the worst food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"""

DEFAULT_CHAT_PROMPT = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""

from typing import Mapping, Any, List, Optional
import pandas as pd

class OpenAI:
    def __init__(
        self,
        client,
        model: str = "text-embedding-3-small",
        prompt: Optional[str] = None,
        generator_kwargs: Optional[Mapping[str, Any]] = None,
        chat: bool = False,
        nr_docs: int = 4,
    ):
        self.client = client
        self.model = model
        self.chat = chat
        self.nr_docs = nr_docs

        self.default_prompt_ = DEFAULT_CHAT_PROMPT if chat else DEFAULT_PROMPT
        self.prompt = prompt or self.default_prompt_

        self.generator_kwargs = generator_kwargs or {}
        self.model = self.generator_kwargs.pop("model", self.model)
        self.generator_kwargs.pop("prompt", None)
        if "stop" not in self.generator_kwargs and not chat:
            self.generator_kwargs["stop"] = "\n"

    def extract_topics(
        self, documents: pd.DataFrame, keywords: List[str]
    ) -> str:
        prompt = self._create_prompt(documents, keywords)

        if self.chat:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **self.generator_kwargs
            )

            content = getattr(response.choices[0].message, "content", "").strip()
            label = content.replace("topic: ", "") if content else "No label returned"
        else:
            response = self.client.completions.create(
                model=self.model, prompt=prompt, **self.generator_kwargs
            )
            label = response.choices[0].text.strip()

        return label

    def _create_prompt(self, docs: pd.DataFrame, keywords: List[str]) -> str:
        prompt = self.prompt
        if "[KEYWORDS]" in prompt:
            prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
        if "[DOCUMENTS]" in prompt:
            prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt: str, docs: pd.DataFrame) -> str:
        to_replace = "\n".join(f"- {doc}" for doc in docs)
        return prompt.replace("[DOCUMENTS]", to_replace)