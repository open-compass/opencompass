# flake8: noqa: E501
SYSTEM_TEMPLATE = '{instruction}\n\n{examples}\n\n{post_prompt}'
USER_TEMPLATE = '<context>\n{context}\n</context>\n\nQuestion: {question}'
DEFAULT_TEMPLATE = f'{SYSTEM_TEMPLATE}\n\n{USER_TEMPLATE}'

CUSTOM_SYSTEM_PROMPTS = {
    # https://github.com/dvlab-research/LongLoRA/blob/2345c6d030f61ac3a031906386a103a5b05e0e6f/inference.py#L18
    'LONGLORA_LLAMA2':
    'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, explain why instead of answering '
    'something not correct. If you don\'t know the answer to a question, please don\'t share false information.'
}


def get_formatted_input(
    context,
    question,
    examples,
    instruction,
    post_prompt,
    template=DEFAULT_TEMPLATE,
):
    # pre_prompt - general instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    formatted_input = template.format(
        instruction=instruction,
        examples=examples,
        post_prompt=post_prompt,
        context=context.strip(),
        question=question,
    )
    return formatted_input.strip()


DEFAULT_PROMPTS = {
    'qa1': {
        'instruction':
        'I will give you context with the facts about positions of different persons hidden in some random text '
        'and a question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location to answer the question.',
        'examples':
        '<example>\n'
        'Charlie went to the hallway. Judith come back to the kitchen. Charlie travelled to balcony. '
        'Where is Charlie?\n'
        'Answer: The most recent location of Charlie is balcony.\n'
        '</example>\n\n'
        '<example>\n'
        'Alan moved to the garage. Charlie went to the beach. Alan went to the shop. Rouse '
        'travelled to balcony. Where is Alan?\n'
        'Answer: The most recent location of Alan is shop.\n'
        '</example>',
        'post_prompt':
        'Always return your answer in the following format: '
        'The most recent location of ’person’ is ’location’. Do not write anything else after that.',
    },
    'qa2': {
        'instruction':
        'I give you context with the facts about locations and actions of different persons '
        'hidden in some random text and a question.'
        'You need to answer the question based only on the information from the facts.\n'
        'If a person got an item in the first location and travelled to the second location '
        'the item is also in the second location. '
        'If a person dropped an item in the first location and moved to the second location '
        'the item remains in the first location.',
        'examples':
        '<example>\n'
        'Charlie went to the kitchen. Charlie got a bottle. Charlie moved to the balcony. '
        'Where is the bottle?\n'
        'Answer: The bottle is in the balcony.\n'
        '</example>\n'
        '<example>\n'
        'Alan moved to the garage. Alan got a screw driver. Alan moved to the kitchen. Where '
        'is the screw driver?\n'
        'Answer: The screw driver is in the kitchen.\n'
        '</example>',
        'post_prompt':
        'Always return your answer in the following format: The ’item’ is in ’location’. '
        'Do not write anything else after that.',
    },
    'qa3': {
        'instruction':
        'I give you context with the facts about locations and actions of different persons '
        'hidden in some random text and a question. '
        'You need to answer the question based only on the information from the facts.\n'
        'If a person got an item in the first location and travelled to the second location '
        'the item is also in the second location. '
        'If a person dropped an item in the first location and moved to the second location '
        'the item remains in the first location.',
        'examples':
        '<example>\n'
        'John journeyed to the bedroom. Mary grabbed the apple. Mary went back to the bathroom. '
        'Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. '
        'Where was the apple before the kitchen?\n'
        'Answer: Before the kitchen the apple was in the bathroom.\n'
        '</example>\n'
        '<example>\n'
        'John went back to the bedroom. John went back to the garden. John went back to the kitchen. '
        'Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. '
        'Where was the football before the bedroom?\n'
        'Answer: Before the bedroom the football was in the garden.\n'
        '</example>',
        'post_prompt':
        'Always return your answer in the following format: '
        'Before the $location_1$ the $item$ was in the $location_2$. Do not write anything else after that.',
    },
    'qa4': {
        'instruction':
        'I will give you context with the facts about different people, their location and actions, hidden in '
        'some random text and a question. '
        'You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'The hallway is south of the kitchen. The bedroom is north of the kitchen. '
        'What is the kitchen south of?\n'
        'Answer: bedroom\n'
        '</example>\n'
        '<example>\n'
        'The garden is west of the bedroom. The bedroom is west of the kitchen. What is west of the bedroom?\n'
        'Answer: garden\n'
        '</example>',
        'post_prompt':
        'Your answer should contain only one word - location. Do not write anything else after that.',
    },
    'qa5': {
        'instruction':
        'I will give you context with the facts about locations and their relations hidden in some random text '
        'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'Mary picked up the apple there. Mary gave the apple to Fred. Mary moved to the bedroom. '
        'Bill took the milk there. Who did Mary give the apple to?\n'
        'Answer: Fred\n'
        '</example>\n'
        '<example>\n'
        'Jeff took the football there. Jeff passed the football to Fred. Jeff got the milk there. '
        'Bill travelled to the bedroom. Who gave the football?\n'
        'Answer: Jeff\n'
        '</example>\n'
        '<example>\n'
        'Fred picked up the apple there. Fred handed the apple to Bill. Bill journeyed to the bedroom. '
        'Jeff went back to the garden. What did Fred give to Bill?\n'
        'Answer: apple\n'
        '</example>',
        'post_prompt':
        'Your answer should contain only one word. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa6': {
        'instruction':
        'I will give you context with the facts about people and their locations hidden in some random text and a '
        'question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'John travelled to the hallway. John travelled to the garden. Is John in the garden?\n'
        'Answer: yes\n'
        '</example>\n'
        '<example>\n'
        'Mary went to the office. Daniel journeyed to the hallway. Mary went to the bedroom. '
        'Sandra went to the garden. Is Mary in the office?\n'
        'Answer: no\n'
        '</example>\n',
        'post_prompt':
        'Your answer should contain only one word - $yes$ or $no$. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa7': {
        'instruction':
        'I will give you context with the facts about people and objects they carry, hidden in some random text '
        'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'Daniel went to the bedroom. Daniel got the apple there. How many objects is Daniel carrying?\n'
        'Answer: one\n'
        '</example>\n'
        '<example>\n'
        'Mary grabbed the apple there. Mary gave the apple to John. How many objects is Mary carrying?\n'
        'Answer: none\n'
        '</example>\n'
        '<example>\n'
        'Sandra travelled to the hallway. Sandra picked up the milk there. Sandra took the apple there. '
        'Mary travelled to the garden. How many objects is Sandra carrying?\n'
        'Answer: two\n'
        '</example>\n',
        'post_prompt':
        'Your answer should contain only one word - $none$ or $number_of_objects$. '
        'Do not write anything else after that. Do not explain your answer.',
    },
    'qa8': {
        'instruction':
        'I will give you context with the facts about people and objects they carry, hidden in some random text '
        'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'Sandra travelled to the garden. Mary grabbed the milk there. What is Mary carrying?\n'
        'Answer: milk\n'
        '</example>\n'
        '<example>\n'
        'Mary travelled to the kitchen. Sandra travelled to the office. John travelled to the office. '
        'Sandra discarded the milk there. What is Sandra carrying?\n'
        'Answer: nothing\n'
        '</example>\n'
        '<example>\n'
        'Daniel grabbed the apple there. Mary went to the office. Daniel moved to the garden. '
        'Daniel grabbed the milk there. Mary went to the kitchen. What is Daniel carrying?\n'
        'Answer: apple,milk\n'
        '</example>\n',
        'post_prompt':
        'Your answer should contain only one or two words: $nothing$ or $object$ or $object_1$, $object_2$. '
        'Do not write anything else. Do not explain your answer.',
    },
    'qa9': {
        'instruction':
        'I will give you context with the facts about people and their locations hidden in some random text and '
        'a question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'John is not in the bathroom. Sandra is not in the bedroom. Is John in the bathroom?\n'
        'Answer: no\n'
        '</example>\n'
        '<example>\n'
        'Mary journeyed to the kitchen. John is in the bedroom. Sandra is not in the garden. '
        'Is Mary in the kitchen?\n'
        'Answer: yes\n'
        '</example>\n',
        'post_prompt':
        'Your answer should contain only one word - $yes$ or $no$. Do not write anything else. '
        'Do not explain your answer.',
    },
    'qa10': {
        'instruction':
        'I will give you context with the facts about people and their locations hidden in some random text and a '
        'question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'Bill is in the kitchen. Julie is either in the school or the cinema. Is Bill in the bedroom?\n'
        'Answer: no\n'
        '</example>\n'
        '<example>\n'
        'Fred is in the bedroom. Mary is either in the school or the cinema. Is Mary in the school?\n'
        'Answer: maybe\n'
        '</example>\n'
        '<example>\n'
        'Fred is either in the kitchen or the park. Bill moved to the cinema. Is Bill in the cinema?\n'
        'Answer: yes\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - $yes$ or $no$ or $maybe$. Do not write anything else. '
        'Do not explain your answer.',
    },
    'qa11': {
        'instruction':
        'I will give you context with the facts about people and their locations hidden in some random text and a '
        'question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'Daniel journeyed to the hallway. After that he journeyed to the garden. Where is Daniel?\n'
        'Answer: garden\n'
        '</example>\n'
        '<example>\n'
        'Mary moved to the office. Afterwards she journeyed to the kitchen. Daniel went to the hallway. '
        'Then he journeyed to the garden. Where is Mary?\n'
        'Answer: kitchen\n'
        '</example>\n'
        '<example>\n'
        'Sandra moved to the kitchen. After that she went back to the hallway. Sandra moved to the bedroom. '
        'Then she went to the hallway. Mary moved to the bedroom. Afterwards she travelled to the bathroom. '
        'Where is Sandra?\n'
        'Answer: hallway\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - location. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa12': {
        'instruction':
        'I will give you context with the facts about people and their locations hidden in some random text and a '
        'question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'Mary and Daniel travelled to the bathroom. John and Daniel travelled to the office. Where is Daniel?\n'
        'Answer: office\n'
        '</example>\n'
        '<example>\n'
        'Sandra and Mary went back to the office. Daniel and Sandra went to the bedroom. Sandra and Mary travelled to the hallway. '
        'John and Mary went to the kitchen. Where is Mary?\n'
        'Answer: kitchen\n'
        '</example>\n'
        '<example>\n'
        'Daniel and Sandra went back to the hallway. Daniel and John moved to the office. Daniel and John moved to the garden. '
        'Daniel and Mary went back to the bathroom. Daniel and John went back to the kitchen. Daniel and Sandra went to the bathroom. '
        'Where is John?\n'
        'Answer: kitchen\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - location. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa13': {
        'instruction':
        'I will give you context with the facts about people and their locations hidden in some random text and a '
        'question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'Mary and Daniel travelled to the bathroom. Then they journeyed to the hallway. Where is Daniel?\n'
        'Answer: hallway\n'
        '</example>\n'
        '<example>\n'
        'Daniel and Sandra travelled to the kitchen. After that they journeyed to the hallway. Mary and Daniel travelled to the bedroom. '
        'After that they travelled to the hallway. Where is Sandra?\n'
        'Answer: hallway\n'
        '</example>\n'
        '<example>\n'
        'John and Mary moved to the bathroom. Then they travelled to the office. John and Mary went to the kitchen. '
        'Afterwards they went to the bedroom. John and Sandra moved to the bathroom. Following that they went back to the kitchen. '
        'Where is Mary?\n'
        'Answer: bedroom\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - location. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa14': {
        'instruction':
        'I will give you context with the facts about people and their locations hidden in some random text and a '
        'question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'Bill went back to the cinema yesterday. Julie went to the school this morning. Fred went to the park yesterday. '
        'Yesterday Julie went to the office. Where was Julie before the school?\n'
        'Answer: office\n'
        '</example>\n'
        '<example>\n'
        'This morning Fred went to the kitchen. Fred journeyed to the bedroom yesterday. Mary travelled to the bedroom this morning. '
        'Yesterday Mary went to the cinema. Where was Mary before the bedroom?\n'
        'Answer: cinema\n'
        '</example>\n'
        '<example>\n'
        'Yesterday Julie went back to the park. Julie went to the bedroom this morning. Bill journeyed to the cinema yesterday. '
        'This morning Bill went back to the park. This evening Julie went to the school. This afternoon Julie went back to the park. '
        'Where was Julie before the bedroom?\n'
        'Answer: park\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - location. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa15': {
        'instruction':
        'I will give you context with the facts about animals, their names and relations. The facts and a question '
        'are hidden in some random text. You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'Mice are afraid of wolves. Gertrude is a mouse. Cats are afraid of sheep. '
        'Winona is a mouse. Sheep are afraid of wolves. Emily is a mouse. Jessica is a wolf. '
        'What is gertrude afraid of?\n'
        'Answer: wolf\n'
        '</example>\n'
        '<example>\n'
        'Mice are afraid of wolves. Gertrude is a mouse. Cats are afraid of sheep. '
        'Winona is a mouse. Sheep are afraid of wolves. Emily is a mouse. Jessica is a wolf. '
        'What is jessica afraid of?\n'
        'Answer: cat\n'
        '</example>\n'
        '<example>\n'
        'Mice are afraid of cats. Wolves are afraid of sheep. Emily is a wolf. '
        'Cats are afraid of sheep. Gertrude is a wolf. Sheep are afraid of cats. Winona is a wolf. '
        'What is emily afraid of?\n'
        'Answer: sheep\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - an animal species. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa16': {
        'instruction':
        'I will give you context with the facts about animals, their names and colors. The facts and a question '
        'are hidden in some random text. You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'Lily is a frog. Bernhard is a frog. Bernhard is green. Brian is a lion. Brian is white. '
        'Julius is a swan. Julius is green. Lily is green. Greg is a swan. What color is Greg?\n'
        'Answer: green\n'
        '</example>\n'
        '<example>\n'
        'Julius is a lion. Lily is a rhino. Bernhard is a swan. Lily is white. Bernhard is green. '
        'Greg is a rhino. Greg is gray. Julius is white. Brian is a lion. What color is Brian?\n'
        'Answer: white\n'
        '</example>\n'
        '<example>\n'
        'Brian is a rhino. Julius is a lion. Bernhard is a lion. Greg is a swan. Brian is gray. '
        'Greg is white. Lily is a rhino. Bernhard is yellow. Lily is gray. What color is Julius?\n'
        'Answer: yellow\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - a color. Do not write anything else after that. '
        'Do not explain your answer.',
    },
    'qa17': {
        'instruction':
        'I will give you context with the facts about different figures, their location and colors, hidden in '
        'some random text and a question. '
        'You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'The triangle is above the pink rectangle. The blue square is to the left of the triangle. '
        'Is the pink rectangle to the right of the blue square?\n'
        'Answer: yes\n'
        '</example>\n'
        '<example>\n'
        'The red sphere is to the left of the yellow square. The red sphere is below the pink rectangle. '
        'Is the pink rectangle to the left of the yellow square?\n'
        'Answer: yes\n'
        '</example>'
        '<example>\n'
        'The red sphere is above the pink rectangle. The red sphere is to the right of the red square. '
        'Is the pink rectangle above the red square?\n'
        'Answer: no\n'
        '</example>',
        'post_prompt':
        'Your answer should contain only one word - $yes$ or $no$. Do not write anything else. '
        'Do not explain your answer.',
    },
    'qa18': {
        'instruction':
        'I will give you context with the facts about different objects and their sizes, hidden in '
        'some random text and a question. '
        'You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'The box of chocolates fits inside the chest. The box is bigger than the chest. The box is bigger than the suitcase. '
        'The suitcase fits inside the box. The container is bigger than the box of chocolates. Does the box fit in the box of chocolates?\n'
        'Answer: no\n'
        '</example>\n'
        '<example>\n'
        'The suitcase is bigger than the container. The container fits inside the box. The chest is bigger than the chocolate.'
        'The suitcase fits inside the box. The chest fits inside the box. Does the chocolate fit in the box?\n'
        'Answer: yes\n'
        '</example>'
        '<example>\n'
        'The chocolate fits inside the box of chocolates. The suitcase fits inside the box. The chocolate fits inside the box. '
        'The box is bigger than the box of chocolates. The suitcase is bigger than the box of chocolates. Is the chocolate bigger than the box?\n'
        'Answer: no\n'
        '</example>',
        'post_prompt':
        'Your answer should contain only one word - $yes$ or $no$. Do not write anything else. '
        'Do not explain your answer.',
    },
    'qa19': {
        'instruction':
        'I will give you context with the facts about different places and their locations, hidden in '
        'some random text and a question. '
        'You need to answer the question based only on the information from the facts.',
        'examples':
        '<example>\n'
        'The office is east of the hallway. The kitchen is north of the office. The garden is west of the bedroom. '
        'The office is west of the garden. The bathroom is north of the garden. How do you go from the kitchen to the garden?\n'
        'Answer: s,e\n'
        '</example>\n'
        '<example>\n'
        'The bedroom is west of the hallway. The office is east of the garden. The garden is north of the kitchen. '
        'The kitchen is north of the bathroom. The hallway is west of the garden. How do you go from the kitchen to the hallway?\n'
        'Answer: n,w\n'
        '</example>\n'
        '<example>\n'
        'The bedroom is south of the hallway. The bathroom is east of the office. The kitchen is west of the garden. '
        'The garden is south of the office. The office is south of the bedroom. How do you go from the garden to the bedroom?\n'
        'Answer: n,n\n'
        '</example>\n',
        'post_prompt':
        'Your answer should contain only two letters, separated by a comma - ordinal directions. You can choose the letters from '
        '$n$, $s$, $e$ and $w$. Do not write anything else after that.',
    },
    'qa20': {
        'instruction':
        'I will give you context with the facts about people, their locations and condition hidden in some random text and a '
        'question. You need to answer the question based only on the information from the facts. '
        'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
        '<example>\n'
        'Sumit is tired. Where will sumit go?\n'
        'Answer: bedroom\n'
        '</example>\n'
        '<example>\n'
        'Yann is hungry. Yann journeyed to the kitchen. Why did yann go to the kitchen?\n'
        'Answer: hungry\n'
        '</example>\n'
        '<example>\n'
        'Antoine is thirsty. Yann is tired. Yann went back to the bedroom. Yann picked up the pajamas there.'
        'Jason is thirsty. Antoine went back to the kitchen. Why did antoine go to the kitchen?\n'
        'Answer: thirsty\n'
        '</example>\n'
        '<context>\n',
        'post_prompt':
        'Your answer should contain only one word - a person condition or a place. Do not write anything else after that. '
        'Do not explain your answer.',
    },
}
