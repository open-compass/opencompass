INSTRUCTIONS_DIRECT_QA = {
    'en': 'You are a language expert specialized in Hungarian. Given a Hungarian phrase:\n\n' +
    '######################\n' +
    'Hungarian Phrase:\n' +
    '----------------------\n' +
    "'{hu_text}'\n" +
    '######################\n\n' +
    'and a context using this phrase:\n\n' +
    '######################\n' +
    'Hungarian Context:\n' +
    '----------------------\n' +
    '{context}\n' +
    '######################\n\n' +
    'What does the person mean by using this phrase? Please select one correct answer from the following two options:\n\n' +
    '######################\n' +
    'Options:\n' +
    '----------------------\n' +
    'Option 1: {option1}\n' +
    'Option 2: {option2}\n' +
    '######################\n\n' +
    "You should only answer the option number, '1' or '2'. Do not output any other content other than the option number. Your answer:"
}

INSTRUCTIONS_OE_DIR_QA = {
    'en': 'You are a language expert specialized in Hungarian. Given a Hungarian phrase:\n\n' +
    '######################\n' +
    'Hungarian Phrase:\n' +
    '----------------------\n' +
    "'{hu_text}'\n" +
    '######################\n\n' +
    'and a context using this phrase:\n\n' +
    '######################\n' +
    'Hungarian Context:\n' +
    '----------------------\n' +
    '{context}\n' +
    '######################\n\n' +
    'What does the person mean by using this phrase? Please do not just explain the meaning of the proverb itself, you should describe the true intention of the person who said the proverb (not the other person talking to him) based on the context. Please answer concisely in one sentence:',
    'hu': 'Ön magyar nyelvi szakértő. Adott egy magyar kifejezés:\n\n' +
    '######################\n' +
    'Magyar kifejezés:\n' +
    '----------------------\n' +
    "'{hu_text}'\n" +
    '######################\n\n' +
    'és egy szövegkörnyezet, amely ezt a kifejezést használja:\n\n' +
    '######################\n' +
    'Magyar kontextus:\n' +
    '----------------------\n' +
    '{context}\n' +
    '######################\n\n' +
    'Mire gondol az illető, amikor ezt a kifejezést használja? Kérjük, ne csak magának a közmondásnak a jelentését magyarázza meg, hanem a szövegkörnyezet alapján írja le a közmondást kimondó személy (nem a vele beszélgető másik személy) valódi szándékát. Kérjük, válaszoljon tömören, egy mondatban:'
}

judge_prompt_template = {
    'en_system':
    "Please act as an impartial judge specialized in Hungarian language and culture. Given a Hungarian saying, a context using that saying, and two analyses explaining 'what does the person mean by using that saying in the context?', please decide whether the given two analyses express the same meaning. If they reflect the same understanding of the saying's meaning, you should answer YES. If they are based on different interpretations of the saying, you should answer NO. Do not output anything other than 'YES' or 'NO'. Avoid any position biases and ensure that the order in which the analyses were presented does not influence your decision. Do not allow the length of the analyses to influence your judge, focus on their core meanings and their understandings of the Hungarian saying.",
    'en_user':
    '[The start of Hungarian saying]\n' +
    '{proverb}\n' +
    '[The end of Hungarian saying]\n\n' +
    '[The start of the context]\n' +
    '{conversation}\n' +
    '[The end of the context]\n\n' +
    '[The start of the first analysis]\n' +
    '{answer}\n' +
    '[The end of the first analysis]\n\n' +
    '[The start of the second analysis]\n' +
    '{raw_pred}\n'+
    '[The end of the second analysis]\n\n' +
    'Your decision:'
}
