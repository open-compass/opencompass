from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMMLUDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


mmmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D','subject'],
    output_column='target',
    train_split='test')

mmmlu_all_sets = [
    'mmlu_AR-XY',
    'mmlu_BN-BD',
    'mmlu_DE-DE',
    'mmlu_ES-LA',
    'mmlu_FR-FR',
    'mmlu_HI-IN',
    'mmlu_ID-ID',
    'mmlu_IT-IT',
    'mmlu_JA-JP',
    'mmlu_KO-KR',
    'mmlu_PT-BR',
    'mmlu_SW-KE',
    'mmlu_YO-NG',
    'mmlu_ZH-CN',
]

mmmlu_datasets = []
for _name in mmmlu_all_sets:
    if 'AR' in _name:
        _hint = f'هناك سؤال اختيار واحد. أجب عن السؤال بالرد على A أو B أو C أو D.'
        _prompt = f'يتعلق بـ {{subject}} \nالسؤال: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nالإجابة:'
        _round = [
                dict(role='HUMAN', prompt="هناك سؤال اختيار من متعدد. أجب عن السؤال بالرد A أو B أو C أو D.\nيتعلق بـ الجبر المجرد\nالسؤال: ابحث عن أقصى حد ممكن لترتيب بعض العناصر في Z_4 x Z_6.\n A.4\nB.6\nC.12\nD.24\nلنفكر خطوة بخطوة\nالإجابة:"),
                dict(role='BOT', prompt='C'),
                dict(role='HUMAN', prompt="هناك سؤال اختيار من متعدد. أجب عن السؤال بالرد A أو B أو C أو D.\nيتعلق بـ الجغرافيا في المدرسة الثانوية\nالسؤال: أي من الأديان أدناه هو دين عالمي؟ A. الطاوية\n B. الإسلام\n C. الشنتو\n D. الكونفوشيوسية\nلنفكر خطوة بخطوة\nالإجابة:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="هناك سؤال اختيار من متعدد. أجب عن السؤال بالرد A أو B أو C أو D.\nيتعلق بـ تعلم الآلة\nالسؤال: في كعكة يان لوكون، الكرز في الأعلى هو: \nA. التعلم المعزز\nB. التعلم الذاتي المراقب\nC. التعلم غير المراقب\nD. التعلم المراقب\nلنفكر خطوة بخطوة\nالإجابة:"),
                dict(role='BOT', prompt="A"),
                dict(role='HUMAN', prompt="هناك سؤال اختيار من متعدد. أجب عن السؤال بالرد A أو B أو C أو D.\nيتعلق بـ الفلسفة\nالسؤال: يقترح سقراط أن المقدس هو جزء واحد من:\nA. ما هو حكيم.\nB. ما هو عادل.\nC. ما هو جميل.\nD. ما هو قانوني.\nلنفكر خطوة بخطوة\nالإجابة:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="هذه سؤال اختيار واحد. أجب عن السؤال بالرد A أو B أو C أو D.\nيتعلق الأمر بتاريخ الولايات المتحدة في المدارس الثانوية.\nسؤال: هذه السؤال يشير إلى المعلومات التالية. ربما، مع ذلك، أنا أكثر وعيًا بأهمية الحريات المدنية في هذه اللحظة المحددة من تاريخنا من أي شخص آخر، لأنني أسافر عبر البلاد وألتقي بالناس وأرى أشياء حدثت لأناس عاديين، أدرك ما يعنيه للديمقراطية الحفاظ على حرياتنا المدنية. طوال السنوات كان علينا أن نقاتل من أجل الحرية المدنية، ونعلم أن هناك أوقاتًا تصبح فيها الأضواء خافتة، وكلما حدث ذلك تكون الديمقراطية في خطر. الآن، إلى حد كبير بسبب الحالة المضطربة للعالم ككل، اختفت الحريات المدنية في العديد من البلدان الأخرى. من المستحيل، بالطبع، أن تكون في حالة حرب وأن تحافظ على حرية الصحافة وحرية التعبير وحرية التجمع. إنها تختفي تلقائيًا. وهكذا في العديد من البلدان التي كانت آمنة عادة، أصبحت اليوم قد اختفت. في بلدان أخرى، حتى قبل أن تأتي الحرب، لم تختف فقط حرية الصحافة وحرية التجمع وحرية التعبير، بل اختفت أيضًا حرية الدين. ولذلك، نحن هنا في هذا البلد، لدينا مسؤولية كبيرة. نحن في السلام. ليس لدينا سبب من المخاوف التي تتحكم في العديد من الشعوب الأخرى في جميع أنحاء العالم؛ لذلك يجب علينا أن نحافظ على حريات الديمقراطية. —إلينور روزفلت، خطاب إلى الاتحاد الأمريكي للحريات المدنية، شيكاغو، إلينوي، 14 مارس 1940.\nفي خطابها، أشارت إلينور روزفلت إلى التهديد السابق للحريات المدنية الذي أنشأته أي مما يلي؟\nA. الحرب العالمية الأولى\nB. الصفقة الجديدة\nC. الحرب الباردة\nD. الكساد العظيم\nدعونا نفكر خطوة بخطوة.\nالجواب:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"\nفقط يجب الرد على الخيار A أو B أو C أو D. \nالإجابة هي:"),
            ]
    elif 'BN' in _name:
        _hint = f'এটি একটি একক পছন্দের প্রশ্ন। এ, বি, সি বা ডি উত্তর দিয়ে প্রশ্নের উত্তর দিন।'
        _prompt = f'এটি {{subject}} সম্পর্কে \nপ্রশ্ন: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nউত্তর:'
        _round = [
                dict(role='HUMAN', prompt="এটি একটি একটি বিকল্প প্রশ্ন। A, B, C অথবা D এর মাধ্যমে উত্তর দিন।\nএটি মেশিন লার্নিং সম্পর্কে\nপ্রশ্ন: ইয়ান লেকুনের কেকের উপর চেরি হল: \nA. শক্তিশালীকরণ শেখা\nB. স্ব-নিরীক্ষিত শেখা\nC. অ-নিরীক্ষিত শেখা\nD. নিরীক্ষিত শেখা\nআসুন ধাপে ধাপে ভাবি\nউত্তর:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="এটি একটি একটি বিকল্প প্রশ্ন। A, B, C অথবা D এর মাধ্যমে উত্তর দিন।\nএটি বিমূর্ত বীজগণিত সম্পর্কে\nপ্রশ্ন: Z_4 x Z_6 এর কোন একটি উপাদানের জন্য সর্বাধিক সম্ভাব্য র‍্যাঙ্ক খুঁজুন।\nA.4\nB.6\nC.12\nD.24\nআসুন ধাপে ধাপে ভাবি\nউত্তর:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="এটি একটি একটি বিকল্প প্রশ্ন। A, B, C অথবা D এর মাধ্যমে উত্তর দিন।\nএটি উচ্চ বিদ্যালয়ের ভূগোল সম্পর্কে\nপ্রশ্ন: নিচের কোন ধর্ম একটি বিশ্বজনীন ধর্ম? \nA. তাওবাদ\nB. ইসলাম\nC. শিন্টোবাদ\nD. কনফুসিয়াসবাদ\nআসুন ধাপে ধাপে ভাবি\nউত্তর:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="এটি একটি একটি বিকল্প প্রশ্ন। A, B, C অথবা D এর মাধ্যমে উত্তর দিন।\nএটি দর্শনশাস্ত্র সম্পর্কে\nপ্রশ্ন: সক্রেটিস নির্দেশ করেন যে পবিত্র হচ্ছে:\nA. যা বিজ্ঞ\nB. যা ন্যায়িক\nC. যা সুন্দর\nD. যা আইনগত\nআসুন ধাপে ধাপে ভাবি\nউত্তর:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="এটি একটি একক পছন্দের প্রশ্ন। প্রশ্নের উত্তর A, B, C অথবা D দিয়ে দিন।\nএটি উচ্চ বিদ্যালয়ের মার্কিন ইতিহাস সম্পর্কে।\nপ্রশ্ন: এই প্রশ্নটি নিম্নলিখিত তথ্যের সাথে সম্পর্কিত। তবে, शायद আমি আমাদের ইতিহাসের এই নির্ভরযোগ্য মুহূর্তে নাগরিক স্বাধীনতার গুরুত্ব সম্পর্কে অন্য যে কারো চেয়ে বেশি সচেতন, কারণ আমি দেশজুড়ে ভ্রমণ করি এবং মানুষদের সঙ্গে দেখা করি এবং ছোট মানুষদের সাথে ঘটে যাওয়া ঘটনার কথা দেখি। আমি বুঝতে পারি যে আমাদের নাগরিক স্বাধীনতাগুলো রক্ষা করা কীভাবে গণতন্ত্রের জন্য গুরুত্বপূর্ণ। আমরা সাল জুড়ে নাগরিক স্বাধীনতার জন্য লড়াই করতে হয়েছে, এবং আমরা জানি যে এমন সময় আসে যখন আলো ধীরে ধীরে ম্লান হয়, এবং যখনই তা ঘটে, গণতন্ত্র বিপদে পড়ে। এখন, বিশালাংশে বিশ্বজুড়ে অস্থির পরিস্থিতির কারণে, অনেক অন্যান্য দেশে নাগরিক স্বাধীনতা হারিয়ে গেছে। যুদ্ধ চলাকালীন সংবাদপত্রের স্বাধীনতা, বক্তৃতার স্বাধীনতা এবং সমাবেশের স্বাধীনতা বজায় রাখা অবশ্যই অসম্ভব। সেগুলি স্বয়ংক্রিয়ভাবে消失 হয়ে যায়। এবং তাই বহু দেশে যেগুলি সাধারণত নিরাপদ ছিল, আজ তারা gone হয়ে গেছে। অন্যান্য দেশে, এমনকি যুদ্ধ আসার আগেই, শুধুমাত্র সংবাদপত্রের স্বাধীনতা, সমাবেশের স্বাধীনতা, এবং বক্তৃতার স্বাধীনতা হারিয়ে যায়নি, তবে ধর্মের স্বাধীনতাও消失 হয়ে গেছে। এবং তাই আমরা জানি যে এই দেশে আমাদের একটি গুরুতর দায়িত্ব আছে। আমরা শান্তিতে আছি। আমাদের কাছে বিশ্বের অনেক অন্যান্য লোকজনের নিয়ন্ত্রণ করা ভয়ের জন্য কোন কারণ নেই; অতএব, আমাদের গণতন্ত্রের স্বাধীনতাগুলোকে রক্ষা করতে হবে। —এলিনর রুজভেল্ট, আমেরিকান সিভিল লিবারটিজ ইউনিয়নের সম্বোধন, শিকাগো, ইলিনয়, ১৪ই মার্চ, ১৯৪০।\nতার বক্তৃতায়, এলিনর রুজভেল্ট পূর্ববর্তী নাগরিক স্বাধীনতার প্রতি হুমকি সম্পর্কে কোনটি উল্লেখ করেছেন?\nA. বিশ্বযুদ্ধ I\nB. নয়া চুক্তি\nC. ঠাণ্ডা যুদ্ধ\nD. গ্রেট ডিপ্রেশন\nধাপে ধাপে চিন্তা করি।\nউত্তর:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"শুধু বিকল্প A, B, C বা D এর উত্তর দিন, \nউত্তর হলো:"),
            ]
    elif 'DE' in _name:
        _hint = f'Es gibt eine Einzelwahlfrage. Beantworte die Frage, indem du A, B, C oder D antwortest.'
        _prompt = f'Es geht um {{subject}} \nFrage: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAntwort:'
        _round = [
                dict(role='HUMAN', prompt="Das ist eine einzelne Auswahlfrage. Beantworte die Frage, indem du A, B, C oder D antwortest.\nEs geht um maschinelles Lernen.\nFrage: In Yann LeCuns Kuchen ist die Kirsche oben:\nA. Verstärkendes Lernen\nB. Selbstüberwachtes Lernen\nC. Unüberwachtes Lernen\nD. Überwachtes Lernen\nLass uns Schritt für Schritt nachdenken.\nAntwort:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Das ist eine einzelne Auswahlfrage. Beantworte die Frage, indem du A, B, C oder D antwortest.\nEs geht um abstrakte Algebra.\nFrage: Finde die maximal mögliche Ordnung für ein Element von Z_4 x Z_6.\nA. 4\nB. 6\nC. 12\nD. 24\nLass uns Schritt für Schritt nachdenken.\nAntwort:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="Das ist eine einzelne Auswahlfrage. Beantworte die Frage, indem du A, B, C oder D antwortest.\nEs geht um Geografie in der High School.\nFrage: Welche der folgenden Religionen ist eine universalisierende Religion? \nA. Taoismus\nB. Islam\nC. Shintoismus\nD. Konfuzianismus\nLass uns Schritt für Schritt nachdenken.\nAntwort:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Das ist eine einzelne Auswahlfrage. Beantworte die Frage, indem du A, B, C oder D antwortest.\nEs geht um Philosophie.\nFrage: Sokrates schlägt vor, dass das Heilige ein Teil von:\nA. was weise ist.\nB. was gerecht ist.\nC. was schön ist.\nD. was legal ist.\nLass uns Schritt für Schritt nachdenken.\nAntwort:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="Dies ist eine Einzelwahlfrage. Beantworten Sie die Frage, indem Sie A, B, C oder D antworten.\nEs geht um die amerikanische Geschichte in der High School.\nFrage: Diese Frage bezieht sich auf die folgenden Informationen. Vielleicht bin ich mir jedoch in diesem bestimmten Moment unserer Geschichte mehr bewusst, wie wichtig die Bürgerrechte sind als jeder andere, weil ich durch das Land reise, Menschen treffe und Dinge sehe, die kleinen Menschen passiert sind. Ich erkenne, was es bedeutet, die Bürgerrechte zu bewahren, um die Demokratie zu erhalten. Im Verlauf der Jahre mussten wir für die Bürgerrechte kämpfen, und wir wissen, dass es Zeiten gibt, in denen das Licht eher schwach wird, und jedes Mal, wenn das passiert, ist die Demokratie in Gefahr. Jetzt, größtenteils aufgrund des angespannten Zustands der Welt als Ganzes, sind die Bürgerrechte in vielen anderen Ländern verschwunden. Es ist unmöglich, im Krieg zu sein und gleichzeitig die Pressefreiheit, die Meinungsfreiheit und die Versammlungsfreiheit aufrechtzuerhalten. Sie verschwinden automatisch. Und so sind in vielen Ländern, in denen sie normalerweise sicher waren, heute verschwunden. In anderen Ländern verschwanden nicht nur die Pressefreiheit und die Versammlungsfreiheit und die Meinungsfreiheit, sondern auch die Religionsfreiheit, selbst bevor der Krieg kam. Und so wissen wir hier in diesem Land, dass wir eine ernsthafte Verantwortung haben. Wir sind in Frieden. Wir haben keinen Grund für die Ängste, die so viele andere Menschen auf der ganzen Welt regieren; daher müssen wir die Freiheiten der Demokratie schützen. —Eleanor Roosevelt, Ansprache an die Amerikanische Vereinigung für Bürgerrechte, Chicago, Illinois, 14. März 1940.\nIn ihrer Rede spielte Eleanor Roosevelt auf die frühere Bedrohung der Bürgerrechte an, die durch welches der folgenden Dinge geschaffen wurde?\nA. Erster Weltkrieg\nB. New Deal\nC. Kalter Krieg\nD. Große Depression\nLassen Sie uns Schritt für Schritt nachdenken.\nAntwort:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Bitte beantworten Sie nur mit Option A, B, C oder D. \nAntwort:"),
            ]
    elif 'ES' in _name:
        _hint = f'Hay una pregunta de elección única. Responde a la pregunta respondiendo A, B, C o D.'
        _prompt = f'Se trata de {{subject}} \nPregunta: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRespuesta:'
        _round = [
                dict(role='HUMAN', prompt="Esta es una pregunta de opción única. Responde a la pregunta contestando A, B, C o D.\nSe trata de aprendizaje automático.\nPregunta: En el pastel de Yann LeCun, la cereza en la cima es: \nA. aprendizaje por refuerzo\nB. aprendizaje auto-supervisado\nC. aprendizaje no supervisado\nD. aprendizaje supervisado\nPensemos paso a paso.\nRespuesta:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Esta es una pregunta de opción única. Responde a la pregunta contestando A, B, C o D.\nSe trata de álgebra abstracta.\nPregunta: Encuentra el orden máximo posible para algún elemento de Z_4 x Z_6.\nA. 4\nB. 6\nC. 12\nD. 24\nPensemos paso a paso.\nRespuesta:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="Esta es una pregunta de opción única. Responde a la pregunta contestando A, B, C o D.\nSe trata de geografía de secundaria.\nPregunta: ¿Cuál de la siguiente religión es una religión universalizante? \nA. Taoísmo\nB. Islam\nC. Shintoísmo\nD. Confucianismo\nPensemos paso a paso.\nRespuesta:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Esta es una pregunta de opción única. Responde a la pregunta contestando A, B, C o D.\nSe trata de filosofía.\nPregunta: Sócrates sugiere que lo sagrado es una parte de:\nA. lo que es prudente.\nB. lo que es justo.\nC. lo que es bello.\nD. lo que es legal.\nPensemos paso a paso.\nRespuesta:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="Esta es una pregunta de opción única. Responde a la pregunta contestando A, B, C o D.\nSe trata de la historia de EE.UU. en la escuela secundaria.\nPregunta: Esta pregunta se refiere a la siguiente información. Sin embargo, quizás soy más consciente de la importancia de las libertades civiles en este momento particular de nuestra historia que cualquier otra persona, porque viajo por el país, encuentro personas y veo cosas que han sucedido a las personas pequeñas, me doy cuenta de lo que significa para la democracia preservar nuestras libertades civiles. A lo largo de los años hemos tenido que luchar por la libertad civil, y sabemos que hay momentos en que la luz se vuelve bastante tenue, y cada vez que eso sucede, la democracia está en peligro. Ahora, en gran parte debido al estado problemático del mundo en su conjunto, las libertades civiles han desaparecido en muchos otros países. Es imposible, por supuesto, estar en guerra y mantener la libertad de prensa, la libertad de expresión y la libertad de reunión. Desaparecen automáticamente. Y así, en muchos países donde normalmente eran seguras, hoy han desaparecido. En otros países, incluso antes de que llegara la guerra, no solo la libertad de prensa y la libertad de reunión, y la libertad de expresión desaparecieron, sino que también desapareció la libertad de religión. Y así sabemos que aquí en este país, tenemos una grave responsabilidad. Estamos en paz. No tenemos razón para los temores que gobiernan a tantas otras personas en todo el mundo; por lo tanto, tenemos que proteger las libertades de la democracia. —Eleanor Roosevelt, Discurso ante la Unión Americana de Libertades Civiles, Chicago, Illinois, 14 de marzo de 1940.\nEn su discurso, Eleanor Roosevelt aludió a la amenaza anterior a las libertades civiles creada por cuál de las siguientes?\nA. Primera Guerra Mundial\nB. El New Deal\nC. La Guerra Fría\nD. La Gran Depresión\nPensemos paso a paso.\nRespuesta:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Solo necesitas responder con la opción A, B, C o D. \nRespuesta:"),
            ]
    elif 'FR' in _name:
        _hint = f'Il y a une question à choix unique. Répondez à la question en répondant A, B, C ou D.'
        _prompt = f'''C'est à propos de {{subject}} \nQuestion : {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRéponse :'''
        _round = [
                dict(role='HUMAN', prompt="C'est une question à choix unique. Répondez à la question en répondant A, B, C ou D.\nIl s'agit d'apprentissage automatique.\nQuestion : Dans le gâteau de Yann LeCun, la cerise sur le dessus est :\nA. apprentissage par renforcement\nB. apprentissage auto-supervisé\nC. apprentissage non supervisé\nD. apprentissage supervisé\nPensons étape par étape.\nRéponse :"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="C'est une question à choix unique. Répondez à la question en répondant A, B, C ou D.\nIl s'agit d'algèbre abstraite.\nQuestion : Trouvez l'ordre maximum possible pour un élément de Z_4 x Z_6.\nA. 4\nB. 6\nC. 12\nD. 24\nPensons étape par étape.\nRéponse :"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="C'est une question à choix unique. Répondez à la question en répondant A, B, C ou D.\nIl s'agit de géographie de lycée.\nQuestion : Laquelle des religions suivantes est une religion universalisante ?\nA. Taoïsme\nB. Islam\nC. Shintoïsme\nD. Confucianisme\nPensons étape par étape.\nRéponse :"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="C'est une question à choix unique. Répondez à la question en répondant A, B, C ou D.\nIl s'agit de philosophie.\nQuestion : Socrate suggère que le sacré est une partie de :\nA. ce qui est prudent.\nB. ce qui est juste.\nC. ce qui est beau.\nD. ce qui est légal.\nPensons étape par étape.\nRéponse :"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="C'est une question à choix unique. Répondez à la question en répondant A, B, C ou D.\nC'est sur l'histoire des États-Unis au lycée.\nQuestion : Cette question se réfère aux informations suivantes. Peut-être, cependant, je suis plus conscient de l'importance des libertés civiles à ce moment particulier de notre histoire que quiconque, car je voyage à travers le pays, rencontre des gens et vois des choses qui sont arrivées à des petites personnes, je réalise ce que signifie pour la démocratie de préserver nos libertés civiles. Au fil des ans, nous avons dû nous battre pour la liberté civile, et nous savons qu'il y a des moments où la lumière devient plutôt faible, et chaque fois que cela se produit, la démocratie est en danger. Maintenant, en grande partie à cause de l'état troublé du monde dans son ensemble, les libertés civiles ont disparu dans de nombreux autres pays. Il est impossible, bien sûr, d'être en guerre et de maintenir la liberté de la presse, la liberté d'expression et la liberté de réunion. Elles disparaissent automatiquement. Et donc dans de nombreux pays où elles étaient normalement en sécurité, aujourd'hui, elles sont parties. Dans d'autres pays, même avant l'arrivée de la guerre, non seulement la liberté de la presse et la liberté de réunion, et la liberté d'expression ont disparu, mais la liberté de religion a aussi disparu. Et donc nous savons ici dans ce pays, nous avons une grave responsabilité. Nous sommes en paix. Nous n'avons aucune raison pour les peurs qui gouvernent tant d'autres peuples à travers le monde ; par conséquent, nous devons garder les libertés de la démocratie. —Eleanor Roosevelt, Adresse à l'Union Américaine pour les Libertés Civiles, Chicago, Illinois, 14 mars 1940\nDans son discours, Eleanor Roosevelt a fait allusion à la menace antérieure pour les libertés civiles créée par laquelle des suivantes ?\nA. Première Guerre Mondiale\nB. Le New Deal\nC. La Guerre froide\nD. La Grande Dépression\nPensons étape par étape.\nRéponse :"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Vous n'avez qu'à répondre avec l'option A, B, C ou D. \nRéponse :"),
            ]
    elif 'HI' in _name:
        _hint = f'Gen yon kesyon chwa sèl. Reponn kesyon an pa reponn A, B, C oswa D.'
        _prompt = f'Sa se sou {{subject}} \nKestyon: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRepons:'
        _round = [
                dict(role='HUMAN', prompt="Sa a se yon kesyon ak yon sèl chwa. Reponn kesyon an pa reyponn A, B, C oswa D.\nLi konsène aprantisaj otomatik.\nKesyon: Nan gato Yann LeCun, sèjan an ki sou tèt se:\nA. aprantisaj pa ranfòsman\nB. aprantisaj oto-supervise\nC. aprantisaj pa supervizè\nD. aprantisaj supervizè\nAnn reflechi etap pa etap.\nRepons:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Sa a se yon kesyon ak yon sèl chwa. Reponn kesyon an pa reyponn A, B, C oswa D.\nLi konsène aljèb abstre.\nKesyon: Jwenn lòd maksimòm posib pou kèk eleman nan Z_4 x Z_6.\nA. 4\nB. 6\nC. 12\nD. 24\nAnn réfléchy etap pa etap.\nRepons:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="Sa a se yon kesyon ak yon sèl chwa. Reponn kesyon an pa reyponn A, B, C oswa D.\nLi konsène jeyografi lekòl segondè.\nKesyon: Ki relijyon ki anba a se yon relijyon universalizant?\nA. Taoïs\nB. Islam\nC. Shintoïs\nD. Konfisyonnis\nAnn réfléchy etap pa etap.\nRepons:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Sa a se yon kesyon ak yon sèl chwa. Reponn kesyon an pa reyponn A, B, C oswa D.\nLi konsène filozofi.\nKesyon: Sokrates sijere ke sa ki sakre se yon pati nan :\nA. sa ki pridan.\nB. sa ki jis.\nC. sa ki bèl.\nD. sa ki legal.\nAnn réfléchy etap pa etap.\nRepons:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="Sa a se yon kesyon chwa sèl. Reponn kesyon an pa reponn A, B, C, oswa D.\nSa a se sou istwa Etazini nan lekòl segondè.\nKesyon: Kesyon sa a fè referans a enfòmasyon ki anba a. Sepandan, petèt mwen plis konsyan de enpòtans libète sivil nan moman patikilye sa a nan istwa nou an pase nenpòt lòt moun, paske mwen vwayaje atravè peyi a, mwen rankontre moun e mwen wè bagay ki te pase bay ti moun, mwen reyalize sa demokrasi vle di pou preserve libète sivil nou yo. Pandan tout ane yo, nou te oblije goumen pou libète sivil, e nou konnen gen moman kote limyè a vin fèb, e chak fwa sa rive, demokrasi nan danje. Kounye a, an gran pati akòz eta pwoblèm mond lan kòm yon antye, libète sivil yo te disparèt nan anpil lòt peyi. Li enposib, natirèlman, pou fè lagè epi kenbe libète laprès ak libète expresyon ak libète reyinyon. Yo disparèt otomatikman. E konsa nan anpil peyi kote yo te nòmalman an sekirite, jodi a yo ale. Nan lòt peyi, menm anvan lagè a te vini, pa sèlman libète laprès ak libète reyinyon, ak libète expresyon disparèt, men libète relijyon tou disparèt. E se konsa nou konnen isit la nan peyi sa a, nou gen yon responsablite grav. Nou nan lapè. Nou pa gen okenn rezon pou laperèz ki gouvène tèlman anpil lòt moun nan tout mond lan; kidonk, nou dwe kenbe libète demokrasi a. —Eleanor Roosevelt, Diskou bay Asosyasyon Ameriken pou Libète Sivil, Chicago, Illinois, 14 Mas 1940\nNan diskou li a, Eleanor Roosevelt te fè alizyon a menas anvan yo pou libète sivil ki te kreye pa ki sa nan sa ki swiv:\nA. Premye Gè Mondyal\nB. Nouvo Kontra\nC. Lagè frèt\nD. Gran Depresyon\nAnn panse etap pa etap.\nRepons:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Ou sèlman bezwen reponn ak opsyon A, B, C oswa D. \nRepons :"),
            ]
    elif 'ID' in _name:
        _hint = f'Ada pertanyaan pilihan tunggal. Jawablah pertanyaan dengan menjawab A, B, C, atau D.'
        _prompt = f'Ini tentang {{subject}} \nPertanyaan: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nJawaban:'
        _round = [
                dict(role='HUMAN', prompt="Ini adalah pertanyaan pilihan ganda. Jawablah pertanyaan ini dengan menjawab A, B, C, atau D.\nIni tentang pembelajaran mesin.\nPertanyaan: Dalam kue Yann LeCun, ceri di atas adalah:\nA. pembelajaran penguatan\nB. pembelajaran mandiri\nC. pembelajaran tak terawasi\nD. pembelajaran terawasi\nMari kita pikirkan langkah demi langkah.\nJawaban:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Ini adalah pertanyaan pilihan ganda. Jawablah pertanyaan ini dengan menjawab A, B, C, atau D.\nIni tentang aljabar abstrak.\nPertanyaan: Temukan urutan maksimum yang mungkin untuk beberapa elemen dari Z_4 x Z_6.\nA. 4\nB. 6\nC. 12\nD. 24\nMari kita pikirkan langkah demi langkah.\nJawaban:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="Ini adalah pertanyaan pilihan ganda. Jawablah pertanyaan ini dengan menjawab A, B, C, atau D.\nIni tentang geografi sekolah menengah.\nPertanyaan: Agama mana di bawah ini yang merupakan agama universal?\nA. Taoisme\nB. Islam\nC. Shintoisme\nD. Konfusianisme\nMari kita pikirkan langkah demi langkah.\nJawaban:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Ini adalah pertanyaan pilihan ganda. Jawablah pertanyaan ini dengan menjawab A, B, C, atau D.\nIni tentang filsafat.\nPertanyaan: Socrates menyarankan bahwa yang suci adalah salah satu bagian dari:\nA. apa yang bijak.\nB. apa yang adil.\nC. apa yang indah.\nD. apa yang legal.\nMari kita pikirkan langkah demi langkah.\nJawaban:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="Ini adalah pertanyaan pilihan ganda. Jawab pertanyaan dengan menjawab A, B, C, atau D.\nIni tentang sejarah AS di sekolah menengah.\nPertanyaan: Pertanyaan ini merujuk pada informasi berikut. Namun, mungkin saya lebih sadar akan pentingnya kebebasan sipil pada momen tertentu dalam sejarah kita daripada siapa pun, karena saya berkeliling negara dan bertemu orang-orang serta melihat hal-hal yang terjadi pada orang-orang kecil, saya menyadari apa artinya bagi demokrasi untuk memelihara kebebasan sipil kita. Selama bertahun-tahun kita harus berjuang untuk kebebasan sipil, dan kita tahu bahwa ada kalanya cahaya menjadi redup, dan setiap kali itu terjadi, demokrasi berada dalam bahaya. Sekarang, sebagian besar karena keadaan dunia yang bermasalah secara keseluruhan, kebebasan sipil telah menghilang di banyak negara lain. Tentu saja, adalah mustahil untuk berperang dan tetap mempertahankan kebebasan pers, kebebasan berbicara, dan kebebasan berkumpul. Mereka menghilang secara otomatis. Dan jadi di banyak negara di mana biasanya mereka aman, hari ini mereka sudah hilang. Di negara-negara lain, bahkan sebelum perang datang, tidak hanya kebebasan pers dan kebebasan berkumpul, serta kebebasan berbicara yang hilang, tetapi kebebasan beragama juga hilang. Dan jadi kami tahu di negara ini, kami memiliki tanggung jawab yang berat. Kami berada dalam keadaan damai. Kami tidak punya alasan untuk ketakutan yang mengatur begitu banyak orang di seluruh dunia; oleh karena itu, kami harus menjaga kebebasan demokrasi. —Eleanor Roosevelt, Pidato kepada Asosiasi Kebebasan Sipil Amerika, Chicago, Illinois, 14 Maret 1940\nDalam pidatonya, Eleanor Roosevelt merujuk pada ancaman sebelumnya terhadap kebebasan sipil yang diciptakan oleh mana di antara berikut ini?\nA. Perang Dunia I\nB. New Deal\nC. Perang Dingin\nD. Depresi Besar\nMari kita pikirkan langkah demi langkah.\nJawaban:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Anda hanya perlu menjawab dengan opsi A, B, C, atau D. \nJawaban:"),
            ]
    elif 'IT' in _name:
        _hint = f'Ci sono domande a scelta singola. Rispondi alla domanda rispondendo A, B, C o D.'
        _prompt = f'Si tratta di {{subject}} \nDomanda: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nRisposta:'
        _round = [
                dict(role='HUMAN', prompt="Si tratta di una domanda a scelta singola. Rispondi alla domanda rispondendo A, B, C o D.\nÈ riguardo al machine learning.\nDomanda: Nella torta di Yann LeCun, la ciliegina sulla torta è:\nA. apprendimento per rinforzo\nB. apprendimento auto-supervisionato\nC. apprendimento non supervisionato\nD. apprendimento supervisionato\nPensiamo passo dopo passo.\nRisposta:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Si tratta di una domanda a scelta singola. Rispondi alla domanda rispondendo A, B, C o D.\nÈ riguardo all'algebra astratta.\nDomanda: Trova l'ordine massimo possibile per alcuni elementi di Z_4 x Z_6.\nA. 4\nB. 6\nC. 12\nD. 24\nPensiamo passo dopo passo.\nRisposta:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="Si tratta di una domanda a scelta singola. Rispondi alla domanda rispondendo A, B, C o D.\nÈ riguardo alla geografia delle scuole superiori.\nDomanda: Quale religione qui sotto è una religione universalista?\nA. Taoismo\nB. Islam\nC. Shintoismo\nD. Confucianesimo\nPensiamo passo dopo passo.\nRisposta:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Si tratta di una domanda a scelta singola. Rispondi alla domanda rispondendo A, B, C o D.\nÈ riguardo alla filosofia.\nDomanda: Socrate suggerisce che il sacro è una parte di:\nA. ciò che è prudente.\nB. ciò che è giusto.\nC. ciò che è bello.\nD. ciò che è legale.\nPensiamo passo dopo passo.\nRisposta:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="Questa è una domanda a scelta singola. Rispondi alla domanda rispondendo A, B, C o D.\nRiguarda la storia degli Stati Uniti delle scuole superiori.\nDomanda: Questa domanda si riferisce alle seguenti informazioni. Tuttavia, forse sono più consapevole dell'importanza delle libertà civili in questo particolare momento della nostra storia rispetto a chiunque altro, perché viaggio per il paese, incontro persone e vedo cose che sono accadute a persone comuni, mi rendo conto di cosa significhi per la democrazia preservare le nostre libertà civili. Negli anni abbiamo dovuto combattere per la libertà civile e sappiamo che ci sono momenti in cui la luce si fa piuttosto fioca, e ogni volta che ciò accade, la democrazia è in pericolo. Ora, principalmente a causa dello stato travagliato del mondo nel suo insieme, le libertà civili sono scomparse in molti altri paesi. È impossibile, naturalmente, essere in guerra e mantenere la libertà di stampa, la libertà di parola e la libertà di riunione. Esse scompaiono automaticamente. E così, in molti paesi dove normalmente erano sicure, oggi sono svanite. In altri paesi, anche prima che arrivasse la guerra, non solo la libertà di stampa e la libertà di riunione, e la libertà di parola sono scomparse, ma anche la libertà di religione è scomparsa. E così sappiamo qui in questo paese, abbiamo una grave responsabilità. Siamo in pace. Non abbiamo ragione per le paure che governano così tante altre persone nel mondo; pertanto, dobbiamo difendere le libertà della democrazia. —Eleanor Roosevelt, Discorso all'Unione Americana per le Libertà Civili, Chicago, Illinois, 14 marzo 1940.\nNel suo discorso, Eleanor Roosevelt alluse alla minaccia precedente alle libertà civili creata da quale delle seguenti?\nA. Prima Guerra Mondiale\nB. Il New Deal\nC. Guerra Fredda\nD. Grande Depressione\nPensiamo passo dopo passo.\nRisposta:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"È sufficiente rispondere con l'opzione A, B, C o D. \nRisposta:"),
            ]
    elif 'JA' in _name:
        _hint = f'単一選択肢の質問があります。この質問にはA、B、C、またはDで答えてください。'
        _prompt = f'これは {{subject}} に関することです \n質問: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n回答:'
        _round = [
                dict(role='HUMAN', prompt="これは単一選択の質問です。A、B、C、またはDで回答してください。\nこれは機械学習に関するものです。\n質問：ヤン・ルカンのケーキにおいて、一番上のチェリーは：\nA. 強化学習\nB. 自己監督学習\nC. 教師なし学習\nD. 教師あり学習\n段階を追って考えましょう。\n回答："),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="これは単一選択の質問です。A、B、C、またはDで回答してください。\nこれは抽象代数学に関するものです。\n質問：Z_4 x Z_6 のいくつかの要素の最大可能な順序を求めなさい。\nA. 4\nB. 6\nC. 12\nD. 24\n段階を追って考えましょう。\n回答："),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="これは単一選択の質問です。A、B、C、またはDで回答してください。\nこれは高校の地理に関するものです。\n質問：以下のどの宗教が普遍化宗教ですか？\nA. 道教\nB. イスラム教\nC. 神道\nD. 儒教\n段階を追って考えましょう。\n回答："),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="これは単一選択の質問です。A、B、C、またはDで回答してください。\nこれは哲学に関するものです。\n質問：ソクラテスは、聖なるものが以下のどれの一部であると示唆していますか？\nA. 賢明なもの\nB. 正義のあるもの\nC. 美しいもの\nD. 合法的なもの\n段階を追って考えましょう。\n回答："),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="これは単一選択の質問です。A、B、C、またはDで回答してください。\nこれはアメリカの歴史についてです。\n質問：この質問は次の情報を参照しています。しかし、私はこの特定の歴史の瞬間における市民の自由の重要性に他の誰よりも敏感であるかもしれません。なぜなら、私は国を旅し、人々に会い、小さな人々に起こったことを見てきたからです。民主主義が市民の自由を守ることが何を意味するのかを理解しています。私たちは市民の自由のために戦わなければならなかった年月があり、光がかなり薄暗くなる時期があることを知っています。そのたびに民主主義は危険にさらされます。今、主に世界全体の不安定な状態によって、多くの他の国で市民の自由が消失しています。もちろん、戦争をしていては報道の自由、言論の自由、集会の自由を保つことは不可能です。それらは自動的に消えてしまいます。そして、通常は安全であった多くの国では、今日、これらはなくなりました。他の国々では、戦争が来る前から、報道の自由や集会の自由、言論の自由だけでなく、宗教の自由も消えていました。したがって、私たちはこの国で重大な責任を負っていることを知っています。私たちは平和です。他の国々の多くの人々が抱える恐れの理由はないので、私たちは民主主義の自由を守らなければなりません。 —エレノア・ルーズベルト、1940年3月14日イリノイ州シカゴでのアメリカ市民自由連合への演説\n彼女の演説で、エレノア・ルーズベルトは市民の自由に対する以前の脅威をどのように言及しましたか？\nA. 第一次世界大戦\nB. ニューディール\nC. 冷戦\nD. 大恐慌\n段階を追って考えてみましょう。\n回答："),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"選択肢A、B、C、またはDで答えるだけで大丈夫です。\n回答："),
            ]
    elif 'KO' in _name:
        _hint = f'단일 선택 질문이 있습니다. A, B, C 또는 D로 답변해 주세요.'
        _prompt = f'이것은 {{subject}}에 관한 것입니다 \n질문: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n답변:'
        _round = [
                dict(role='HUMAN', prompt="단일 선택 질문입니다. A, B, C 또는 D로 답하십시오.\n이 질문은 기계 학습에 관한 것입니다.\n질문: 얀 르쿤의 케이크에서 가장 위에 있는 체리는:\nA. 강화 학습\nB. 자기 지도 학습\nC. 비지도 학습\nD. 지도 학습\n단계별로 생각해 봅시다.\n답변:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="단일 선택 질문입니다. A, B, C 또는 D로 답하십시오.\n이 질문은 추상 대수학에 관한 것입니다.\n질문: Z_4 x Z_6의 어떤 요소의 최대 가능한 순서를 찾으세요.\nA. 4\nB. 6\nC. 12\nD. 24\n단계별로 생각해 봅시다.\n답변:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="단일 선택 질문입니다. A, B, C 또는 D로 답하십시오.\n이 질문은 고등학교 지리에 관한 것입니다.\n질문: 아래의 어떤 종교가 보편화 종교입니까?\nA. 도교\nB. 이슬람교\nC. 신도\nD. 유교\n단계별로 생각해 봅시다.\n답변:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="단일 선택 질문입니다. A, B, C 또는 D로 답하십시오.\n이 질문은 철학에 관한 것입니다.\n질문: 소크라테스는 신성한 것이 다음 중 어떤 것의 일부라고 제안합니까?\nA. 신중한 것\nB. 정의로운 것\nC. 아름다운 것\nD. 합법적인 것\n단계별로 생각해 봅시다.\n답변:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="이것은 단일 선택 질문입니다. A, B, C 또는 D로 답하십시오.\n이는 미국 역사에 관한 것입니다.\n질문: 이 질문은 다음 정보를 참조합니다. 그러나 어쩌면 나는 이 특정 역사적 순간에 시민 권리의 중요성에 대해 다른 누구보다 더 잘 인식하고 있습니다. 왜냐하면 나는 나라를 여행하며 사람들을 만나고 작은 사람들에게 일어난 일들을 보았기 때문입니다. 나는 민주주의가 시민 권리를 보존한다는 것이 무엇을 의미하는지 깨닫습니다. 우리는 시민 권리를 위해 싸워야 했던 여러 해를 거쳐 왔으며, 빛이 흐려지는 순간이 있음을 알고 있습니다. 그럴 때마다 민주주의가 위험에 처한 것처럼 느껴집니다. 지금은 전 세계의 불안정한 상태로 인해 많은 다른 국가에서 시민 권리가 사라졌습니다. 전쟁 중에 언론의 자유와 표현의 자유, 집회의 자유를 유지하는 것은 불가능합니다. 그것들은 자동으로 사라집니다. 그리고 따라서 일반적으로 안전했던 많은 국가에서는 오늘날 그것들이 사라졌습니다. 다른 국가에서는 전쟁이 오기 전에도 언론의 자유와 집회의 자유, 표현의 자유가 사라졌을 뿐만 아니라 종교의 자유도 사라졌습니다. 그래서 우리는 이 나라에서 중대한 책임을 지고 있다는 것을 압니다. 우리는 평화로운 상태에 있습니다. 전 세계의 많은 사람들이 느끼는 두려움에 대한 이유가 없으므로 우리는 민주주의의 자유를 지켜야 합니다. —엘리노르 루즈벨트, 1940년 3월 14일 일리노이주 시카고에서 미국 시민 자유 연합에 대한 연설\n그녀의 연설에서 엘리노르 루즈벨트는 시민 권리에 대한 이전의 위협이 어떤 것에 의해 발생했는지 언급했습니다.\nA. 제1차 세계 대전\nB. 뉴딜\nC. 냉전\nD. 대공황\n단계별로 생각해봅시다.\n답변:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"옵션 A, B, C 또는 D로만 대답하면 됩니다. \n답변:"),
            ]
    elif 'PT' in _name:
        _hint = f'Há uma pergunta de escolha única. Responda à pergunta escolhendo A, B, C ou D.'
        _prompt = f'É sobre {{subject}} \nPergunta: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nResposta:'
        _round = [
                dict(role='HUMAN', prompt="Esta é uma pergunta de escolha única. Responda à pergunta respondendo A, B, C ou D.\nÉ sobre aprendizado de máquina.\nPergunta: No bolo de Yann LeCun, a cereja no topo é:\nA. aprendizado por reforço\nB. aprendizado auto-supervisionado\nC. aprendizado não supervisionado\nD. aprendizado supervisionado\nVamos pensar passo a passo.\nResposta:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Esta é uma pergunta de escolha única. Responda à pergunta respondendo A, B, C ou D.\nÉ sobre álgebra abstrata.\nPergunta: Encontre a ordem máxima possível para algum elemento de Z_4 x Z_6.\nA. 4\nB. 6\nC. 12\nD. 24\nVamos pensar passo a passo.\nResposta:"),
                dict(role='BOT', prompt="C"),
                dict(role='HUMAN', prompt="Esta é uma pergunta de escolha única. Responda à pergunta respondendo A, B, C ou D.\nÉ sobre geografia do ensino médio.\nPergunta: Qual religião abaixo é uma religião universalizante?\nA. Taoísmo\nB. Islamismo\nC. Xintoísmo\nD. Confucionismo\nVamos pensar passo a passo.\nResposta:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Esta é uma pergunta de escolha única. Responda à pergunta respondendo A, B, C ou D.\nÉ sobre filosofia.\nPergunta: Sócrates sugere que o sagrado é uma parte de:\nA. o que é prudente.\nB. o que é justo.\nC. o que é belo.\nD. o que é legal.\nVamos pensar passo a passo.\nResposta:"),
                dict(role='BOT', prompt='B'),
                dict(role='HUMAN', prompt="Esta é uma pergunta de escolha única. Responda à pergunta respondendo A, B, C ou D.\nÉ sobre história dos Estados Unidos do ensino médio.\nPergunta: Esta pergunta se refere à seguinte informação. Talvez, no entanto, eu esteja mais consciente da importância das liberdades civis neste momento particular da nossa história do que qualquer outra pessoa, porque eu viajo pelo país e conheço pessoas e vejo coisas que aconteceram com pessoas pequenas, percebo o que significa para a democracia preservar nossas liberdades civis. Ao longo dos anos, tivemos que lutar pela liberdade civil, e sabemos que há momentos em que a luz fica bastante fraca, e toda vez que isso acontece, a democracia está em perigo. Agora, em grande parte por causa do estado problemático do mundo como um todo, as liberdades civis desapareceram em muitos outros países. É impossível, é claro, estar em guerra e manter a liberdade de imprensa, a liberdade de expressão e a liberdade de reunião. Elas desaparecem automaticamente. E assim, em muitos países onde normalmente estavam seguras, hoje desapareceram. Em outros países, mesmo antes da guerra chegar, não apenas a liberdade de imprensa e a liberdade de reunião, e a liberdade de expressão desapareceram, mas a liberdade de religião também desapareceu. E assim, sabemos aqui neste país, temos uma grave responsabilidade. Estamos em paz. Não temos razão para os medos que governam tantas outras pessoas ao redor do mundo; portanto, temos que proteger as liberdades da democracia. —Eleanor Roosevelt, Discurso à União Americana pelas Liberdades Civis, Chicago, Illinois, 14 de março de 1940\nEm seu discurso, Eleanor Roosevelt aludiu à ameaça anterior às liberdades civis criada por qual das seguintes?\nA. Primeira Guerra Mundial\nB. O Novo Pacto\nC. A Guerra Fria\nD. A Grande Depressão\nVamos pensar passo a passo.\nResposta:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Você só precisa responder com a opção A, B, C ou D. \nResposta:"),
            ]
    elif 'ZH' in _name:
        _hint = f'这里有一个单项选择题。请通过选择 A、B、C 或 D 来回答该问题。'
        _prompt = f'这是关于 {{subject}} 的内容\n问题：{{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案：'
        _round = [
                dict(role='HUMAN', prompt="这是一个单项选择题。请回复 A、B、C 或 D。\n这是关于美国历史的。\n问题：这个问题参考以下信息。或许，我对我们历史这一特定时刻公民自由重要性的认识，比其他任何人都要深刻，因为我在全国各地旅行，见到人们，看到那些发生在小人物身上的事情，我意识到民主意味着要保护我们的公民自由。在这些年里，我们不得不为公民自由而奋斗，我们知道有时光芒会变得微弱，每当这种情况发生时，民主就处于危险之中。现在，主要由于整个世界的动荡状态，许多其他国家的公民自由已经消失。当然，在战争中保持新闻自由、言论自由和集会自由是不可能的。它们会自动消失。因此，在许多通常是安全的国家里，今天它们已经消失。在其他国家，即使在战争到来之前，不仅新闻自由、集会自由和言论自由消失了，宗教自由也消失了。因此，我们知道，在这个国家，我们有着重大的责任。我们处于和平状态。我们没有理由去害怕其他世界上许多人所感受到的恐惧；因此，我们必须保护民主的自由。——埃莉诺·罗斯福，1940年3月14日在伊利诺伊州芝加哥的美国公民自由联盟演讲\n在她的演讲中，埃莉诺·罗斯福提到了哪一事件对公民自由造成了早期威胁？\nA. 第一次世界大战\nB. 新政\nC. 冷战\nD. 大萧条\n让我们一步一步思考。\n答案："),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="这是一个单项选择题。请回复 A、B、C 或 D。\n这是关于高中地理的。\n问题：以下哪个宗教是普世宗教？\nA. 道教\nB. 伊斯兰教\nC. 神道教\nD. 儒教\n让我们一步一步思考。\n答案："),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="这是一个单项选择题。请回复 A、B、C 或 D。\n这是关于哲学的。\n问题：苏格拉底建议神圣是以下哪个部分：\nA. 什么是谨慎的。\nB. 什么是正义的。\nC. 什么是美的。\nD. 什么是合法的。\n让我们一步一步思考。\n答案："),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="这是一个单项选择题。请回复 A、B、C 或 D。\n这是关于抽象代数的。\n问题：找到 Z_4 x Z_6 中某个元素的最大可能阶数。\nA. 4\nB. 6\nC. 12\nD. 24\n让我们一步一步思考。\n答案："),
                dict(role='BOT', prompt='C'),
                dict(role='HUMAN', prompt="这是一个单项选择题。请回复 A、B、C 或 D。\n这是关于机器学习的。\n问题：在 Yann LeCun 的蛋糕中，最上面的樱桃是：\nA. 强化学习\nB. 自监督学习\nC. 无监督学习\nD. 监督学习\n让我们一步一步思考。\n答案："),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"只需选择 A、B、C 或 D 来回答该问题。 \n回答："),
            ]
    elif 'YO' in _name:
        _hint = f'Ibeere kan wa ti o ni yiyan kan. Fesi si ibeere naa nipa fesi A, B, C tabi D.'
        _prompt = f'Eyi jẹ nipa {{subject}}.\nIbeere: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nFesi:'
        _round = [
                dict(role='HUMAN', prompt="Ibeere kan wa ti o ni yiyan kan. Fesi si ibeere naa nipa fesi A, B, C tabi D.\nit jẹ nipa filosofía\nIbeere: Ibeere yii tọka si alaye ti n bọ. Boya, sibẹsibẹ, Mo ni imọ diẹ sii nipa pataki awọn ominira ilu ni akoko pataki yii ti itan wa ju ẹnikẹni miiran lọ, nitori Mo n rin irin-ajo kọja ilẹ naa ati pe Mo pade awọn eniyan ati pe Mo ri awọn nkan ti o ti ṣẹlẹ si awọn eniyan kekere, Mo mọ ohun ti o tumọ si fun ijọba mimọ lati pa awọn ominira ilu wa. Ni gbogbo ọdun, a ti ni lati ja fun ominira ilu, ati pe a mọ pe awọn akoko wa nigba ti ina di dimu, ati nigbami ti eyi ṣẹlẹ, ijọba wa ni ewu. Bayi, ni pataki nitori ipo iṣoro agbaye ni apapọ, awọn ominira ilu ti parẹ ni ọpọlọpọ awọn orilẹ-ede miiran. O jẹ alailẹgbẹ, dajudaju, lati wa ni ogun ati ki o ṣetọju ominira iwe irohin ati ominira ẹtọ ati ominira apejọ. Wọn parẹ laifọwọyi. Ati pe nitorina ni ọpọlọpọ awọn orilẹ-ede nibiti wọn ti jẹ ailewu ni deede, loni wọn ti parẹ. Ni awọn orilẹ-ede miiran, paapaa ṣaaju ki ogun wa, kii ṣe nikan ominira iwe irohin ati ominira apejọ , ati ominira ẹtọ ti parẹ, ṣugbọn ominira ẹsin ti parẹ. Ati pe nitorina a mọ nibi ninu orilẹ-ede yii, a ni ojuse pataki. A wa ni alaafia. A ko ni idi fun awọn bẹru ti o ṣakoso ọpọlọpọ awọn eniyan miiran ni gbogbo agbaye; nitorina, a ni lati daabobo awọn ominira ti ijọba mimọ. —Eleanor Roosevelt, Ikede si American Civil Liberties Union, Chicago, Illinois, Oṣu Kẹta 14, 1940 Ninu ọrọ rẹ, Eleanor Roosevelt daba pe ewu ti tẹlẹ si awọn ominira ilu ṣẹda nipasẹ eyi ti o tẹle? \nA.Ija Agbaye I\nB.Iwọn Ilana Tuntun\nC.Ijakadi Tutu\nD.Ipe wọn nla\nJẹ ki a ronu ni igbesẹ nipasẹ igbesẹ\nFesi:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Ibeere kan wa ti o ni yiyan kan. Fesi si ibeere naa nipa fesi A, B, C tabi D.\nit jẹ nipa геography ile-iwe gíga\nIbeere: Igbagbọ wo ni isalẹ jẹ igbagbọ agbaye? A.Taoism\n B.Islam\n C.Shintoism\n D.Confucianism\nJẹ ki a ronu ni igbesẹ nipasẹ igbesẹ\nFesi:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Ibeere kan wa ti o ni yiyan kan. Fesi si ibeere naa nipa fesi A, B, C tabi D.\nit jẹ nipa filosofía\nIbeere: Socrates daba pe mímọ̀ jẹ apakan kan ti:\nA. ohun ti o jẹ ọlọgbọn.\nB. ohun ti o jẹ ododo.\nC. ohun ti o jẹ ẹwa.\nD. ohun ti o jẹ ofin.\nJẹ ki a ronu ni igbesẹ nipasẹ igbesẹ\nFesi:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Ibeere kan wa ti o ni yiyan kan. Fesi si ibeere naa nipa fesi A, B, C tabi D.\nit jẹ nipa aljebra igba\nIbeere: Wa aṣẹ to pọju fun diẹ ninu awọn eroja ti Z_4 x Z_6.\n A.4\nB.6\nC.12\nD.24\nJẹ ki a ronu ni igbesẹ nipasẹ igbesẹ\nFesi:"),
                dict(role='BOT', prompt='C'),
                dict(role='HUMAN', prompt="Ibeere kan wa ti o ni yiyan kan. Fesi si ibeere naa nipa fesi A, B, C tabi D.\nit jẹ nipa ikẹkọ ẹrọ\nIbeere: Ninu akara Yann LeCun, eso cherry lori oke ni: \nA.ikẹkọ imudara\nB.ikẹkọ ara-ṣaaju\nC.ikẹkọ aibojumu\nD.ikẹkọ ti a fojusi\nJẹ ki a ronu ni igbesẹ nipasẹ igbesẹ\nFesi:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Kan yan A, B, C tabi D lati fesi si ibeere naa. \nFesi:"),
            ]
    elif 'SW' in _name:
        _hint = f'Kuna swali moja la chaguo. Jibu swali kwa kujibu A, B, C au D.'
        _prompt = f'Hii ni kuhusu {{subject}}.\nSwali: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nJibu:'
        _round = [
                dict(role='HUMAN', prompt="Kuna swali moja la chaguo. Jibu swali kwa kujibu A, B, C au D.\nit ni kuhusu filosofia\nSwali: Swali hili linahusiana na taarifa ifuatayo. Huenda, hata hivyo, mimi nina ufahamu zaidi wa umuhimu wa uhuru wa kiraia katika wakati huu maalum wa historia yetu kuliko mtu mwingine yeyote, kwa sababu nasafiri nchini na kukutana na watu na kuona mambo yanayotokea kwa watu wadogo, ninatambua inamaanisha nini kwa demokrasia kuhifadhi uhuru wetu wa kiraia. kwa kupitia miaka yote tumepaswa kupigania uhuru wa kiraia, na tunajua kuwa kuna nyakati ambapo mwanga unakuwa dhaifu, na kila wakati hii inatokea, demokrasia iko katika hatari. Sasa, hasa kwa sababu ya hali ya machafuko ya ulimwengu kwa jumla, uhuru wa kiraia umepotea katika nchi nyingi nyingine. Haiwezekani, kwa kweli, kuwa katika vita na kudumisha uhuru wa vyombo vya habari na uhuru wa kusema na uhuru wa kukusanyika. Vinapotea kiotomatiki. Na hivyo katika nchi nyingi ambapo kwa kawaida walikuwa salama, leo wameondoka. Katika nchi zingine, hata kabla ya vita kuja, si tu uhuru wa vyombo vya habari na uhuru wa kukusanyika, na uhuru wa kusema umepotea, bali pia uhuru wa dini umepotea. Na hivyo tunajua hapa katika nchi hii, tuna wajibu mzito. Tuko katika amani. Hatuna sababu ya hofu ambazo zinatawala watu wengi wengine duniani; kwa hiyo, ni lazima tulinde uhuru wa demokrasia. —Eleanor Roosevelt, Hotuba kwa Muungano wa Uhuru wa Kiraia wa Marekani, Chicago, Illinois, Machi 14, 1940 Katika hotuba yake, Eleanor Roosevelt alizungumzia tishio la awali kwa uhuru wa kiraia lililotolewa na ipi kati ya yafuatayo? \nA.Vita vya Kwanza vya Dunia\nB.Mkataba Mpya\nC.Vita vya Baridi\nD.Mapinduzi Makuu\nHebu tufikirie hatua kwa hatua\nJibu:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt="Kuna swali moja la chaguo. Jibu swali kwa kujibu A, B, C au D.\nit ni kuhusu jiografia ya shule ya sekondari\nSwali: Dini ipi hapa chini ni dini ya kueneza? \nA.Taoism\nB.Islam\nC.Shintoism\nD.Confucianism\nHebu tufikirie hatua kwa hatua\nJibu:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Kuna swali moja la chaguo. Jibu swali kwa kujibu A, B, C au D.\nit ni kuhusu filosofia\nSwali: Socrates anapendekeza kwamba kitakatifu ni sehemu moja ya:\nA. kile kilicho busara.\nB. kile kilicho haki.\nC. kile kilicho kizuri.\nD. kile kilicho halali.\nHebu tufikirie hatua kwa hatua\nJibu:"),
                dict(role='BOT', prompt="B"),
                dict(role='HUMAN', prompt="Kuna swali moja la chaguo. Jibu swali kwa kujibu A, B, C au D.\nit ni kuhusu algebra ya kiabstract\nSwali: Pata kipindi chenye uwezo mkubwa zaidi kwa kipengele baadhi ya Z_4 x Z_6.\nA.4\nB.6\nC.12\nD.24\nHebu tufikirie hatua kwa hatua\nJibu:"),
                dict(role='BOT', prompt='C'),
                dict(role='HUMAN', prompt="Kuna swali moja la chaguo. Jibu swali kwa kujibu A, B, C au D.\nit ni kuhusu kujifunza kwa mashine\nSwali: Katika keki ya Yann LeCun, cherii juu ni: \nA.kujifunza kwa nguvu\nB.kujifunza kwa kujisimamia\nC.kujifunza bila usimamizi\nD.kujifunza kwa usimamizi\nHebu tufikirie hatua kwa hatua\nJibu:"),
                dict(role='BOT', prompt='A'),
                dict(role='HUMAN', prompt=f'{_hint}\n{_prompt}\n'+"Chagua tu A, B, C au D kujibu swali hili. \nJibu:"),
            ]
    else:
        _hint = f'There is a single choice question. Answer the question by replying A, B, C or D.'
        _prompt = f'it is about {{subject}} \nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer:'
    mmmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=_round
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    mmmlu_datasets.append(
        dict(
            abbr=f'openai_m{_name}',
            type=MMMLUDataset,
            path='openai/MMMLU',
            name=_name,
            reader_cfg=mmmlu_reader_cfg,
            infer_cfg=mmmlu_infer_cfg,
            eval_cfg=mmmlu_eval_cfg,
        ))

del _name, _hint, _prompt, _round
