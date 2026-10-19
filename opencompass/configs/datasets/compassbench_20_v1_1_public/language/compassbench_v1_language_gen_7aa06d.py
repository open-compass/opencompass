from opencompass.openicl.icl_evaluator import EMEvaluator, BleuFloresEvaluator, RougeEvaluator, JiebaRougeEvaluator

compassbench_v1_language_datasets = [
     # dict(path='data/compassbench_v1.1.public/language/information_retrieval_en.jsonl',
     #      abbr='information_retrieval_en_public', data_type= 'qa', infer_method='gen', output_column='answers',
     #      human_prompt='{context}\n{origin_question}\nConcise and direct answer is',
     #      evaluator=EMEvaluator),
     # dict(path='data/compassbench_v1.1.public/language/information_retrieval_zh.jsonl',
     #      abbr='information_retrieval_zh_public', data_type= 'qa', infer_method='gen', output_column='answers',
     #      human_prompt='{context}\n{origin_question}\n简单直接的回答是',
     #      evaluator=EMEvaluator),

     dict(path='data/compassbench_v1.1.public/language/intention_recognition_en.jsonl',
          abbr='intention_recognition_en_circular_public', data_type='circular-mcq', infer_method='gen'),
     dict(path='data/compassbench_v1.1.public/language/intention_recognition_zh.jsonl',
          abbr='intention_recognition_zh_circular_public', data_type='circular-mcq', infer_method='gen'),

     dict(path='data/compassbench_v1.1.public/language/sentiment_analysis_en.jsonl',
          abbr='sentiment_analysis_en_circular_public', data_type='circular-mcq', infer_method='gen'),
     dict(path='data/compassbench_v1.1.public/language/sentiment_analysis_zh.jsonl',
          abbr='sentiment_analysis_zh_circular_public', data_type='circular-mcq', infer_method='gen'),

     dict(path='data/compassbench_v1.1.public/language/translation.jsonl',
          abbr='translation_public', data_type= 'qa', infer_method='gen',
          evaluator=BleuFloresEvaluator),

     dict(path='data/compassbench_v1.1.public/language/content_critic_en.jsonl',
          abbr='content_critic_en_circular_public', data_type='circular-mcq', infer_method='gen'),
     dict(path='data/compassbench_v1.1.public/language/content_critic_zh.jsonl',
          abbr='content_critic_zh_circular_public', data_type='circular-mcq', infer_method='gen'),

     dict(path='data/compassbench_v1.1.public/language/content_summarization_en.jsonl',
          abbr='content_summarization_en_public', data_type= 'qa', infer_method='gen', output_column='summary',
          human_prompt='{article}\nSummary of the article is:\n',
          evaluator=RougeEvaluator),
     dict(path='data/compassbench_v1.1.public/language/content_summarization_zh.jsonl',
          abbr='content_summarization_zh_public', data_type= 'qa', infer_method='gen', output_column='summary',
          human_prompt='{article}\n上述内容摘要如下：\n',
          evaluator=JiebaRougeEvaluator),

     dict(path='data/compassbench_v1.1.public/language/traditional_cultural_understanding_zh.jsonl',
          abbr='traditional_cultural_understanding_zh_circular_public', data_type='circular-mcq', infer_method='gen'),

     dict(path='data/compassbench_v1.1.public/language/chinese_semantic_understanding_zh.jsonl',
          abbr='chinese_semantic_understanding_zh_circular_public', data_type='circular-mcq', infer_method='gen'),
]
