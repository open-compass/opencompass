# IFEval
## Introduction
The following introduction comes from the abstract in [Instruction Following Evaluation for Large Language Models](arxiv.org/abs/2311.07911)

```
IFEval is a straightforward and easy-to-reproduce evaluation benchmark. It focuses on a set of "verifiable instructions" such as "write in more than 400 words" and "mention the keyword of AI at least 3 times". We identified 25 types of those verifiable instructions and constructed around 500 prompts, with each prompt containing one or more verifiable instructions.
```

## Official link

### Paper

[Instruction Following Evaluation for Large Language Models](arxiv.org/abs/2311.07911)

### Repository

[instruction_following_eval](github.com/google-research/google-research/tree/master/instruction_following_eval)

## Examples
Input example I:
```
Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.
```
Output example I (from GPT-4):
```
Raymond III was the Count of Tripoli from 1152 to 1187 and Prince of Galilee and Tiberias in the Kingdom of Jerusalem. He was born in 1140 to Raymond II of Tripoli and Hodierna of Jerusalem. His mother was the daughter of Baldwin II of Jerusalem. \n\n*Early Life and Succession*\n\nRaymond III was only a child when his father was murdered. His mother Hodierna was regent until Raymond came of age. In 1155 Raymond married Eschiva the daughter of Walter I of Beirut. They had three children: Raymond IV Bertrand and a daughter who married Guy of Lusignan. \n\n*Reign*\n\nRaymond III's reign was marked by a series of conflicts and alliances with the Muslim world. He was involved in the defense of the Kingdom of Jerusalem against Nur ad-Din and later Saladin. He was also a key figure in the internal politics of the kingdom. He was a supporter of the queen mother Amalric of Jerusalem and opposed the succession of Guy of Lusignan. \n\n*Later Life and Death*\n\nIn 1187 Raymond III was part of the disastrous Battle of Hattin where the Christian forces were defeated by Saladin. He was one of the few to escape the battlefield but died later that year. His son Raymond IV succeeded him as Count of Tripoli. \n\nRaymond III's life and reign were marked by the complex politics and warfare of the Crusader states. He was a key figure in the defense of the Kingdom of Jerusalem and his death marked a significant loss for the Christian forces in the region. His legacy is a testament to the turbulent times in which he lived and the challenges faced by the Crusader states in their relations with the Muslim world.
```
Input example II:
```
"I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response."
```
Output example II (from GPT-4):
```
Hark! Hearken to the tale of thy journey to the land of the rising sun, Japan. \n\nUpon the morrow of thy arrival, thou shalt find thyself in Tokyo, the city of lights and towers. In the morn, thou shalt explore the grandeur of the Imperial Palace, a sight to behold. As the sun reaches its zenith, thou shalt partake in a feast of sushi, a delicacy of the land. As the day wanes, thou shalt wander through the electric streets of Akihabara, a realm of technology and animation.\n\nOn the second day, thou shalt journey to the ancient city of Kyoto. In the soft light of dawn, thou shalt visit the golden pavilion of Kinkaku-ji, a vision of beauty and tranquility. As the day unfolds, thou shalt wander through the bamboo grove of Arashiyama, a place of peace and serenity. As the sun sets, thou shalt partake in a traditional tea ceremony, a ritual of grace and harmony.\n\nOn the third day, thou shalt travel to the city of Hiroshima. In the morning, thou shalt pay thy respects at the Peace Memorial Park, a place of remembrance and hope. As the day progresses, thou shalt journey to the island of Miyajima, home to the floating torii gate, a sight of wonder and awe. As the day ends, thou shalt feast on okonomiyaki, a dish of delight and satisfaction.\n\nOn the fourth day, thou shalt return to Tokyo. In the morning, thou shalt ascend the Tokyo Skytree, a tower of height and grandeur. As the day unfolds, thou shalt explore the bustling streets of Shibuya, a realm of fashion and culture. As the sun sets, thou shalt dine in the vibrant district of Roppongi, a place of gastronomy and nightlife.\n\nOn the fifth and final day, thou shalt bid farewell to the land of the rising sun. With memories of beauty and wonder, thou shalt embark on thy journey home, carrying with thee the spirit of Japan. \n\nThus ends the tale of thy journey, a tale of exploration and discovery. May thy travels be filled with joy and adventure, and may the land of the rising sun forever hold a place in thy heart.
```
## Evaluation results

```
dataset    version    metric                        mode      baichuan2-7b-chat-hf    baichuan2-13b-chat-hf    internlm2-chat-7b-hf    internlm2-chat-20b-hf    llama-2-7b-chat-hf    llama-2-13b-chat-hf
---------  ---------  ----------------------------  ------  ----------------------  -----------------------  ----------------------  -----------------------  --------------------  ---------------------
IFEval     3321a3     Prompt-level-strict-accuracy  gen                      36.04                    35.49                   38.26                    33.09                 33.46                  33.64
IFEval     3321a3     Inst-level-strict-accuracy    gen                      46.76                    46.76                   49.16                    45.32                 45.68                  45.44
IFEval     3321a3     Prompt-level-loose-accuracy   gen                      37.52                    37.71                   42.51                    39.37                 43.81                  47.32
IFEval     3321a3     Inst-level-loose-accuracy     gen                      48.44                    49.16                   53.72                    51.08                 55.64                  58.03
```

## Reference
```
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```
