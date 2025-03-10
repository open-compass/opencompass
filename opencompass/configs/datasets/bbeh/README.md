# BB#H

```bash
python3 run.py --models hf_internlm2_7b --datasets bbeh_gen --debug
python3 run.py --models hf_meta_llama3_8b_instruct --datasets bbeh_gen --debug
```

## Models

|                   model                    | score |
|:-----------------------------------------:|------:|
| Meta-Llama-3-8B-Instruct-LMDeploy-API     | 10.93 |

### Details

|                   model                    | boolean_expressions | disambiguation_qa | geometric_shapes | hyperbaton | movie_recommendation | nycc | shuffled_objects | boardgame_qa |
|:-----------------------------------------:|--------------------:|------------------:|-----------------:|-----------:|---------------------:|-----:|-----------------:|-------------:|
| Meta-Llama-3-8B-Instruct-LMDeploy-API     |               14.00 |             33.33 |            13.50 |       1.00 |               28.00 | 11.00 |            10.00 |        18.50 |

|                   model                    | buggy_tables | causal_understanding | dyck_languages | linguini | multistep_arithmetic | object_counting | object_properties | sarc_triples |
|:-----------------------------------------:|-------------:|---------------------:|---------------:|---------:|---------------------:|----------------:|------------------:|-------------:|
| Meta-Llama-3-8B-Instruct-LMDeploy-API     |         0.00 |               42.50 |           3.50 |     2.00 |                 0.00 |            0.00 |              1.00 |        17.00 |

|                   model                    | spatial_reasoning | sportqa | temporal_sequence | time_arithmetic | web_of_lies | word_sorting | zebra_puzzles |
|:-----------------------------------------:|------------------:|-------:|-----------------:|----------------:|------------:|-------------:|--------------:|
| Meta-Llama-3-8B-Instruct-LMDeploy-API     |              4.00 |   5.00 |             2.00 |            3.00 |        7.50 |         2.00 |          3.50 |
