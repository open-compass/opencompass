# ARC Prize Public Evaluation

#### Overview
The spirit of ARC Prize is to open source progress towards AGI. To win prize money, you will be required to publish reproducible code/methods into public domain.

ARC Prize measures AGI progress using the [ARC-AGI private evaluation set](https://arcprize.org/guide#private), [the leaderboard is here](https://arcprize.org/leaderboard), and the Grand Prize is unlocked once the first team reaches [at least 85%](https://arcprize.org/guide#grand-prize-goal).

Note: the private evaluation set imposes limitations on solutions (eg. no internet access, so no GPT-4/Claude/etc). There is a [secondary leaderboard](https://arcprize.org/leaderboard) called ARC-AGI-Pub, it measures the [public evaluation set](https://arcprize.org/guide#public-tasks) and imposes no limits but it is not part of ARC Prize 2024 at this time.


#### Tasks
ARC-AGI tasks are a series of three to five input and output tasks followed by a final task with only the input listed. Each task tests the utilization of a specific learned skill based on a minimal number of cognitive priors.

![alt text](https://arcprize.org/media/images/arc-task-grids.jpg)

Tasks are represented as JSON lists of integers. These JSON objects can also be represented visually as a grid of colors using an ARC-AGI task viewer.

A successful submission is a pixel-perfect description (color and position) of the final task's output.

#### Format

As mentioned above, tasks are stored in JSON format. Each JSON file consists of two key-value pairs.

`train`: a list of two to ten input/output pairs (typically three.) These are used for your algorithm to infer a rule.

`test`: a list of one to three input/output pairs (typically one.) Your model should apply the inferred rule from the train set and construct an output solution. You will have access to the output test solution on the public data. The output solution on the private evaluation set will not be revealed.

Here is an example of a simple ARC-AGI task that has three training pairs along with a single test pair. Each pair is shown as a 2x2 grid. There are four colors represented by the integers 1, 4, 6, and 8. Which actual color (red/green/blue/black) is applied to each integer is arbitrary and up to you.

```json
{
  "train": [
    {"input": [[1, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
    {"input": [[0, 0], [4, 0]], "output": [[4, 4], [4, 4]]},
    {"input": [[0, 0], [6, 0]], "output": [[6, 6], [6, 6]]}
  ],
  "test": [
    {"input": [[0, 0], [0, 8]], "output": [[8, 8], [8, 8]]}
  ]
}
```

#### Performance

| Qwen2.5-72B-Instruct | LLaMA3.1-70B-Instruct | gemma-2-27b-it | 
| ----- | ----- |  ----- | 
| 0.09 | 0.06 | 0.05 |