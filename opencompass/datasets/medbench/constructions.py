# flake8: noqa
import pandas as pd


class TaskSchema(object):

    def __init__(self,
                 passage=None,
                 question=None,
                 options=None,
                 label=None,
                 answer=None,
                 other=None):
        self.passage = passage
        self.question = question
        self.options = options
        self.label = label
        self.answer = answer
        self.other = other

    def to_dict(self):
        return {
            'passage': self.passage,
            'question': self.question,
            'options': self.options,
            'label': self.label,
            'answer': self.answer,
            'other': self.other
        }


# define README.json
class MedBenchInstance(object):

    def __init__(self, task_description, data_source, task_schema, output,
                 evaluation_metric, task_example):
        self.task_description = task_description
        self.data_source = data_source
        self.task_schema = task_schema
        self.output = output
        self.evaluation_metric = evaluation_metric
        self.task_example = task_example

    def to_dict(self):
        return {
            'task description': self.task_description,
            'data source': self.data_source,
            'task schema': self.task_schema.to_dict(),
            'output': self.output,
            'evaluation metric': self.evaluation_metric,
            'task example': self.task_example
        }


class ChatGPTSchema(object):

    def __init__(self, context=None, metadata=''):
        self.context = context
        self.metadata = metadata

    def to_dict(self):
        return {'context': self.context, 'metadata': self.metadata}


class ResultsForHumanSchema(object):

    def __init__(self,
                 index,
                 problem_input,
                 label,
                 model_input='',
                 model_output='',
                 parse_result='',
                 first_stage_output='',
                 second_stage_input='',
                 is_correct=False):
        self.index = index
        self.problem_input = problem_input
        self.model_input = model_input
        self.model_output = model_output
        self.parse_result = parse_result
        self.label = label
        self.first_stage_output = first_stage_output
        self.second_stage_input = second_stage_input
        self.is_correct = is_correct

    def to_dict(self):
        return {
            'index': self.index,
            'problem_input': self.problem_input,
            'model_input': self.model_input,
            'model_output': self.model_output,
            'parse_result': self.parse_result,
            'label': self.label,
            'is_correct': self.is_correct,
            'first_stage_output': self.first_stage_output,
            'second_stage_input': self.second_stage_input,
        }

    @staticmethod
    def to_tsv(result_list, path):
        result_json = [item.to_dict() for item in result_list]
        table = pd.json_normalize(result_json)
        table.to_excel(path, index=False)
