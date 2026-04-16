import copy
import unittest

from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate


class TestRawPromptTemplate(unittest.TestCase):

    def setUp(self) -> None:
        """设置测试用例的通用数据"""
        self.simple_messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': 'Translate to French: {input}'
            },
            {
                'role': 'assistant',
                'content': '{output}'
            },
        ]
        self.multi_round_messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': 'Hello!'
            },
            {
                'role': 'assistant',
                'content': 'Hi there!'
            },
            {
                'role': 'user',
                'content': '{input}'
            },
            {
                'role': 'assistant',
                'content': '{output}'
            },
        ]
        self.entry = {'input': 'Good morning.', 'output': 'Bonjour.'}

    def test_init(self):
        """测试正常初始化"""
        rpt = RawPromptTemplate(self.simple_messages)

        self.assertEqual(rpt.messages, self.simple_messages)
        self.assertTrue(rpt.format_variables)
        self.assertEqual(rpt.prompt_type, 'raw_messages')

    def test_init_with_format_variables_false(self):
        """测试 format_variables=False 的情况"""
        rpt = RawPromptTemplate(self.simple_messages, format_variables=False)

        self.assertFalse(rpt.format_variables)

    def test_validation_not_list(self):
        """测试 messages 不是列表的情况"""
        with self.assertRaises(TypeError) as ctx:
            RawPromptTemplate('not a list')
        self.assertIn('must be a list', str(ctx.exception))

    def test_validation_item_not_dict(self):
        """测试 messages 元素不是字典的情况"""
        with self.assertRaises(TypeError) as ctx:
            RawPromptTemplate([{
                'role': 'user',
                'content': 'hello'
            }, 'not a dict'])
        self.assertIn('must be a dict', str(ctx.exception))

    def test_validation_missing_role(self):
        """测试缺少 role 字段的情况"""
        with self.assertRaises(ValueError) as ctx:
            RawPromptTemplate([{'content': 'hello'}])
        self.assertIn("missing 'role'", str(ctx.exception))

    def test_validation_missing_content(self):
        """测试缺少 content 字段的情况"""
        with self.assertRaises(ValueError) as ctx:
            RawPromptTemplate([{'role': 'user'}])
        self.assertIn("missing 'content'", str(ctx.exception))

    def test_validation_invalid_role(self):
        """测试无效的 role 的情况"""
        with self.assertRaises(ValueError) as ctx:
            RawPromptTemplate([{'role': 'invalid_role', 'content': 'hello'}])
        self.assertIn('invalid role', str(ctx.exception))

    def test_generate_item(self):
        """测试生成 messages 并替换变量"""
        rpt = RawPromptTemplate(self.simple_messages)
        result = rpt.generate_item(self.entry)

        expected = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': 'Translate to French: Good morning.'
            },
            {
                'role': 'assistant',
                'content': 'Bonjour.'
            },
        ]
        self.assertEqual(result, expected)

    def test_generate_item_no_format(self):
        """测试 format_variables=False 时不替换变量"""
        rpt = RawPromptTemplate(self.simple_messages, format_variables=False)
        result = rpt.generate_item(self.entry)

        # 变量不应该被替换
        self.assertEqual(result[1]['content'], 'Translate to French: {input}')
        self.assertEqual(result[2]['content'], '{output}')

    def test_generate_item_does_not_modify_original(self):
        """测试 generate_item 不修改原始 messages"""
        rpt = RawPromptTemplate(self.simple_messages)
        original_messages = copy.deepcopy(rpt.messages)

        rpt.generate_item(self.entry)

        self.assertEqual(rpt.messages, original_messages)

    def test_generate_item_with_output_field(self):
        """测试 output_field 参数（兼容现有接口）"""
        rpt = RawPromptTemplate(self.simple_messages)
        result = rpt.generate_item(self.entry,
                                   output_field='output',
                                   output_field_replace_token='[ANSWER]')

        # RawPromptTemplate 暂不使用 output_field 参数
        # 但应正常返回结果
        self.assertEqual(len(result), 3)

    def test_generate_ice_item(self):
        """测试 generate_ice_item（兼容接口）"""
        rpt = RawPromptTemplate(self.simple_messages)
        result = rpt.generate_ice_item(self.entry)

        expected = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': 'Translate to French: Good morning.'
            },
            {
                'role': 'assistant',
                'content': 'Bonjour.'
            },
        ]
        self.assertEqual(result, expected)

    def test_generate_label_prompt_item(self):
        """测试 generate_label_prompt_item（兼容接口）"""
        rpt = RawPromptTemplate(self.simple_messages)
        result = rpt.generate_label_prompt_item(self.entry,
                                                ice='some ice content',
                                                label='some label')

        expected = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': 'Translate to French: Good morning.'
            },
            {
                'role': 'assistant',
                'content': 'Bonjour.'
            },
        ]
        self.assertEqual(result, expected)

    def test_repr(self):
        """测试 __repr__ 方法"""
        rpt = RawPromptTemplate(self.simple_messages)
        repr_str = repr(rpt)

        self.assertIn('RawPromptTemplate', repr_str)
        self.assertIn('messages=', repr_str)


if __name__ == '__main__':
    unittest.main()
