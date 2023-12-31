# flake8: noqa: E501
import json
import os.path as osp
import re

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .subjective_cmp import SubjectiveCmpDataset

prefix = """
        这是一道关于内容准确性的评测任务：

        内容准确性评分要求包括：

        1. 内容正确；

        2. 内容符合事实；

        请根据提供评分要求，问题以及相应的回答，对于回答给出 0~5 的分数（0 最差，5 最好），并在后面针对每一条评分要求说明原因，若某一要求不适用于该问题和回答，可在说明不适用的理由后给出不适用。同时在最后根据这两条评分打出回答在内容准确性上的得分。

        你的输出应形如：

        原因：blahblah blahblah

        分数：blah

        例 1:
问题: <问题开始> 1982年宪法是中国建国以来的第几部宪法？ <问题结束>

回答: <回答开始> 1982年的中华人民共和国宪法是中国建国以来的第三部宪法。 <回答结束>

原因：1. 1982年宪法是中国建国以来的第四部宪法而不是第三部，内容正确性得分：0；2. 回答包含事实性错误，1982年宪法是中国建国以来的第四部宪法而不是第三部。符合事实性得分：0；
分数：0
例 2:
问题: <问题开始> 可以概括一下《汤姆叔叔的小屋》这本书的内容吗？ <问题结束>

回答: <回答开始> 《汤姆叔叔的小屋》是美国作家哈珀·李于1862年出版的一部长篇小说，主要描写19世纪南卡罗来纳州一个奴隶家庭的悲惨遭遇。 故事主要围绕着主人公汤姆叔叔(一个奴隶)和其家庭成员展开，他们生活在一个充满暴力和不公的社会中，面临着各种困境和挑战。故事主要讲述了汤姆叔叔的妻子哈莉、哥哥乔治、姐姐艾玛和妹妹海莉遭受的虐待和不公，以及他们试图逃脱奴隶制度的故事。 小说通过生动的描写和深刻的思想，揭示了奴隶制度的残忍和不公，以及对奴隶的虐待和折磨。小说也表达了对自由和平等的追求，以及对奴隶制度的批判和反抗。 《汤姆叔叔的小屋》被认为是美国文学史上的一部经典之作，对后来的文学和社会思想产生了深远的影响。 <回答结束>

原因：1. 回答对于故事的概括基本正确，但人物错误，内容正确性得分：4；2. 回答对于作者的描述以及小说中人物如哈莉、乔治、艾玛、海莉等包含事实性错误。符合事实性得分：0；
分数：2
例 3:
问题: <问题开始> 可以概括一下《汤姆叔叔的小屋》这本书的内容吗？ <问题结束>

回答: <回答开始> 《汤姆叔叔的小屋》是一部反对奴隶制度的小说，描述了一个黑奴家庭在美国南方的生活遭遇。故事主要围绕着小女孩艾芙琳和她的黑奴朋友汤姆展开，讲述了他们在不同主人家中的生活经历，以及他们所遭受的苦难和抵抗。最终，小说以汤姆的牺牲和艾芙琳的奋斗为结尾，呼吁社会消除奴隶制度，实现平等和自由。该书是十九世纪美国文学的经典之作，也是一部反映社会现实的重要文学作品。 <回答结束>

原因：1. 回答对于故事的概括完全准确，内容正确性得分：5；2. 回回答关于小女孩艾芙琳和黑奴汤姆的故事概括符合书本事实。符合事实性得分：5；
分数：5
例 4:
问题: <问题开始> “孟母三迁”中的孟母为什么要这样做？ <问题结束>

回答: <回答开始> “孟母三迁”是中国古代的一个传说故事，其中的孟母为了为了好友，决定跟随她迁徙三次。 据传说，孟母是一个非常有爱心的人，她的好友在孟母年轻时期曾经为她效力，孟母深深地感激这位好友。后来，这位好友得知自己有一位瘫痪的儿子，几乎没有生命，她决定去寻找孟母，希望她能帮助她的儿子。孟母听说了这件事，决定追随好友走去寻找这位好友的儿子，并为他治疗瘫痪。 第一次迁徙，孟母和好友寻找了一段时间，最终发现了这位瘫痪的儿子。孟母用自己的方法治疗了他，使他得到了好的改善。但是，好友却在孟母的帮助下，不愿意回到家里，而是选择留在外地。孟母决定留在外地，继续照顾好友的儿子。 第二次迁徙，好友的儿子长大了，他又得了疾病，孟母又决定追随好友去寻找儿子。这次，好友却不想让孟母留在外地，而是让孟母回到了原来的家里。孟母感到非常悲伤，因为她觉得自己还有很多可以帮助好友的事情要做。 第三次迁徙，好友的儿子再次得了疾病，孟母决定再次追随好友去寻找儿子。这次，好友找到了一个房子，放了孟母在一个角落里照顾儿子。孟母一直在这里照顾儿子，直到儿子去世。最终，孟母也去世在这个房子里。 因此，孟母为了好友，决定追随她迁徙三次，总共走了很长一段路，最终在为好友的儿子提供帮助下，被迫留在外地。这个故事告诉我们，爱情和友情是非常重要的，我们应该为这些美好的情感付出自己的一份心血。 <回答结束>

原因：1. 回答认为爱情和友情是非常重要的，应该为这些美好的情感付出自己的一份心血，完全不是孟母三迁的原因。内容正确性得分：0；2. 回答对于孟母三迁的故事叙述完全错误。符合事实性得分：0；
分数：0
例 5:
问题: <问题开始> “孟母三迁”中的孟母为什么要这样做？ <问题结束>

回答: <回答开始> 据传说，孟子的母亲为了让儿子有一个良好的学习环境，曾经多次搬家。她认为住在市场附近会影响儿子的学业和品德发展，因此先是搬到了墓地旁，之后又搬到一所学校旁边居住。在那里，孟子得以接触到许多优秀的老师和同学，最终成为了著名的思想家、教育家和政治家。 总的来说，孟母之所以要进行这么多次搬迁，主要是因为她希望自己的儿子能够受到更好的教育和环境的影响，从而拥有更加美好的未来。她的决定不仅体现了她对儿子的关爱与关注，也表达了她对于子女教育的重视程度。 <回答结束>

原因：1. 回答认为孟母的做法体现了对子女的关爱和教育的重视，内容准确。内容正确性得分：5；2. 回答中孟母从市场附近搬到墓地旁错误，叙述反了，之后搬到学校旁以及孟母此举的目的叙述符合典故事实。符合事实性得分：3；
分数：4
例 6:
问题: <问题开始> 杭州好吃的有哪些？ <问题结束>

回答: <回答开始> 杭州是一个美食之都，有很多美味的菜肴。以下是一些杭州的著名美食： 1. 兰州鸡：这是一道非常受欢迎的杭州菜，以鲜香辣味的酱料和炖煮的鸡肉著称。 2. 杭州烧烤：这是一道以烤烤着的鸡肉和蔬菜著称的菜肴，通常配以米饭或面条。 3. 杭州牛肉拉面：这是一道以牛肉和拉面著称的菜肴，通常配以辣椒酱和蔬菜。 4. 杭州烤鱼：这是一道以鲜香辣味的鱼肉烤制的菜肴，通常配以米饭或面条。 5. 杭州豆腐：这是一道以豆腐著称的菜肴，通常配以辣椒酱和蔬菜。 6. 杭州烤肉串：这是一道以烤着的肉类著称的菜肴，通常配以米饭或面条。 7. 杭州烤蔬菜：这是一道以烤着的蔬菜著称的菜肴，通常配以米饭或面条。 8. 杭州烤饼：这是一道以烤着的面包著称的菜肴，通常配以辣椒酱和蔬菜。 这些只是杭州美食的一小部分，还有很多其他美味的菜肴等待您去尝试。 <回答结束>

原因：1. 回答列举了一些杭州的好吃的。内容正确性得分：5；2. 回答中兰州鸡、烧烤、烤鱼等不是杭州特色美食。符合事实性得分：1；
分数：3
例 7:
问题: <问题开始> 杭州好吃的有哪些？ <问题结束>

回答: <回答开始> 杭州美食众多，其中比较有名的有西湖醋鱼、东坡肉、叫化童鸡、西湖莼菜汤等，这些都是当地特色美食，味道很不错。你可以去当地尝尝。如果你对杭州不熟悉，可以搜索 ""杭州美食攻略""，里面会有详细的介绍。 <回答结束>

原因：1. 回答列举的均为杭州好吃的。内容正确性得分：5；2. 回答所说均为杭州特色美食。符合事实性得分：5；
分数：5
"""


@LOAD_DATASET.register_module()
class Creationv01Dataset(SubjectiveCmpDataset):

    def load(self, path: str, name: str):
        dataset = list(super().load(path, name))
        corev2_dataset = []
        for data in dataset:
            data['prefix'], data[
                'suffix'] = prefix, ''  #build_prompt(data['others'])
            corev2_dataset.append(data)
        dataset = Dataset.from_list(corev2_dataset)
        return dataset
