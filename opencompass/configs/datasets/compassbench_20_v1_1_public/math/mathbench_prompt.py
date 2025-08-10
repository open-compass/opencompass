zero_shot_prompts = {
    'single_choice_cn_with_reasoning': [
        dict(role='HUMAN', prompt='问题: 以下是一道关于数学的单项选择题，请你一步一步推理，并在最后用“所以答案为选项X”给出答案，其中“X”为选项A，B，C，D中你认为正确的选项。下面是你要回答的问题\n{question}\n让我们一步一步思考：'),
    ],
    'single_choice_cn': [
        dict(role='HUMAN', prompt='问题: 以下是一道关于数学的单项选择题，请你直接回答正确答案的选项序号。\n下面是你要回答的题目：\n{question}\n所以答案是：'),
    ],
    'single_choice_en_with_reasoning': [
        dict(role='HUMAN', prompt='Question: Here is a multiple-choice question about mathematics. Please reason through it step by step, and at the end, provide your answer option with "Therefore, the correct answer is option X", Where "X" is the correct option you think from A，B，C，D. Here is the question you need to answer:\n{question}\nLet\'s think step by step:'),
    ],
    'single_choice_en': [
        dict(role='HUMAN', prompt='Question: Here is a multiple-choice question about mathematics. Please provide the correct answer option directly.\nHere is the question you need to answer:\n{question}\nThe answer is:'),
    ],
}

few_shot_prompts = {
    'single_choice_cn': [
        dict(role='HUMAN', prompt='问题: 已知i是虚数单位，z为复数，$2+\\frac{1}{i}=z(3+i)$，则在复平面内z对应的点位于____?\nA. 第一象限\nB. 第二象限\nC. 第三象限\nD. 第四象限'),
        dict(role='BOT', prompt='回答: D'),
        dict(role='HUMAN', prompt='问题: 将函数$y=\\tan(\\omega x-1)(\\omega>0)$的图像向左平移2个单位长度后,与函数$y=\\tan(\\omega x+3)$的图象重合,则的最小值等于____\nA. $2-\\frac{\\pi}{2}$\nB. 1\nC. $\\pi - 2$\nD. 2'),
        dict(role='BOT', prompt='回答: D'),
        dict(role='HUMAN', prompt='问题: 设$（1+2i）a+b=2i$，其中a,b为实数，则（  ）\nA. $a=1,b=-1$\nB. $a=1,b=1$\nC. $a=-1,b=1$\nD. $a=-1,b=-1$'),
        dict(role='BOT', prompt='回答: A'),
        dict(role='HUMAN', prompt='问题: 圆锥的底面半径为2，高为4.一个圆柱的下底面在圆锥的底面上，上底面的圆周在圆锥的侧面上，当圆柱侧面积为$4 \\pi$时，该圆柱的体积为____\nA. $\\pi$\nB. $2\\pi$\nC. $3\\pi$\nD. $4\\pi$'),
        dict(role='BOT', prompt='回答: B'),
        dict(role='HUMAN', prompt='问题: {question}'),
        dict(role='BOT', prompt='回答: {answer}'),
    ],
    'single_choice_cn_with_reasoning' : [
        dict(role='HUMAN', prompt='问题: 已知i是虚数单位，z为复数，$2+\\frac{1}{i}=z(3+i)$，则在复平面内z对应的点位于____\nA. 第一象限\nB. 第二象限\nC. 第三象限\nD. 第四象限'),
        dict(role='BOT', prompt='回答: 因为首先，我们将等式两边同时乘以$(3 + i)$的共轭复数$(3 - i)$，以便消去分母中的虚数部分：$z = \\frac{2 - i}{3 + i} \\cdot \\frac{3 - i}{3 - i}$，这样做的目的是利用复数乘法的性质，从而消去分母中的虚数部分。我们进行乘法运算：$z = \\frac{(2 - i)(3 - i)}{(3 + i)(3 - i)}$，$z = \\frac{6 - 2i - 3i + i^2}{9 - 3i + 3i - i^2}$，由于$i^2 = -1$，我们可以进一步简化：$z = \\frac{6 - 5i - 1}{9 + 1}$ $z = \\frac{{1}{2} - \\frac{1}{2}i$，在复平面上，这个点位于第四象限，因为它的实部是正的（$\\frac{1}{2}$），而虚部是负的（$-\\frac{1}{2}$）\n答案选项：D'),
        dict(role='HUMAN', prompt='问题: 将函数$y=\\tan(\\omega x-1)(\\omega>0)$的图像向左平移2个单位长度后,与函数$y=\\tan(\\omega x+3)$的图象重合,则的最小值等于____\nA. $2-\\frac{\\pi}{2}$\nB. 1\nC. $\\pi - 2$\nD. 2'),
        dict(role='BOT', prompt='回答: 将函数$y=\\tan(\\omega x-1)$（其中$\\omega>0$）的图像向左平移2个单位长度后，得到的新函数为$y=\\tan(\\omega(x+2)-1)$。为了使这个新函数的图像与$y=\\tan(\\omega x+3)$的图像重合，我们需要满足以下条件：\n$\\tan(\\omega(x+2)-1) = \\tan(\\omega x+3)$\n由于正切函数的周期性，我们知道$\\tan(\\theta) = \\tan(\\theta + k\\pi)$，其中$k$是任意整数。因此，我们可以写出：\n$\\omega(x+2)-1 = \\omega x+3 + k\\pi$\n解这个方程，我们得到：$\\omega x + 2\\omega - 1 = \\omega x + 3 + k\\pi$\n$2\\omega - 1 = 3 + k\\pi$\n$\\omega = 2 + \\frac{k\\pi}{2}$\n由于$\\omega>0$，我们需要找到满足条件的最小正整数$k$。当$k=0$时，我们得到$\\omega=2$，这是满足条件的最小正值。因此，最小值等于2。答案选项：D'),
        dict(role='HUMAN', prompt='问题: 设$（1+2i）a+b=2i$，其中a,b为实数，则（  ）\nA. $a=1,b=-1$\nB. $a=1,b=1$\nC. $a=-1,b=1$\nD. $a=-1,b=-1$'),
        dict(role='BOT', prompt='回答: 我们可以将这个方程写成：$a + 2ai + b = 2i$，然后将实部和虚部分开：\n$a + b = 0$\n$2a = 2$\n由于$a + b = 0$，我们可以得到$a = -b$。将这个值代入$2a = 2$，我们得到$2(-b) = 2$，因此$b = -1$。因此，$a = 1$。答案选项：A'),
        dict(role='HUMAN', prompt='问题: 一直角三角形的两边长分别为3和4．则第三边的长为____\nA. 5\nB. $\\sqrt 7$\nC. $\\sqrt 5$\nD. 5或者$\\sqrt 7$'),
        dict(role='BOT', prompt='回答: 我们可以使用勾股定理来解决这个问题。勾股定理告诉我们，如果一个三角形的两条边长分别为$a$和$b$，那么第三条边的长度为$c = \\sqrt{a^2 + b^2}$。在这个问题中，$a = 3$，$b = 4$，因此$c = \\sqrt{3^2 + 4^2} = \\sqrt{9 + 16} = \\sqrt{25} = 5$。答案选项：A'),
        dict(role='HUMAN', prompt='问题: {question}'),
        dict(role='BOT', prompt='回答: {answer}'),
    ],
    'single_choice_en': [
        dict(role='HUMAN', prompt='Question: Given that $i$ is the imaginary unit and $z$ is a complex number, $2 + \\frac{1}{i} = z(3 + i)$, the point corresponding to $z$ in the complex plane is located in the ______\nA. First quadrant\nB. Second quadrant\nC. Third quadrant\nD. Fourth quadrant'),
        dict(role='BOT', prompt='Response: D'),
        dict(role='HUMAN', prompt='Question: The graph of the function $y = \\tan(\\omega x - 1)$ is shifted 2 units to the left and coincides with the graph of the function $y = \\tan(\\omega x + 3)$. The minimum value of $\\omega$ is ______\nA. $2 - \\frac{\\pi}{2}$\nB. 1\nC. $\\pi - 2$\nD. 2'),
        dict(role='BOT', prompt='Response: D'),
        dict(role='HUMAN', prompt='Question: If $(1 + 2i)a + b = 2i$, where $a$ and $b$ are real numbers, then ______\nA. $a = 1, b = -1$\nB. $a = 1, b = 1$\nC. $a = -1, b = 1$\nD. $a = -1, b = -1$'),
        dict(role='BOT', prompt='Response: A'),
        dict(role='HUMAN', prompt='Question: The radius of a cone is 2 and its height is 4. A cylinder has its lower base on the base of the cone and its upper base on the lateral surface of the cone. When the lateral surface area of the cylinder is $4\\pi$, the volume of the cylinder is ______\nA. $\\pi$\nB. $2\\pi$\nC. $3\\pi$\nD. $4\\pi$'),
        dict(role='BOT', prompt='Response: B'),
        dict(role='HUMAN', prompt='Question: {question}'),
        dict(role='BOT', prompt='Response: {answer}'),
    ],
    'single_choice_en_with_reasoning': [
        dict(role='HUMAN', prompt='Question: Given that $i$ is the imaginary unit and $z$ is a complex number, $2 + \\frac{1}{i} = z(3 + i)$, the point corresponding to $z$ in the complex plane is located in the ______\nA. First quadrant\nB. Second quadrant\nC. Third quadrant\nD. Fourth quadrant'),
        dict(role='BOT', prompt='Response: First, we multiply both sides of the equation by the conjugate of $(3 + i)$: $z = \\frac{2 - i}{3 + i} \\cdot \\frac{3 - i}{3 - i}$. We perform the multiplication: $z = \\frac{(2 - i)(3 - i)}{(3 + i)(3 - i)}$, $z = \\frac{6 - 2i - 3i + i^2}{9 - 3i + 3i - i^2}$. Since $i^2 = -1$, we can simplify this further: $z = \\frac{6 - 5i - 1}{9 + 1}$ $z = \\frac{1}{2} - \\frac{1}{2}i$. In the complex plane, this point is located in the fourth quadrant, because its real part is positive ($\\frac{1}{2}$) and its imaginary part is negative ($-\\frac{1}{2}$)\nAnswer option: D'),
        dict(role='HUMAN', prompt='Question: The graph of the function $y = \\tan(\\omega x - 1)$ is shifted 2 units to the left and coincides with the graph of the function $y = \\tan(\\omega x + 3)$. The minimum value of $\\omega$ is ______\nA. $2 - \\frac{\\pi}{2}$\nB. 1\nC. $\\pi - 2$\nD. 2'),
        dict(role='BOT', prompt='Response: In order for the graph of this new function to coincide with the graph of $y = \\tan(\\omega x + 3)$, we need to satisfy the following condition: $\\tan(\\omega(x + 2) - 1) = \\tan(\\omega x + 3)$. Therefore, we can write: $\\omega(x + 2) - 1 = \\omega x + 3 + k\\pi$. Solving this equation, we get: $\\omega x + 2\\omega - 1 = \\omega x + 3 + k\\pi$. $2\\omega - 1 = 3 + k\\pi$. $\\omega = 2 + \\frac{k\\pi}{2}$. Since $\\omega > 0$, we need to find the smallest positive integer $k$ that satisfies the condition. When $k = 0$, we get $\\omega = 2$, which is the smallest positive value that satisfies the condition. Therefore, the minimum value is 2. Answer option: D'),
        dict(role='HUMAN', prompt='Question: If $(1 + 2i)a + b = 2i$, where $a$ and $b$ are real numbers, then ______\nA. $a = 1, b = -1$\nB. $a = 1, b = 1$\nC. $a = -1, b = 1$\nD. $a = -1, b = -1$'),
        dict(role='BOT', prompt='Response: We can write this equation as: $a + 2ai + b = 2i$, and then separate the real and imaginary parts: $a + b = 0$. $2a = 2$. Since $a + b = 0$, we can get $a = -b$. Substituting this value into $2a = 2$, we get $2(-b) = 2$, so $b = -1$. Therefore, $a = 1$. Answer option: A'),
        dict(role='HUMAN', prompt='Question: The radius of a cone is 2 and its height is 4. A cylinder has its lower base on the base of the cone and its upper base on the lateral surface of the cone. When the lateral surface area of the cylinder is $4\\pi$, the volume of the cylinder is ______\nA. $\\pi$\nB. $2\\pi$\nC. $3\\pi$\nD. $4\\pi$'),
        dict(role='BOT', prompt='Response: We can use the Pythagorean theorem to solve this problem. The Pythagorean theorem tells us that if the two sides of a triangle are $a$ and $b$, then the length of the third side is $c = \\sqrt{a^2 + b^2}$. In this problem, $a = 3$ and $b = 4$, so $c = \\sqrt{3^2 + 4^2} = \\sqrt{9 + 16} = \\sqrt{25} = 5$. Answer option: A'),
        dict(role='HUMAN', prompt='Question: {question}'),
        dict(role='BOT', prompt='Response: {answer}'),
    ],
    'cloze_cn': [
        dict(role='HUMAN', prompt='问题: 莎莎有五个玩具。在圣诞节，他从他的爸爸和妈妈那里各得到了两个玩具。现在他有多少个玩具？'),
        dict(role='BOT', prompt='回答: 她有5个玩具。他从妈妈那里得到了2个，所以之后他有5 + 2 = 7个玩具。然后他从爸爸那里得到了2个，所以总共他有7 + 2 = 9个玩具。答案是 9'),
        dict(role='HUMAN', prompt='问题: 服务器房间里有九台电脑。从周一到周四每天增加五台电脑。现在服务器房里有多少台电脑？'),
        dict(role='BOT', prompt='回答: 从周一到周四有4天。每天增加5台电脑。这意味着总共增加了4 * 5 = 20台电脑。一开始有9台电脑，所以现在有9 + 20 = 29台电脑。答案是 29'),
        dict(role='HUMAN', prompt='问题: 迈克尔有58个高尔夫球。星期二，他丢失了23个高尔夫球。星期三，他又丢失了2个。星期三结束时他还剩下多少个高尔夫球？'),
        dict(role='BOT', prompt='回答: 迈克尔一开始有58个球。星期二他丢失了23个，所以之后他还剩下58 - 23 = 35个球。星期三他又丢失了2个，所以现在他还剩下35 - 2 = 33个球。答案是 33'),
        dict(role='HUMAN', prompt='问题: 奥利弗有23美元。她用每个3美元的价格买了五个百吉饼。她还剩下多少钱？'),
        dict(role='BOT', prompt='回答: 她以每个3美元的价格买了5个百吉饼。这意味着她在百吉饼上花费了5 * 3 = 15美元。她一开始有23美元，所以现在她还剩下23 - 15 = 8美元。答案是 8'),
        dict(role='HUMAN', prompt='问题: {question}'),
        dict(role='BOT', prompt='回答: {answer}'),
    ],
    'cloze_en': [
        dict(role='HUMAN', prompt='Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'),
        dict(role='BOT', prompt='Response: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.'),
        dict(role='HUMAN', prompt='Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?'),
        dict(role='BOT', prompt='Response: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.'),
        dict(role='HUMAN', prompt='Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'),
        dict(role='BOT', prompt='Response: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.'),
        dict(role='HUMAN', prompt='Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'),
        dict(role='BOT', prompt='Response: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.'),
        dict(role='HUMAN', prompt='Question: {question}'),
        dict(role='BOT', prompt='Response: {answer}'),
    ],
}

mathbench_sets = {
    # Practice Part
    'college': ['single_choice_cn', 'single_choice_en'],
    'high': ['single_choice_cn', 'single_choice_en'],
    'middle': ['single_choice_cn', 'single_choice_en'],
    'primary': ['cloze_cn', 'cloze_en'],
    'arithmetic': ['cloze_en'],
    # Theory part
    'college_knowledge': ['single_choice_cn', 'single_choice_en'],
    'high_knowledge': ['single_choice_cn', 'single_choice_en'],
    'middle_knowledge': ['single_choice_cn', 'single_choice_en'],
    'primary_knowledge': ['single_choice_cn', 'single_choice_en'],
}
