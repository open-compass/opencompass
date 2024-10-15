# GaoKao MATH Answer Evaluation Dataset
A dataset for testing the performance of the model in the GaoKao MATH Answer Extraction task.
Now support the following format of GAOKAO math questions:
1. '单选题'：Single choice question
2. '多选题'：Multiple choice question
3. '填空题'：Fill in the blank question, can be multiple blanks
4. '解答题'：Answer question, can be multiple answers

Sample data:
```json
[
    {
        "id": "3b270bc4-570a-4d77-b122-a2fc372f7d6a",
        "question": "过椭圆${x^2\\over {16}} +{ y^2 \\over {4}}=1$ %内一点$M(2,1)$ %引一条弦，使该弦被点$M$ %平分，则这条弦所在直线的方程为（ ）．\nA. $x+2y-4=0$ %\nB. $x-2y-4=0$ %\nC. $x+2y+4=0$ %\nD. $x-2y+4=0$ %\n\n",
        "response": "本题主要考查直线与圆锥曲线．设所求直线与椭圆的一个交点为$A(x,y)$ %，由于中点$M(2,1)$ %，所以另一个交点$B$ %为$(4-x,2-y)$ %．因为$A$ %，$B$ %两点都在椭圆上，所以$x^2+4y^2=16$ %，$(4-x)^2+4(2-y)^2=16$ %，两式相减，整理可得$x+2y-4=0$ %．由于过$A$ %，$B$ %两点的直线只有一条，所以这条弦所在直线的方程为$x+2y-4=0$ %．故本题正确答案为A．\n答案是：A",
        "extract_answer": "A",
        "question_type": "单选题"
    },
    {
        "id": "d60e42d7-30ee-44f9-a94d-aff6a8127750",
        "question": "若函数$f(x)$ 具有下列性质：1.定义域为$(-1,1)$ ；2.对于任意的$x,y\\in(-1,1)$ ，都有$f(x)+f(y)=f\\left({\\dfrac{x+y}{1+xy}}\\right)$ ；3.当$-1< x< 0$ 时，$f(x)>0$ ，则称函数$f(x)$ 为$δ$ 的函数$.$ 若函数$f(x)$ 为$δ$ 的函数，则以下结论正确的是$(\\quad)$\nA. $\nB. x)$ 为奇函数\nC. $\nD. x)$ 为偶函数\nE. $\nF. x)$ 为单调递减函数\nG. $\nH. x)$ 为单调递增函数\n\n",
        "response": "函数$f(x)$ 为$δ$ 的函数，令$x=y=0$ ，则$f(0)+f(0)=f(0)$ ，即$f(0)=0$ ，令$y=-x$ ，则$f(x)+f(-x)=f\\left(\\dfrac{x-x}{1-{x}^{2}}\\right)=f(0)=0$ ，则$f(-x)=-f(x)$ ，即函数$f(x)$ 是奇函数，设$-1< x< y< 1$ ，则$f(x)-f(y)=f(x)+f(-y)=f\\left(\\dfrac{x-y}{1-xy}\\right)$ ，$∵-1< x< y< 1$ ，$∴-1< \\dfrac{x-y}{1-xy}< 0$ ，则$f\\left(\\dfrac{x-y}{1-xy}\\right)>0$ ，即$f(x)-f(y)>0$ ，则$f(x)>f(y)$ ，即$f(x)$ 在$(-1,1)$ 上是减函数.故选$AC.$ 本题考查函数的奇偶性和单调性的判断，注意运用定义法，考查运算能力和推理能力，属于中档题.可令$x=y=0$ ，求得$f(0)=0$ ，再令$y=-x$ 可得$f(-x)=-f(x)$ ，可得$f(x)$ 的奇偶性；再令$-1< x< y< 1$ ，运用单调性的定义，结合其偶性的定义可得其单调性．\n答案是：A; C",
        "extract_answer": "A, C",
        "question_type": "多选题"
    },
    {
        "id": "31b3f702-e60c-4a20-9a40-73bd72b92d1e",
        "question": "请完成以下题目(1)曲线$$y=-5\\text{e}^{x}+3$$在点$$(0,-2)$$处的切线方程为___.(2)若曲线$$f(x)=x \\sin x+1$$在$$x=\\dfrac{ \\pi }{2}$$处的切线与直线$$ax+2y+1=0$$相互垂直,则实数$$a=$$___.\n\n",
        "response": "(1)由$$y=-5\\text{e}^{x}+3$$,得$$y'=-5\\text{e}^{x}$$,所以切线的斜率$$k=y'|_{x=0}=-5$$,所以切线方程为$$y+2=-5(x-0)$$,即$$5x+y+2=0$$.(2)因为$$f'(x)= \\sin x+x \\cos x$$,所以$$f'\\left(\\dfrac{ \\pi }{2}\\right)= \\sin \\dfrac{ \\pi }{2}+\\dfrac{ \\pi }{2}\\cdot \\cos \\dfrac{ \\pi }{2}=1$$.又直线$$ax+2y+1=0$$的斜率为$$-\\dfrac{a}{2}$$,所以根据题意得$$1\\times \\left(-\\dfrac{a}{2}\\right)=-1$$,解得$$a=2$$.\n答案是：(1)$$5x+y+2=0$$ (2)$$2$$",
        "extract_answer": "['(1)$$5x+y+2=0$$ (2)$$2$$']",
        "question_type": "填空题"
    },
    {
        "id": "16878941-1772-4290-bc61-00b193d5cf70",
        "question": "已知函数$f\\left( x \\right)=\\left| 2x-1 \\right|$.（1）若不等式$f\\left( x+\\frac{1}{2} \\right)\\ge 2m+1\\left( m > 0 \\right)$的解集为$\\left( -\\infty ,-2 \\right]\\bigcup \\left[ 2,+\\infty \\right)$，求实数$m$的值；（2）若不等式$f\\left( x \\right)\\le {{2}^{y}}+\\frac{a}{{{2}^{y}}}+\\left| 2x+3 \\right|$对任意的实数$x,y\\in R$恒成立，求实数$a$的最小值.\n\n",
        "response": "（1）直接写出不等式，解含有绝对值的函数不等式即可；（2）这是恒成立求参的问题,根据绝对值三角不等式得到左侧函数的最值，再结合均值不等式得最值.（1）由条件得$\\left| 2x \\right|\\le 2m+1$得$-m-\\frac{1}{2}\\le x\\le m+\\frac{1}{2}$，所以$m=\\frac{3}{2}$.（2）原不等式等价于$\\left| 2x-1 \\right|-\\left| 2x+3 \\right|\\le {{2}^{y}}+\\frac{a}{{{2}^{y}}}$，而$\\left| 2x-1 \\right|-\\left| 2x+3 \\right|\\le \\left| \\left( 2x-1 \\right)-\\left( 2x+3 \\right) \\right|=4$，所以${{2}^{y}}+\\frac{a}{{{2}^{y}}}\\ge 4$，则$a\\ge {{\\left[ {{2}^{y}}\\left( 4-{{2}^{y}} \\right) \\right]}_{\\text{max}}}=4$，当且仅当$y=1$时取得.\n答案是：(1) $m=\\frac{3}{2}$；(2) 最小值为$a=4$.",
        "extract_answer": [
            "(1) $m=\\frac{3}{2}$；(2) 最小值为$a=4$."
        ],
        "question_type": "解答题"
    }
]
```
## How to use

### 1. Prepare the dataset
```bash
cd opencompass
cp -rf /cpfs01/shared/public/liuhongwei/data/gaokao_math_dataset/gaokao_math ./data
```
📢：If you want to evaluate your own gaokao math data, replace the `test_v2.jsonl` with your own data, but follow the format above.

### 2. Set the evaluation model

open `opencompass.datasets.gaokao_math.gaokao_math_gen_9b076f` and set the model name and api url for evaluation, multiple urls are supported for acceleration.

```python
...

gaokao_math_eval_cfg = dict(
    evaluator=dict(type=GaoKaoMATHEvaluator, model_name='EVALUATE_MODEL_NAME', url=['http://0.0.0.0:23333/v1', 'http://...']))

...

```
We recommand `Qwen2.5-72B-Instruct` model for evaluation.


### 3. Set Extractor model and run the evaluation

```python
from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate


with read_base():
    from opencompass.datasets.gaokao_math.gaokao_math_gen_9b076f import gaokao_math_datasets


trained_qwen2_1_5b_model = [ # trained extractor model
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gaokao_math_extractor_1_5b_v02',
        path='/cpfs01/shared/public/liuhongwei/models/gaokao_math_trained/gaokao_math_extractor_1_5b_v02',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = sum([v for k, v in locals().items() if k.endswith("_datasets")], [])
models = sum([v for k, v in locals().items() if k.endswith("_model")], [])

...
```

### 4. Run the evaluation

```bash
python run.py eval.py --dump-eval-details # eval and dump the evaluation details to `results` folder
```


### 5. Evaluation results

| Evaluator / Extractor | Qwen2.5-72B-Instruct | gaokao_math_extractor_1.5b_v0.2 |
|-----------------------|-----------------------|----------------------------------|
| Qwen2.5-72B-Instruct (ACC) | 95.85 | 95.2 |
