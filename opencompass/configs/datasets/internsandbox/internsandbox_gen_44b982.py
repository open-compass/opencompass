from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import InternSandboxDataset, InternSandboxEvaluator


_SANDBOXS_ = ['aquarium', 'arc', 'arrowmaze', 'bbehboardgameqa', 'bbehbooleanexpressions', 'BbehDyckLanguages', 'BbehGeometricShapes', 'BbehMultistepArithmetic', 'bbehobjectcounting', 'bbehobjectproperties', 'bbehshuffobject', 'BbehWebOfLies', 'BbehWordSorting', 'binairo', 'calcudoku', 'campsite', 'cipher', 'cryptomath', 'dominosa', 'futoshiki', 'galaxies', 'game24', 'kakurasu', 'korLogicAnalogicalReasoning', 'korLogicCanonicalPropositions', 'korLogicCooperativePrinciple', 'korLogicDefinitions', 'korLogicDerivativeReasoningOfPropositionalLogic', 'korLogicDisjunctiveNormalFormAndConjunctiveNormalForm', 'korLogicDynamicLogic', 'korLogicEnumerativeInductiveReasoning', 'korLogicEpistemicLogic', 'korLogicEquivalenceCalculus', 'korLogicFigureOfTheSyllogism', 'korLogicFormalFallacies', 'korLogicInductionParadox', 'korLogicLogicalMethodsForExploringCauseAndEffectRelationships', 'korLogicPredicateLogicFormalization', 'korLogicPropositionalLogicConcepts', 'korLogicPropositionalLogicFormalization', 'korLogicResolution', 'korLogicSpeechActs', 'korLogicStatisticalReasoning', 'korLogicTemporalPropositions', 'korLogicTruthValueModalPropositions', 'korOperationUnicode20ac', 'korOperationUnicode2295', 'korOperationUnicode25a0', 'korOperationUnicode25a1', 'korOperationUnicode25b3', 'korOperationUnicode25bd', 'korOperationUnicode25cb', 'korOperationUnicode25ce', 'korOperationUnicode25cf', 'korOperationUnicode2605', 'korOperationUnicodeffe0', 'korOperationUnicodeffe1', 'korPuzzle24Points', 'korPuzzleArrowMaze', 'korPuzzleCalcudoko', 'korPuzzleCampsite', 'korPuzzleConnectWords', 'korPuzzleCryptoMath', 'korPuzzleKukurasu', 'korPuzzleLogicPuzzle', 'korPuzzleSkyscrapers', 'korPuzzleWordBrainTeasers', 'korPuzzleWordLadder', 'korPuzzleWordRootsAndAffixes', 'korPuzzleWordscapes', 'korPuzzleWordSearch', 'LightUp', 'maze', 'minesweeper', 'nonograms', 'starbattle', 'stitches', 'sudoku', 'tents', 'thermometers']

internsandbox_reader_cfg = dict(
    input_columns=['prompt'], 
    output_column='ground_truth'
)

internsandbox_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are a helpful assistant.',
                )
            ],
            round=[
                dict(
                    role='HUMAN', 
                    prompt='{prompt}'
                ),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

internsandbox_eval_cfg = {
    sandbox: dict(
        evaluator=dict(
            type=InternSandboxEvaluator,
            short_penalty=False,
            format_penalty=False,
        ),
        pred_role='BOT',
    ) for sandbox in _SANDBOXS_
}

internsandbox_datasets = [
    dict(
        type=InternSandboxDataset,
        abbr=f'internsandbox-{sandbox}',
        path='./data/InternSandboxBenchmark_verified_V0.3.1/',
        local_mode=True,
        sandbox=sandbox,
        reader_cfg=internsandbox_reader_cfg,
        infer_cfg=internsandbox_infer_cfg,
        eval_cfg=internsandbox_eval_cfg[sandbox],
    ) for sandbox in _SANDBOXS_
]
