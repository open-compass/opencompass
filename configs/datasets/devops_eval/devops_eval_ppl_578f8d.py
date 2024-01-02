from opencompass.datasets.devops_eval import DevOpsEvalDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator

devops_eval_subject_mapping = {
    "Visualization": "MONITOR/Visualization.csv",
    "Logging": "MONITOR/Data/Logging.csv",
    "Storage": "MONITOR/Data/Storage.csv",
    "DataAcquisition": "MONITOR/Data/DataAcquisition.csv",
    "IntegrationTesting": "TEST/IntegrationTesting.csv",
    "UserAcceptanceTesting": "TEST/UserAcceptanceTesting.csv",
    "SecurityTesting": "TEST/SecurityTesting.csv",
    "UnitTesting": "TEST/UnitTesting.csv",
    "PerformanceTesting": "TEST/PerformanceTesting.csv",
    "SystemTesting": "TEST/SystemTesting.csv",
    "ProgM": "PLAN/ProgM.csv",
    "REQM": "PLAN/REQM.csv",
    "RiskMgmt": "PLAN/RiskMgmt.csv",
    "InfrastructureAsCode": "DEPLOY/InfrastructureAsCode.csv",
    "Provisioning": "DEPLOY/Provisioning.csv",
    "ConfigMgmt": "DEPLOY/ConfigMgmt.csv",
    "Azure": "DEPLOY/Cloud:IaaS:PaaS/Azure.csv",
    "GoogleCloud": "DEPLOY/Cloud:IaaS:PaaS/GoogleCloud.csv",
    "AWS": "DEPLOY/Cloud:IaaS:PaaS/AWS.csv",
    "LogDesign": "CODE/Design/LogDesign.csv",
    "ServiceDesign": "CODE/Design/ServiceDesign.csv",
    "CapabilityDesign": "CODE/Design/CapabilityDesign.csv",
    "CloudNativeDesign": "CODE/Design/CloudNativeDesign.csv",
    "CacheDesign": "CODE/Design/CacheDesign.csv",
    "DBDesign": "CODE/Design/DBDesign.csv",
    "ArtificialIntelligence": "CODE/Develop/GeneralKnowledge/ArtificialIntelligence.csv",
    "ComputerBasics": "CODE/Develop/GeneralKnowledge/ComputerBasics.csv",
    "DataBase": "CODE/Develop/GeneralKnowledge/DataBase.csv",
    "ComputerNetwork": "CODE/Develop/GeneralKnowledge/ComputerNetwork.csv",
    "OperatingSystem": "CODE/Develop/GeneralKnowledge/OperatingSystem.csv",
    "Go": "CODE/Develop/ProgrammingLanguage/Go.csv",
    "Java": "CODE/Develop/ProgrammingLanguage/Java.csv",
    "C:C++": "CODE/Develop/ProgrammingLanguage/C:C++.csv",
    "Python": "CODE/Develop/ProgrammingLanguage/Python.csv",
    "BigData": "CODE/Develop/Frameworks&Libraries/BigData.csv",
    "Front-end": "CODE/Develop/Frameworks&Libraries/Front-end.csv",
    "MobileApp": "CODE/Develop/Frameworks&Libraries/MobileApp.csv",
    "MachineLearning": "CODE/Develop/Frameworks&Libraries/MachineLearning.csv",
    "Back-end": "CODE/Develop/Frameworks&Libraries/Back-end.csv",
    "ArtifactMgmt": "RELEASE/ArtifactMgmt.csv",
    "CI:CD": "RELEASE/CI:CD.csv",
    "Linux": "RELEASE/OperatingSystem/Linux.csv",
    "ContainerOrchestration": "OPERATE/ContainerOrchestration.csv",
    "Virtualization": "OPERATE/Virtualization.csv",
    "TimeSeriesAnomalyDetection": "OPERATE/AIOps/TimeSeriesAnomalyDetection.csv",
    "TimeSeriesClassification": "OPERATE/AIOps/TimeSeriesClassification.csv",
    "RootCauseAnalysis": "OPERATE/AIOps/RootCauseAnalysis.csv",
    "LogParser": "OPERATE/AIOps/LogParser.csv",
    "VersionControl": "BUILD/VersionControl.csv",
    "DBMgnt": "BUILD/DBMgnt.csv",
    "Dependency": "BUILD/Build/Dependency.csv",
    "Compile": "BUILD/Build/Compile.csv",
    "Package": "BUILD/Build/Package.csv",
    "OperateScene": "OPERATE/OperateScene.csv",
    "TimeSeriesForecasting": "OPERATE/AIOps/TimeSeriesForecasting.csv"
    }
devops_eval_all_sets = list(devops_eval_subject_mapping.keys())

devops_eval_datasets = []
zh_prompt = f"以下是关于DevOps相关的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: "
en_prompt = f"Here is a multiple-choice question related to DevOps; please select the correct answer.\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nthe answer: "

for language in ["zh", "en"]:
    _split = "test"
    for _name in devops_eval_all_sets:
        devops_eval_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template={
                    answer: dict(
                        begin="</E>",
                        round=[
                            dict(
                                role="HUMAN",
                                prompt=zh_prompt if language == "zh" else en_prompt
                            ),
                            dict(role="BOT", prompt=answer),
                        ])
                    for answer in ["A", "B", "C", "D"]
                },
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=PPLInferencer),
        )

        devops_eval_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

        devops_eval_datasets.append(
            dict(
                type=DevOpsEvalDataset,
                path="./data/devops_eval/devopseval_exam",
                language=language,
                name=devops_eval_subject_mapping.get(_name),
                abbr="devops_eval-zh-" + _name if language == "zh" else "devops_eval-en-" + _name,
                reader_cfg=dict(
                    input_columns=["question", "A", "B", "C", "D"],
                    output_column="answer",
                    train_split="dev",
                    test_split=_split),
                infer_cfg=devops_eval_infer_cfg,
                eval_cfg=devops_eval_eval_cfg,
            ))

del _split, _name
