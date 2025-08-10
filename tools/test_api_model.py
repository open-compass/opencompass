import argparse
from typing import Dict

from mmengine.config import Config, ConfigDict

from opencompass.utils import Menu, build_model_from_cfg, model_abbr_from_cfg
from opencompass.utils.prompt import PromptList

test_prompts = [
    PromptList([
        {
            'section': 'begin',
            'pos': 'begin'
        },
        {
            'role':
            'SYSTEM',
            'fallback_role':
            'HUMAN',
            'prompt':
            'The following are multiple choice questions (with answers) about professional law.'  # noqa
        },
        '',
        {
            'section': 'ice',
            'pos': 'begin'
        },
        {
            'role':
            'HUMAN',
            'prompt':
            "Without a warrant, police officers searched the garbage cans in the alley behind a man's house and discovered chemicals used to make methamphetamine, as well as cooking utensils and containers with the man's fingerprints on them. The alley was a public thoroughfare maintained by the city, and the garbage was picked up once a week by a private sanitation company. The items were found inside the garbage cans in plastic bags that had been tied closed and further secured with tape. The man was charged in federal court with the manufacture of methamphetamine. Did the search of the garbage cans violate the Fourth Amendment?\nA. No, because the man had no reasonable expectation of privacy in garbage left in the alley.\nB. No, because the probative value of the evidence outweighs the man's modest privacy claims in his garbage.\nC. Yes, because the alley was within the curtilage of the man's home and entry without a warrant was unconstitutional.\nD. Yes, because there is a reasonable expectation of privacy in one's secured garbage containers.\nAnswer: "  # noqa
        },
        {
            'role': 'BOT',
            'prompt': 'A\n'
        },
        {
            'section': 'ice',
            'pos': 'end'
        },
        {
            'section': 'ice',
            'pos': 'begin'
        },
        {
            'role':
            'HUMAN',
            'prompt':
            'A man borrowed $500,000 from a bank, securing the loan with a mortgage on a commercial building he owned. The mortgage provided as follows: "No prepayment may be made on this loan during the first two years after the date of this mortgage. Thereafter, prepayment may be made in any amount at any time but only if accompanied by a prepayment fee of 5% of the amount prepaid." One year later, the man received an unexpected cash gift of $1 million and wished to pay off the $495,000 principal balance still owed on the loan. $495,000 principal balance still owed on the loan. Concerned that the bank might refuse prepayment, despite a rise in market interest rates in the year since the loan was made, or at least insist on the 5% prepayment fee, the man consulted an attorney concerning the enforceability of the above-quoted clause. There is no applicable statute. What is the attorney likely to say? \nA. The entire clause is unenforceable, because it violates a public policy favoring the prompt and early repayment of debt.\nB. The entire clause is unenforceable, because the rise in interest rates will allow the bank to reloan the funds without loss.\nC. The two-year prepayment prohibition and the prepayment fee provision are both valid and enforceable.\nD. The two-year prepayment prohibition is unenforceable, but the prepayment fee provision is enforceable.\nAnswer: '  # noqa
        },
        {
            'role': 'BOT',
            'prompt': 'D\n'
        },
        {
            'section': 'ice',
            'pos': 'end'
        },
        {
            'section': 'ice',
            'pos': 'begin'
        },
        {
            'role':
            'HUMAN',
            'prompt':
            "A woman and a defendant entered into an arrangement where the woman promised to pay the defendant $10,000 to act as a surrogate mother. In return, the defendant agreed to be implanted with the woman's embryo and carry the baby to term. The woman paid the defendant the $10,000 upfront. During the seventh month of the pregnancy, the defendant changed her mind and decided to keep the child herself. The defendant moved out of state and gave birth to the baby, which she refuses to turn over to the woman. The defendant is guilty of\nA. no crime.\nB. embezzlement.\nC. kidnapping.\nD. false pretenses.\nAnswer: "  # noqa
        },
        {
            'role': 'BOT',
            'prompt': 'A\n'
        },
        {
            'section': 'ice',
            'pos': 'end'
        },
        {
            'section': 'ice',
            'pos': 'begin'
        },
        {
            'role':
            'HUMAN',
            'prompt':
            "A rescuer was driving on an isolated portion of a country road. His headlights caught a figure lying at the side of the road. The rescuer stopped to investigate and found a victim, who was bleeding from head wounds and appeared to have been severely beaten. The rescuer then lifted the victim into his car and drove her to the hospital, a half-hour trip. When they arrived at the hospital, the rescuer carried the victim into the emergency room. He left her with a nurse and then returned home. Although the victim recovered from her injuries, she sued the hospital for malpractice, claiming that she was not promptly given medical attention. At trial, the nurse proposes to testify that when the victim was first brought to the hospital, she was unconscious. The victim's attorney objects and moves to strike the nurse's testimony. The trial judge should\nA. sustain the objection, because it goes to an ultimate issue in the case. \nB. sustain the objection, because the nurse is not qualified to render an expert opinion. \nC. overrule the objection, because it is a shorthand rendition of what she observed. \nD. overrule the objection, because there are independent grounds to show a present sense impression. \nAnswer: "  # noqa
        },
        {
            'role': 'BOT',
            'prompt': 'C\n'
        },
        {
            'section': 'ice',
            'pos': 'end'
        },
        {
            'section': 'ice',
            'pos': 'begin'
        },
        {
            'role':
            'HUMAN',
            'prompt':
            "A young woman who attended a rock concert at a nightclub was injured when the band opened its performance with illegal fireworks that ignited foam insulation in the club's ceiling and walls. The young woman sued the radio station that sponsored the performance. The radio station has moved for summary judgment, claiming that it owed no duty to audience members. The evidence has established the following facts: The station advertised its sponsorship on the radio and in print, distributed free tickets to the concert, and in print, distributed free tickets to the concert, staffed the event with the station's interns to assist with crowd control, and provided a station disc jockey to serve as master of ceremonies. The master of ceremonies had the authority to stop or delay the performance at any time on the basis of any safety concern. The station knew or should have known that the band routinely used unlicensed, illegal fireworks in its performances. Should the court grant the radio station's motion for summary judgment? \nA. No, because there is sufficient evidence of knowledge and control on the part of the station to impose on it a duty of care to audience members.\nB. No, because under respondeat superior, the radio station is vicariously liable for the negligent actions of the band.\nC. Yes, because it is the band and the nightclub owners who owed audience members a duty of care.\nD. Yes, because the conduct of the band in setting off illegal fireworks was criminal and setting off illegal fireworks was criminal and was a superseding cause as a matter of law.\nAnswer: "  # noqa
        },
        {
            'role': 'BOT',
            'prompt': 'A\n'
        },
        {
            'section': 'ice',
            'pos': 'end'
        },
        '\n',
        '',
        {
            'section': 'begin',
            'pos': 'end'
        },
        {
            'section': 'round',
            'pos': 'begin'
        },
        {
            'role':
            'HUMAN',
            'prompt':
            'A state statute provides: "Whenever a person knows or should know that he (or she) is being arrested by a police officer, it is the duty of such person to refrain from using force or any weapon in resisting arrest. " Violation of the statute is made punishable by fine and/or imprisonment. One morning, there was a bank robbery in the state. That afternoon, a police officer arrested a suspect who he believed was involved in the crime. However, the police officer and the suspect have given different accounts concerning what happened next. According to the police officer, after the suspect was apprehended, he resisted arrest and hit the police officer in the mouth with his fist. The police officer, who was momentarily stunned, pulled out his nightstick and struck the suspect over the head with it. On the other hand, the suspect claimed that after he was arrested, he cursed at the policeman, whereupon the police officer began hitting the suspect with his nightstick. To avoid being hit again, the suspect hit the police officer with his fist, knocking him down. The suspect was charged with assault. The suspect should be found\nA. not guilty, if the arrest was unlawful without probable cause and the jury believes the suspect\'s account.\nB. not guilty, if the arrest was lawful, provided that the jury believes the suspect\'s account.\nC. guilty, if the arrest was lawful, regardless which account the jury believes.\nD. guilty, if the arrest was unlawful, regardless which account the jury believes.\nAnswer: '  # noqa
        },
        {
            'section': 'round',
            'pos': 'end'
        }
    ]),
    'Hello! How are you?'
]

meta_templates = [
    None,
    dict(round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ], ),
    dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True)
        ],
        reserved_roles=[
            dict(role='SYSTEM', api_role='SYSTEM'),
        ],
    )
]


def test_model(model_cfg: ConfigDict):
    for meta_template in meta_templates:
        print('Testing meta_template: ', meta_template)
        model_cfg['meta_template'] = meta_template
        model = build_model_from_cfg(model_cfg)
        print('Prompt 0 length:',
              model.get_token_len_from_template(test_prompts[0]))
        print('Prompt 1 length:',
              model.get_token_len_from_template(test_prompts[1]))
        print('Prompt lengths: ',
              model.get_token_len_from_template(test_prompts))
        msgs = model.generate_from_template(test_prompts, max_out_len=100)
        print('Prompt 0 response:', msgs[0])
        print('Prompt 1 response:', msgs[1])
        print('-' * 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test if a given API model wrapper works properly')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('-n', '--non-interactive', action='store_true')
    args = parser.parse_args()
    return args


def parse_model_cfg(model_cfg: ConfigDict) -> Dict[str, ConfigDict]:
    model2cfg = {}
    for model in model_cfg:
        model2cfg[model_abbr_from_cfg(model)] = model
    return model2cfg


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if 'models' not in cfg:
        raise ValueError('No "models" specified in config file!')
    model2cfg = parse_model_cfg(cfg.models)

    if not args.non_interactive and len(model2cfg) > 1:
        model = Menu([list(model2cfg.keys())],
                     ['Please make a selection of models:']).run()
    else:
        model = list(model2cfg.keys())[0]
    model_cfg = model2cfg[model]
    test_model(model_cfg)


if __name__ == '__main__':
    main()
