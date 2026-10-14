"""Check NLI on short QA fragments before applying it to natural RAG answers."""
import argparse
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure,require_ram
from ragscope.experiments.ledger import RecordStore
from ragscope.indexing.identity import artifact,atomic_json
from ragscope.v3.nli import WindowedNLI


def fixtures():
    examples=[]
    for i in range(6):
        examples.append({'question':f'How many samples did experiment {i+1} use?',
            'context':f'Experiment {i+1} used exactly {128+i} samples.',
            'positive':str(128+i),'negative':str(129+i),'negative_relation':'contradiction',
            'positive_statement':f'Experiment {i+1} used {128+i} samples.',
            'negative_statement':f'Experiment {i+1} used {129+i} samples.'})
        examples.append({'question':f'Did experiment {i+1} use dataset Delta?',
            'context':f'Experiment {i+1} did not use dataset Delta.',
            'positive':'No','negative':'Yes','negative_relation':'contradiction',
            'positive_statement':f'Experiment {i+1} did not use dataset Delta.',
            'negative_statement':f'Experiment {i+1} used dataset Delta.'})
    return examples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(args.campaign).resolve()
    if not run.is_relative_to(runtime/'runs') or not run.name.startswith('v3_'):raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        models=json.loads((run/'METRIC_MODEL_MANIFEST.json').read_text('utf-8'))
        model=next(m for m in models['models'] if m['model'].startswith('MoritzLaurer/'))
        config={'model':model,'runner':artifact(__file__),'implementation':artifact(PROJECT/'ragscope/v3/nli.py'),
            'truth_source':'RULE_CONSTRUCTION_NOT_HUMAN','maximum':512,'threshold':0.8,'dtype':'float32','device':'cpu'}
        store=RecordStore(run/'NLI_QA_STYLE_CALIBRATION.jsonl',config,max_records=72)
        tasks=[]
        for i,example in enumerate(fixtures()):
            for variant in ['plain_fragment','question_answer','complete_statement']:
                for direction in ['positive','negative']:
                    if variant=='question_answer':
                        premise=example['question']+'\nContext: '+example['context']
                        hypothesis=example['question']+'\nAnswer: '+example[direction]
                    elif variant=='complete_statement':
                        premise=example['context'];hypothesis=example[direction+'_statement']
                    else:
                        premise=example['context'];hypothesis=example[direction]
                    tasks.append({'record_id':f'{i}:{variant}:{direction}','variant':variant,
                        'premise':premise,'hypothesis':hypothesis,'expected':'entailment' if direction=='positive' else example['negative_relation']})
        pending=[t for t in tasks if t['record_id'] not in store.ids]
        started=time.monotonic()
        if pending:
            nli=WindowedNLI(model['directory'])
            try:
                for task in pending:
                    require_ram(2*1024**3)
                    result=nli.classify(task['premise'],task['hypothesis'])
                    pred=result['windows'][0]['argmax'] if len(result['windows'])==1 else 'UNKNOWN'
                    store.add(task['record_id'],{**task,'result':result,'prediction':pred,
                        'correct':pred==task['expected'],'human_review_status':'NOT_REVIEWED'})
            finally:nli.close()
        summaries={}
        for variant in ['plain_fragment','question_answer','complete_statement']:
            rows=[r for r in store.records if r['variant']==variant]
            summaries[variant]={'n':len(rows),'correct':sum(r['correct'] for r in rows),
                'accuracy':sum(r['correct'] for r in rows)/len(rows),
                'confident_verdict_n':sum(r['result']['verdict']!='UNKNOWN' for r in rows)}
        output={'status':'PASS_CALIBRATION_EXECUTED','styles':summaries,'records':artifact(store.path),
            'engineering_usage_rule':'QA wrapper is eligible only if accuracy>=0.90; otherwise short fragments must not be blindly treated as factual claims',
            'qa_wrapper_gate':summaries['question_answer']['accuracy']>=.90,
            'encoder_tokens':sum(r['result']['encoder_tokens'] for r in store.records),
            'elapsed_seconds':time.monotonic()-started,'human_review_status':'NOT_REVIEWED','new_llm_requests':0}
        atomic_json(run/'NLI_QA_STYLE_CALIBRATION.json',output);print(json.dumps(output))
        return 0


if __name__=='__main__':raise SystemExit(main())
