"""Process-isolated observations from the real loader/evaluator construction path."""
import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from stok.cli.train import _build_dataloaders, _maybe_get_accelerator
from stok.eval import Evaluator
from stok.utils.losses import token_ce_loss


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, tokens, labels=None, ignore_index=-100, **kwargs):
        self.calls += 1
        logits = torch.zeros((*tokens.shape, 128), device=tokens.device) + self.anchor
        loss = token_ce_loss(logits, labels, ignore_index)
        return {"logits": logits, "classification_loss": loss, "loss": loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['coverage', 'eval-tail'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', choices=['map', 'iterable', 'map-mixture', 'mixed-mixture'], default='map')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--accum', type=int, default=1)
    args = parser.parse_args()
    accelerator = _maybe_get_accelerator()
    root = args.output
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2]/'src/stok/configs'), version_base=None):
        cfg = compose(config_name='config', overrides=['data.batch_size=2', 'data.max_len=5',
            f'data.num_workers={args.workers}', 'data.pin_memory=false', 'train.seed=7',
            'data.shuffle_shards=false', 'data.shuffle_rows=false'])
    train_path = str(root / ('shards' if args.source == 'iterable' else 'train.csv'))
    if args.source.endswith('mixture'):
        other = root / ('shards' if args.source == 'mixed-mixture' else 'train.csv')
        cfg.data.train = {'a': {'path': train_path, 'fraction': .4},
                          'b': {'path': str(other), 'fraction': .6}}
    else:
        cfg.data.train = train_path
    cfg.data.eval = str(root / ('eval_shards' if args.source == 'iterable' else 'eval.csv'))
    train, evaluations = _build_dataloaders(cfg, codebook_size=128, pad_id=1)
    loader = train if args.case == 'coverage' else evaluations['default']
    ids = []
    batches = 0
    for batch in loader:
        batches += 1
        ids.extend(batch[1][:, 1].tolist())
    metrics = {}
    if args.case == 'eval-tail':
        model = accelerator.prepare(FixedModel())
        model.eval()
        metrics = Evaluator(cfg, model, accelerator).evaluate(loader, 'default')
        assert not model.training
        assert accelerator.unwrap_model(model).calls.item() == batches
    rank = accelerator.process_index if accelerator else 0
    suffix = f'rank_{rank}' if accelerator.num_processes > 1 else 'reference'
    (root / f'{suffix}.json').write_text(json.dumps({'ids': ids,
        'micro_steps': batches, 'optimizer_steps': 0, 'metrics': metrics}))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
