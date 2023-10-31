from __future__ import annotations

from mgz.ds.sentence_datasets.synthetic_memorization import \
    SyntheticMemorization
from archive.models.bert_basic import PositionalEncoding
from mgz.model_running.run_ops import *
from mgz.model_running.learning_ops import *
from settings import to_cuda
from mgz.typing import *
from mgz.model_running.basic_ops import greedy_decode


def show_example(fn, args=[]):
    if __name__ == "__main__":
        return fn(*args)


def example_positional():
    max_len, embed = 256, 20
    pe = PositionalEncoding(max_len=max_len, d_model=embed, dropout_p=.0)
    b, seq, embed = 1, 100, 20
    y = pe.forward(torch.zeros(b, seq, embed))
    print(y.shape)
    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .show()
    )


# Example of label smoothing.


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .show()
    )


def example_simple_model():
    n_cls = 15
    criterion = LabelSmoothing(n_cls=n_cls, padding_idx=0, smoothing=0.0)
    model = to_cuda(make_model(n_cls, n_cls, N=2))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed.d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    n_train_samples = 80
    n_eval_samples = 80
    iters = 1
    for epoch in range(iters):
        model.train()
        run_epoch(
            SyntheticMemorization(vocab_size=n_cls, out_vocab_size=n_cls,
                                  max_length=10,
                                  batch_size=batch_size,
                                  n_samples=n_train_samples).cuda().gen(),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )

        model.eval()
        run_epoch(
            SyntheticMemorization(vocab_size=n_cls, out_vocab_size=n_cls,
                                  max_length=10,
                                  batch_size=batch_size,
                                  n_samples=n_eval_samples).cuda().gen(),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )

    model.eval()
    src = to_cuda(torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    max_len = src.shape[1]
    src_mask = to_cuda(torch.ones(1, 1, max_len))
    print('greeeedy')
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# execute_example(example_simple_model)


def main():
    # show_example(example_positional)
    # show_example(example_label_smoothing)
    example_simple_model()


if __name__ == '__main__':
    main()
