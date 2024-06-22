import torch
import torch.nn.functional as F


class GramModel:
    def __init__(self, method: str = "counts"):
        self.loss = None
        self.W = None
        self.charset = None
        self._method = method if method in {"counts", "nn"} else "counts"
        self.n = None
        self.stoi_in = None
        self.itos_in = None
        self.stoi_out = None
        self.itos_out = None

    def _set_alphabet(self, corpus: list):
        self.charset = sorted(list(set("".join(corpus))))
        self.itos_out = {k: v for k, v in enumerate(self.charset)}
        self.stoi_out = {v: k for k, v in self.itos_out.items()}

    def _set_mappings(self):
        input_chars = self.charset
        for i in range(self.n - 1):
            input_chars = [x + y for x in input_chars for y in self.charset]
        self.stoi_in = {k: v for v, k in enumerate(sorted(input_chars))}
        self.itos_in = {v: k for k, v in self.itos_out.items()}

    def _get_ngrams(self, corpus: list):
        ngrams = []
        for word in corpus:
            word = "." * self.n + word + "."
            ngrams.append(
                *[
                    (word[c : c + self.n - 1], word[c + self.n - 1])
                    for c in range(len(word) - self.n + 1)
                ]
            )
        return ngrams

    def _set_w_from_counts(self, ngrams: list, smooth_factor: float):
        self.W += smooth_factor
        for gram in ngrams:
            inp_i = self.stoi_in[gram[0]]
            outp_i = self.stoi_out[gram[1]]
            self.W[inp_i, outp_i] += 1
        probs = self.W / self.W.sum(1, keepdim=True)
        nll = None
        for gram in ngrams:
            inp_i = self.stoi_in[gram[0]]
            outp_i = self.stoi_out[gram[1]]
            nll += self.W[inp_i, outp_i].log()
        nll /= len(ngrams)
        nll *= -1
        self.loss = nll
        self.W = self.W.log()

    def _train_w_nn(self, ngrams: list, epochs: int, lr: float):
        x_data, y_data = map(list, zip(*ngrams))
        x_tensor = torch.tensor([self.stoi_in[x] for x in x_data])
        y_tensor = torch.tensor([self.stoi_out[y] for y in y_data])
        X = F.one_hot(x_tensor).float()
        y = F.one_hot(y_tensor).float()
        loss = None
        for i in range(epochs):
            probs = self._nn_forward(X)
            loss = -(probs[X, y]).log().mean()
            self.W.grad = None
            loss.backward()
            self.W.data += -lr * self.W.grad
            print(f"Epoch {i}: loss\t{loss}")
        self.loss = loss

    def _nn_forward(self, X: torch.Tensor):
        logits = X @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        return probs

    def fit(
        self,
        corpus: list,
        epochs: int = 10,
        lr: float = 10.0,
        smooth_factor=1.0,
        verbose=1,
    ):
        self._set_alphabet(corpus)
        self._set_mappings()
        ngrams = self._get_ngrams()
        self.W = torch.zeros(
            size=(len(self.itos_in), len(self.itos_out)),
            dtype=torch.float,
            requires_grad=True,
        )

        if self._method == "counts":
            self._set_w_from_counts(ngrams, smooth_factor)
        elif self._method == "nn":
            self._train_w_nn(ngrams, epochs, lr)

        if verbose:
            print(f"Finised with loss: {self.loss}")
