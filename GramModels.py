import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(0)


class GramModel:
    """
    A class representing a Gram model for language modeling.

    Attributes:
        loss (float): The loss of the model.
        W (torch.Tensor): The weights of the model.
        charset (list): The character set of the model.
        _method (str): The method used to train the model.
        n (int): The size of the n-grams used in the model.
        stoi_in (dict): A dictionary mapping input characters to their indices.
        itos_in (dict): A dictionary mapping input indices to their characters.
        stoi_out (dict): A dictionary mapping output characters to their indices.
        itos_out (dict): A dictionary mapping output indices to their characters.
    """

    def __init__(self, method: str = "counts", n_grams=3):
        """
        Initialize a GramModel object.

        Args:
            method (str, optional): The method used to train the model. Defaults to "counts".
            n_grams (int, optional): The size of the n-grams used in the model. Defaults to 3.
        """

        self.loss = None
        self.W = None
        self.charset = None
        self._method = method if method in {"counts", "nn"} else "counts"
        self.n = n_grams
        self.stoi_in = None
        self.itos_in = None
        self.stoi_out = None
        self.itos_out = None

    def _set_alphabet(self, corpus: list):
        """
        Set the character set of the model based on the given corpus.

        Args:
            corpus (list): The corpus used to set the character set.
        """
        print("Generating alphabet")
        self.charset = ["."] + sorted(list(set("".join(corpus))))
        self.itos_out = {k: v for k, v in enumerate(self.charset)}
        self.stoi_out = {v: k for k, v in self.itos_out.items()}
        print("alphabet Generated", flush=True)

    def _set_mappings(self):
        """
        Set the mappings between input and output characters.
        """
        print("Generating mappings")
        input_chars = self.charset
        for i in range(self.n - 2):
            input_chars = [x + y for x in input_chars for y in self.charset]
        self.stoi_in = {k: v for v, k in enumerate(sorted(input_chars))}
        self.itos_in = {v: k for k, v in self.stoi_in.items()}
        print("Mappings generated", flush=True)

    def _get_ngrams(self, corpus: list):
        """
        Get the n-grams from the given corpus.

        Args:
            corpus (list): The corpus used to get the n-grams.

        Returns:
            list: The list of n-grams.
        """
        print("Generatng n-grams")
        ngrams = []
        for word in corpus:
            word = "." * self.n + word + "."
            ngrams.extend(
                [
                    (word[c : c + self.n - 1], word[c + self.n - 1])
                    for c in range(len(word) - self.n + 1)
                ]
            )
        print(f"NGrams of length {len(ngrams)} Generated", flush=True)
        return ngrams

    def _set_w_from_counts(self, ngrams: list, smooth_factor: float):
        """
        Set the weights of the model from the counts based on the given n-grams and smooth factor.

        Args:
            ngrams (list): The list of n-grams.
            smooth_factor (float): The smooth factor used to smooth the counts.
        """

        self.W.data += smooth_factor
        for gram in tqdm(ngrams, desc="Counting grams.."):
            inp_i = self.stoi_in[gram[0]]
            outp_i = self.stoi_out[gram[1]]
            self.W[inp_i, outp_i].data += 1
        probs = self.W / self.W.sum(1, keepdim=True)
        nll = torch.tensor(0.0)
        for gram in tqdm(ngrams, desc="calculating loss.."):
            inp_i = self.stoi_in[gram[0]]
            outp_i = self.stoi_out[gram[1]]
            nll += probs[inp_i, outp_i].log()
        nll /= len(ngrams)
        nll *= -1
        self.loss = nll
        self.W = self.W.log()

    def _train_w_nn(self, ngrams: list, epochs: int, lr: float, verbose=1):
        """
        Train the model using a neural network approach.

        Args:
            ngrams (list): List of input-output pairs.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for training.
            verbose (int, optional): The verbosity level. Defaults to 1.
        """

        x_data, y_data = map(list, zip(*ngrams))
        x_tensor = torch.tensor([self.stoi_in[x] for x in x_data])
        y_tensor = torch.tensor([self.stoi_out[y] for y in y_data])
        X = F.one_hot(x_tensor).float()
        # y = F.one_hot(y_tensor).float()
        loss = None
        for i in range(epochs):
            probs = self._nn_forward(X)
            loss = -(probs[x_tensor, y_tensor]).log().mean()
            self.W.grad = None
            loss.backward()
            self.W.data += -lr * self.W.grad
            if verbose:
                print(f"Epoch {i+1}: loss\t{loss}")
        self.loss = loss

    def _nn_forward(self, X: torch.Tensor):
        """
        Perform the forward pass of the neural network.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output probabilities after the forward pass.
        """

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
        """
        Train the model on the given corpus using the method specified in self._method.

        Args:
            corpus (list): The corpus used for training.
            epochs (int, optional): The number of training epochs only used with self._method = 'nn'. Defaults to 100.
            lr (float, optional): The learning rate for training only used with self._method = 'nn'. Defaults to 0.1.
            smooth_factor (float, optional): The smoothing factor for the counts only used with self._method = 'counts'. Defaults to 1.0.
            verbose (int, optional): The verbosity level. Defaults to 1.
        """

        self._set_alphabet(corpus)
        self._set_mappings()
        ngrams = self._get_ngrams(corpus)
        if self._method == "counts":
            self.W = torch.zeros(
                size=(len(self.itos_in), len(self.itos_out)),
                dtype=torch.float,
                requires_grad=True,
            )
            self._set_w_from_counts(ngrams, smooth_factor)
        elif self._method == "nn":
            self.W = torch.randn(
                size=(len(self.itos_in), len(self.itos_out)),
                dtype=torch.float,
                requires_grad=True,
            )
            self._train_w_nn(ngrams, epochs, lr, verbose)
        print(f"Here's the loss: {self.loss}")
        if verbose:
            print(f"Finised with loss: {self.loss}")
