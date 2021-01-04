import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnf
import torch.autogradtorch.aut as autograd
import torch.from_numpy as from_numpy
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from numpy.random import multinomial
from tqdm import tqdm


class WordDataLoader:
    def __init__(
        self,
        datapath: str,
        min_count: int,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        load standard random paths from node2vec random walks
        """
        self.datapath = datapath
        self.min_count = min_count
        self.word_tuples = []
        self.word_freq = {}
        self.vocab_index = {}
        self.index_vocab = {}
        if random_seed is not None:
            random.seed(random_seed)

    def read(self):
        for line in open(self.datapath, encoding="utf8"):
            line = line.split()
            if not line:
                continue
            for word in line:
                if not word:
                    continue
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        wid = 0
        for word, count in self.word_freq.items():
            if count < self.min_count:
                continue
            self.vocab_index[word] = wid
            self.index_vocab[wid] = word
            wid += 1
        logging.info(f"Vocabulary size: {wid}")

    def negative_sampling(self, window_size: int = 2, num_negatives: int = 0):
        """
        """
        normalizing = sum([v ** 0.75 for v in self.word_freq.values()])
        sample_prob = {w: c ** 0.75 / normalizing for w, c in self.word_freq.items()}
        words = np.array(list(self.word_freq.keys()))

        negative_samples = []
        sampled_index = np.array(multinomial(num_negatives, list(sample_prob.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                negative_samples.append(words[index])

        for line in open(self.datapath, encoding="utf8"):
            line = line.split()
            if not line:
                continue
            for i, word in enumerate(line):
                if not word:
                    continue
                context_idx1 = max(0, i - window_size)
                context_idx2 = min(i + window_size, len(line))
                for j in range(context_idx1, context_idx2):
                    if i != j:
                        self.word_tuples.append((word, line[j], negative_samples[0]))
        logging.info(f"Num of target/context pairs: {len(self.word_tuples)}")


class Word2VecTorch(nn.Module):
    """
    A SkipGram based word2vec implementation using PyTorch.
    """
    def __init__(
        self,
        word_tuples: List[Tuple[str, str, Optional[Any]]],
        vocab_index: Dict[str, int],
        vector_size: int = 128,
        learning_rate: float = 0.025,
        iterations: int = 10,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        A driver class for the distributed Node2Vec algorithm for vertex embedding,
        read and write vectors and/or models.

        :param word_tuples: Spark dataframe of all random walk paths, [src, walk]
        :param vocab_index: dict of parameters to pass to gensim's word2vec module (not
                           to set embedding dim here)
        :param vector_size: int, dimension of the output graph embedding representation
                            num of codes after transforming from words (dimension
                            of embedding feature representation), usually power of 2,
                            e.g. 64, 128, 256
        :param random_seed: optional random seed, for testing only
        """
        logging.info("__init__(): preprocssing in spark ...")
        super(Word2VecTorch, self).__init__()
        self.word_tuples = word_tuples
        self.vocab_index = vocab_index
        self.vector_size = vector_size
        self.vocabulary_size = len(vocab_index)
        self.u_embeddings = nn.Embedding(self.vocabulary_size, vector_size, sparse=True)
        self.v_embeddings = nn.Embedding(self.vocabulary_size, vector_size, sparse=True)

        self.init_lr = learning_rate
        self.iterations = iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.cuda()
        if random_seed is not None:
            random.seed(random_seed)

    def forward(self, target_word, context_word, negative):
        """
        """
        # score for target words
        emb_target = self.u_embeddings(target_word)
        emb_context = self.v_embeddings(context_word)
        score_pos = torch.sum(torch.mul(emb_target, emb_context), dim=1)
        score = -torch.sum(nnf.logsigmoid(score_pos))
        # score for negative examples
        emb_negative = self.v_embeddings(negative)
        score_neg = torch.sum(torch.bmm(emb_negative, emb_target.unsqueeze(dim=2)), dim=1)
        score += -torch.sum(nnf.logsigmoid(-score_neg))
        return score

    def get_batches(self, batch_size: int = 100):
        """
        """
        random.shuffle(self.word_tuples)
        batches = []
        batch_target, batch_context, batch_negative = [], [], []
        for i in range(len(self.word_tuples)):
            batch_target.append(self.vocab_index[self.word_tuples[i][0]])
            batch_context.append(self.vocab_index[self.word_tuples[i][1]])
            batch_negative.append([self.vocab_index[w] for w in self.word_tuples[i][2]])
            if (i + 1) % batch_size == 0 or i == len(self.word_tuples) - 1:
                target = autograd.Variable(from_numpy(np.array(batch_target)).long())
                context = autograd.Variable(from_numpy(np.array(batch_context)).long())
                negative = autograd.Variable(from_numpy(np.array(batch_negative)).long())
                batches.append((target, context, negative))
                batch_target, batch_context, batch_negative = [], [], []
        return batches

    def train(self):
        """
        """
        for iteration in range(self.iterations):
            logging.info("\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.parameters(), lr=self.init_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(self.word_tuples),
            )

            running_loss = 0.0
            batches = self.get_batches()
            for i, sample_batched in enumerate(tqdm(batches)):
                if not sample_batched[0]:
                    continue
                target_word = sample_batched[0].to(self.device)
                context_word = sample_batched[1].to(self.device)
                negative_example = sample_batched[2].to(self.device)
                scheduler.step()
                optimizer.zero_grad()
                loss = self.forward(target_word, context_word, negative_example)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 0:
                    logging.info(f"Iteration {i} loss: {running_loss}")

    def get_vector(self, word2id, word: str) -> List[float]:
        """
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        wid = word2id[word]
        return embedding[wid]

    def save_vectors(self, id2word, output_path):
        """
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(output_path, 'w') as f:
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
