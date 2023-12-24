from __future__ import annotations

import unittest

from mgz.metrics.nlp.metrics import *


class TestBert(unittest.TestCase):
    def setUp(self):
        pass

    def test_distance_measures_euclidean_sparseMatch(self):
        class_embeddings: FloatTensorT['NClasses,EmbedLen'] = torch.tensor(
            [[0, 0, 0], [5, 5, 5]]).float()
        query_embeddings: FloatTensorT['NQuery,EmbedLen'] = torch.tensor(
            [[0, 0, 0], [5, 5, 5]]).float()
        distance_to_classes = DistanceMeasuresPerClass.euclidean_distance(
            class_embeddings=class_embeddings,
            query_embeddings=query_embeddings)
        print(distance_to_classes)
        torch.testing.assert_allclose(distance_to_classes, torch.tensor(
            [[0.0, 8.6603], [8.6603, 0.0]]).float())
        print(ProbTensorT(torch.softmax(
            -1 * distance_to_classes, dim=-1)))

    def test_distance_measures_cosine_sparseMatch(self):
        cos = nn.CosineSimilarity(dim=-1, eps=1e-4)
        print(cos(torch.tensor([0, 0, 0]).float(),
                  torch.tensor([0, 0, 0]).float()))
        print(cos(torch.tensor([5, 5, 5]).float(),
                  torch.tensor([5, 5, 5]).float()))
        print(cos(torch.tensor([0, 0, 0]).float(),
                  torch.tensor([5, 5, 5]).float()))
        print(cos(torch.tensor([5, 5, 5]).float(),
                  torch.tensor([0, 0, 0]).float()))

        class_embeddings: FloatTensorT['NClasses,EmbedLen'] = torch.tensor(
            [[0, 0, 0],
             [5, 5, 5]]).float()
        query_embeddings: FloatTensorT['NQuery,EmbedLen'] = torch.tensor(
            [[0, 0, 0],
             [5, 5, 5]]).float()
        distance_to_classes = DistanceMeasuresPerClass.cosine_similarity(
            class_embeddings=class_embeddings,
            query_embeddings=query_embeddings)
        print(distance_to_classes)
        torch.testing.assert_allclose(distance_to_classes, torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]]).float())
        print(ProbTensorT(torch.softmax(
            -1 * distance_to_classes, dim=-1)))

    def test_distance_measures_euclidean_sparseMatch(self):
        class_embeddings: FloatTensorT['NClasses,EmbedLen'] = torch.tensor(
            [[0, 0, 0], [5, 5, 5]]).float()
        query_embeddings: FloatTensorT['NQuery,EmbedLen'] = torch.tensor(
            [[0, 0, 0], [5, 5, 5]]).float()
        distance_to_classes = DistanceMeasuresPerClass.inner_dot_product(
            class_embeddings=class_embeddings,
            query_embeddings=query_embeddings)
        print(distance_to_classes)
        torch.testing.assert_allclose(distance_to_classes, torch.tensor(
            [[0.0, 0.0], [0.0, 75.0]]).float())
        print(ProbTensorT(torch.softmax(
            -1 * distance_to_classes, dim=-1)))
