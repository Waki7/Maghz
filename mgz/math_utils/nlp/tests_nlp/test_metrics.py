from __future__ import annotations

import unittest

from mgz.math_utils.nlp.metrics import *


class TestBert(unittest.TestCase):
    def setUp(self):
        pass

    def test_distance_measures_euclidean_sparseMatch(self):
        class_embeddings: FloatTensorT['NClasses,EmbedLen'] = torch.tensor(
            [[1, 1, 1],
             [1, 2, 3],
             [5, 5, 5]]).float()
        query_embeddings: FloatTensorT['NQuery,NClasses'] = torch.tensor(
            [[1, 2, 1],
             [5, 5, 5]]).float()
        distance_to_classes: FloatTensorT['NQuery,NClasses'] = (
            DistanceMeasuresPerClass.euclidean_distance(
                class_embeddings=class_embeddings,
                query_embeddings=query_embeddings))
        print(distance_to_classes)

        self.assertTrue(distance_to_classes.shape == (
            query_embeddings.shape[0], class_embeddings.shape[0]))
        self.assertTrue(distance_to_classes[0, 0] < distance_to_classes[0, 1])
        self.assertTrue(distance_to_classes[0, 0] < distance_to_classes[0, 2])
        self.assertTrue(distance_to_classes[1, 2] < distance_to_classes[1, 0])
        self.assertTrue(distance_to_classes[1, 2] < distance_to_classes[1, 1])
        probs = torch.softmax(distance_to_classes, dim=-1)
        print(probs)
        print(probs.argmax(dim=-1))

    def test_distance_measures_cosine_sparseMatch(self):
        cos = nn.CosineSimilarity(dim=-1, eps=1e-4)
        print(cos(torch.tensor([1, 1, 1]).float(),
                  torch.tensor([1, 2, 1]).float()))
        print(cos(torch.tensor([1, 2, 3]).float(),
                  torch.tensor([1, 2, 1]).float()))

        class_embeddings: FloatTensorT['NClasses,EmbedLen'] = torch.tensor(
            [[1, 1, 1],
             [1, 2, 3],
             [5, 5, 5]]).float()
        query_embeddings: FloatTensorT['NQuery,NClasses'] = torch.tensor(
            [[1, 2, 1],
             [5, 5, 5]]).float()
        similarity_to_classes = DistanceMeasuresPerClass.cosine_similarity(
            class_embeddings=class_embeddings,
            query_embeddings=query_embeddings)
        print(similarity_to_classes)
        self.assertTrue(similarity_to_classes.shape == (
            query_embeddings.shape[0], class_embeddings.shape[0]))
        self.assertTrue(
            similarity_to_classes[0, 0] > similarity_to_classes[0, 1])
        # cosine similarity is more for the angle between the vectors,
        # so different magnitude but same orientation should be the same (1,
        # 1,1) and (5,5,5)
        self.assertTrue(
            similarity_to_classes[0, 0] == similarity_to_classes[0, 2])
        self.assertTrue(
            similarity_to_classes[1, 2] == similarity_to_classes[1, 0])
        self.assertTrue(
            similarity_to_classes[1, 2] > similarity_to_classes[1, 1])
        probs = torch.softmax(similarity_to_classes, dim=-1)
        print(probs)
        print(probs.argmax(dim=-1))

    def test_distance_measures_dot_sparseMatch(self):
        class_embeddings: FloatTensorT['NClasses,EmbedLen'] = torch.tensor(
            [[1, 1, 1], [5, 5, 5]]).float()
        query_embeddings: FloatTensorT['NQuery,EmbedLen'] = torch.tensor(
            [[1, 0, 1], [5, 5, 5]]).float()
        similarity_to_classes = DistanceMeasuresPerClass.inner_dot_product(
            class_embeddings=class_embeddings,
            query_embeddings=query_embeddings)
        print(similarity_to_classes)
        self.assertTrue(similarity_to_classes.shape == (
            query_embeddings.shape[0], class_embeddings.shape[0]))
        self.assertTrue(
            similarity_to_classes[0, 0] > similarity_to_classes[0, 1])
        self.assertTrue(
            similarity_to_classes[0, 0] > similarity_to_classes[0, 2])
        self.assertTrue(
            similarity_to_classes[1, 2] > similarity_to_classes[1, 0])
        self.assertTrue(
            similarity_to_classes[1, 2] > similarity_to_classes[1, 1])
        probs = torch.softmax(similarity_to_classes, dim=-1)
        print(probs)
        print(probs.argmax(dim=-1))
