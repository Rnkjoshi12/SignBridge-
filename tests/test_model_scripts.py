"""Unit tests for SignBridge training and evaluation scripts.

Validates CLI argument parsing, path resolution, label loading,
confusion matrix computation, and data pipeline logic.
Gracefully handles missing optional machine learning packages.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import torch
    import torchvision
    import sklearn
    import matplotlib
    ML_DEPS_AVAILABLE = True
except ImportError:
    ML_DEPS_AVAILABLE = False


class TestModelScripts(unittest.TestCase):
    """Test suite for train_model and evaluate_model utility functions and CLI."""

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_train_model_parser_defaults(self):
        """Verifies default CLI arguments for train_model.py."""
        from scripts.train_model import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.epochs, 15)
        self.assertEqual(args.batch_size, 32)
        self.assertAlmostEqual(args.lr, 0.001)
        self.assertAlmostEqual(args.weight_decay, 1e-4)
        self.assertTrue(any(sub in args.dataset_path for sub in ("cleaned_images", "images for phrases")))
        self.assertIn("sign_model.pth", args.output_model)

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_train_model_parser_custom_args(self):
        """Verifies custom CLI overrides for train_model.py."""
        from scripts.train_model import build_parser

        parser = build_parser()
        custom_args = [
            "--dataset-path", "/custom/data",
            "--epochs", "25",
            "--batch-size", "16",
            "--lr", "0.0005",
            "--output-model", "/custom/model.pth",
            "--seed", "123",
        ]
        args = parser.parse_args(custom_args)
        self.assertEqual(args.dataset_path, "/custom/data")
        self.assertEqual(args.epochs, 25)
        self.assertEqual(args.batch_size, 16)
        self.assertAlmostEqual(args.lr, 0.0005)
        self.assertEqual(args.output_model, "/custom/model.pth")
        self.assertEqual(args.seed, 123)

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_evaluate_model_parser_defaults(self):
        """Verifies default CLI arguments for evaluate_model.py."""
        from scripts.evaluate_model import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.top_k, 10)
        self.assertIn("sign_model.pth", args.model_path)
        self.assertTrue(any(sub in args.dataset_path for sub in ("cleaned_images", "images for phrases")))

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_load_class_labels_dict_format(self):
        """Tests loading class_labels.json with dict mapping format."""
        from scripts.evaluate_model import load_class_labels

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            data = {"0": "hello", "1": "thank you", "2": "goodbye"}
            json.dump(data, f)
            temp_path = f.name

        try:
            idx_to_class, class_names = load_class_labels(temp_path)
            self.assertEqual(len(class_names), 3)
            self.assertEqual(idx_to_class[0], "hello")
            self.assertEqual(idx_to_class[1], "thank you")
            self.assertEqual(idx_to_class[2], "goodbye")
            self.assertEqual(class_names, ["hello", "thank you", "goodbye"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_load_class_labels_list_format(self):
        """Tests loading class_labels.json with list format."""
        from scripts.evaluate_model import load_class_labels

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            data = ["again", "boy", "cat"]
            json.dump(data, f)
            temp_path = f.name

        try:
            idx_to_class, class_names = load_class_labels(temp_path)
            self.assertEqual(len(class_names), 3)
            self.assertEqual(idx_to_class[0], "again")
            self.assertEqual(idx_to_class[1], "boy")
            self.assertEqual(idx_to_class[2], "cat")
            self.assertEqual(class_names, ["again", "boy", "cat"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_top_confused_pairs(self):
        """Tests extraction and ranking of top confused class pairs."""
        from scripts.evaluate_model import get_top_confused_pairs

        class_names = ["again", "boy", "cat", "dog"]
        cm = np.array([
            [10,  2,  1,  0],
            [ 0,  8,  0,  4],
            [ 0,  0, 12,  0],
            [ 1,  0,  0,  9],
        ])

        top_pairs = get_top_confused_pairs(cm, class_names, top_k=3)
        self.assertEqual(len(top_pairs), 3)
        self.assertEqual(top_pairs[0], ("boy", "dog", 4))
        self.assertEqual(top_pairs[1], ("again", "boy", 2))
        self.assertEqual(top_pairs[2][2], 1)

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_top_confused_pairs_perfect_accuracy(self):
        """Tests top confused pairs when confusion matrix is purely diagonal."""
        from scripts.evaluate_model import get_top_confused_pairs

        class_names = ["a", "b", "c"]
        cm = np.eye(3, dtype=int) * 10
        top_pairs = get_top_confused_pairs(cm, class_names, top_k=5)
        self.assertEqual(top_pairs, [])

    @unittest.skipUnless(ML_DEPS_AVAILABLE, "Requires numpy, torch, torchvision, sklearn, matplotlib")
    def test_path_resolvers(self):
        """Verifies resolve_default_dataset_path and resolve_default_model_path fallbacks."""
        from scripts.train_model import (
            DEFAULT_DATASET_PATH,
            DEFAULT_OUTPUT_MODEL,
            resolve_default_dataset_path,
            resolve_default_model_path,
        )

        dataset_path = resolve_default_dataset_path()
        model_path = resolve_default_model_path()

        self.assertTrue(len(dataset_path) > 0)
        self.assertTrue(any(sub in dataset_path for sub in ("cleaned_images", "images for phrases")))
        self.assertTrue(model_path.endswith("sign_model.pth"))

    def test_script_syntax_valid(self):
        """Verifies that train_model.py and evaluate_model.py compile without syntax errors."""
        import py_compile
        train_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "train_model.py",
        )
        eval_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "evaluate_model.py",
        )
        py_compile.compile(train_path, doraise=True)
        py_compile.compile(eval_path, doraise=True)


if __name__ == "__main__":
    unittest.main()
