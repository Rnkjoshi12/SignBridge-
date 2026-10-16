"""Unit tests for preprocessing utilities."""

import unittest
from unittest.mock import MagicMock, patch

import app.vision.preprocessing as prep


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing utilities."""

    def test_imagenet_constants(self):
        """Standard ImageNet mean and std should match specification."""
        self.assertEqual(prep.IMAGENET_MEAN, [0.485, 0.456, 0.406])
        self.assertEqual(prep.IMAGENET_STD, [0.229, 0.224, 0.225])

    def test_missing_torchvision_raises_importerror(self):
        """Functions should raise a helpful ImportError if torch/torchvision are absent."""
        with patch.object(prep, "torch", None), patch.object(prep, "transforms", None):
            with self.assertRaises(ImportError) as ctx:
                prep.get_train_transform()
            self.assertIn("torch and torchvision are required", str(ctx.exception))

            with self.assertRaises(ImportError):
                prep.get_val_transform()

            with self.assertRaises(ImportError):
                prep.get_inference_transform()

            with self.assertRaises(ImportError):
                prep.preprocess_frame(None)

    def test_bgr_to_pil_image_invalid_input(self):
        """BGRToPILImage should reject invalid input types when Image library is available."""
        mock_image = MagicMock()
        with patch.object(prep, "Image", mock_image):
            transformer = prep.BGRToPILImage()
            with self.assertRaises(TypeError):
                transformer(12345)

    def test_transforms_structure_with_torchvision(self):
        """When torchvision is present or mocked, transforms contain expected pipeline."""
        mock_transforms = MagicMock()
        mock_torch = MagicMock()

        # Configure mock transforms to record what classes are instantiated
        created_steps = []

        def make_mock_step(name):
            def factory(*args, **kwargs):
                step = MagicMock()
                step.__name__ = name
                step.name = name
                step.args = args
                step.kwargs = kwargs
                created_steps.append((name, kwargs))
                return step

            return factory

        mock_transforms.RandomResizedCrop = make_mock_step("RandomResizedCrop")
        mock_transforms.RandomRotation = make_mock_step("RandomRotation")
        mock_transforms.ColorJitter = make_mock_step("ColorJitter")
        mock_transforms.ToTensor = make_mock_step("ToTensor")
        mock_transforms.Normalize = make_mock_step("Normalize")
        mock_transforms.Resize = make_mock_step("Resize")
        mock_transforms.CenterCrop = make_mock_step("CenterCrop")

        def mock_compose(step_list):
            comp = MagicMock()
            comp.transforms = step_list
            return comp

        mock_transforms.Compose = mock_compose

        with patch.object(prep, "torch", mock_torch), patch.object(
            prep, "transforms", mock_transforms
        ):
            # Test Train Transform
            train_t = prep.get_train_transform(img_size=224)
            step_names = [getattr(s, "name", str(type(s))) for s in train_t.transforms]
            self.assertIn("RandomResizedCrop", step_names)
            self.assertIn("RandomRotation", step_names)
            self.assertIn("ColorJitter", step_names)
            self.assertIn("ToTensor", step_names)
            self.assertIn("Normalize", step_names)
            # Crucial requirement: RandomHorizontalFlip must NOT be present
            self.assertNotIn("RandomHorizontalFlip", step_names)

            # Test Val Transform
            val_t = prep.get_val_transform(img_size=224)
            val_step_names = [getattr(s, "name", str(type(s))) for s in val_t.transforms]
            self.assertIn("Resize", val_step_names)
            self.assertIn("CenterCrop", val_step_names)
            self.assertIn("ToTensor", val_step_names)
            self.assertIn("Normalize", val_step_names)

            # Test Inference Transform
            inf_t = prep.get_inference_transform(img_size=224)
            first_step = inf_t.transforms[0]
            self.assertIsInstance(first_step, (prep.BGRToPILImage, prep.ConvertBGRToPIL))

    def test_preprocess_frame_adds_batch_dimension(self):
        """preprocess_frame should add batch dimension (1, C, H, W)."""
        mock_torch = MagicMock()
        mock_transforms = MagicMock()

        # Mock a 3D tensor: shape (3, 224, 224)
        mock_tensor = MagicMock()
        mock_tensor.dim.return_value = 3
        mock_tensor_batched = MagicMock()
        mock_tensor_batched.dim.return_value = 4
        mock_tensor.unsqueeze.return_value = mock_tensor_batched

        mock_transform = MagicMock(return_value=mock_tensor)

        with patch.object(prep, "torch", mock_torch), patch.object(
            prep, "transforms", mock_transforms
        ):
            dummy_frame = MagicMock()
            out = prep.preprocess_frame(dummy_frame, transform=mock_transform)

            mock_transform.assert_called_once_with(dummy_frame)
            mock_tensor.unsqueeze.assert_called_once_with(0)
            self.assertEqual(out, mock_tensor_batched)


if __name__ == "__main__":
    unittest.main()
