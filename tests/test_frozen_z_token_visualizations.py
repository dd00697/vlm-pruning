from __future__ import annotations

import unittest

from PIL import Image

from scripts.build_frozen_z_token_visualizations import mapped_box, processor_crop


class FrozenZTokenVisualizationTests(unittest.TestCase):
    def test_landscape_processor_crop_matches_shortest_edge_geometry(self) -> None:
        image = Image.new("RGB", (640, 480))
        cropped, transform = processor_crop(image)
        self.assertEqual(cropped.size, (336, 336))
        self.assertEqual(transform, (80.0, 0.0, 0.7))

    def test_box_mapping_clips_to_processor_crop(self) -> None:
        mapped = mapped_box([3.0, 160.0, 160.0, 472.0], (80.0, 0.0, 0.7))
        self.assertEqual(mapped, (0.0, 112.0, 56.0, 330.4))


if __name__ == "__main__":
    unittest.main()
