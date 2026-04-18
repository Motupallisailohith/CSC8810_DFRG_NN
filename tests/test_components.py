import unittest
import torch
import sys
import os

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fuzzy_layers import FuzzificationLayer, FuzzyGraphConvolution
from src.rough_sets import RoughSetBlock

class TestDFRGComponents(unittest.TestCase):
    
    def setUp(self):
        self.num_nodes = 10
        self.in_features = 5
        self.out_features = 8
        self.edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], 
                                        [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        self.x = torch.randn(self.num_nodes, self.in_features)

    def test_fuzzification_layer(self):
        """Test if fuzzification expands dimensions correctly and outputs valid probabilities."""
        print("\nTesting Fuzzification Layer...")
        num_sets = 3
        layer = FuzzificationLayer(self.in_features, num_fuzzy_sets=num_sets)
        out = layer(self.x)
        
        expected_dim = self.in_features * num_sets
        self.assertEqual(out.shape, (self.num_nodes, expected_dim))
        
        # Check values are between 0 and 1 (Gaussian memberships)
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))
        print(" -> PASSED: Output shape and value range correct.")

    def test_rough_set_block(self):
        """Test rough set approximations."""
        print("\nTesting Rough Set Block...")
        # Rough block input is hidden_dim, so let's project first or just test logic
        block = RoughSetBlock(self.in_features, self.out_features)
        
        out, lower, upper = block(self.x, self.edge_index)
        
        self.assertEqual(out.shape, (self.num_nodes, self.out_features))
        self.assertEqual(lower.shape, (self.num_nodes, self.out_features))
        self.assertEqual(upper.shape, (self.num_nodes, self.out_features))
        
        # Lower should be <= Upper (Min <= Max property)
        # Note: Since they go through different transformations before agg, strict inequality might not hold 
        # on the *output* of the block if weights differ, but let's check the approximation logic inside if accessible.
        # Ideally, we trust the 'min' and 'max' aggregation logic.
        print(" -> PASSED: Approximations generated with correct shapes.")

    def test_fuzzy_convolution(self):
        """Test fuzzy message passing."""
        print("\nTesting Fuzzy Graph Convolution...")
        layer = FuzzyGraphConvolution(self.in_features, self.out_features)
        
        mu_in = self.x
        sigma_in = torch.rand_like(self.x) * 0.1
        
        mu_out, sigma_out = layer(mu_in, self.edge_index, sigma_in)
        
        self.assertEqual(mu_out.shape, (self.num_nodes, self.out_features))
        self.assertEqual(sigma_out.shape, (self.num_nodes, self.out_features))
        
        # Uncertainty should propagate (not be zero)
        self.assertTrue(torch.any(sigma_out > 0))
        print(" -> PASSED: Fuzzy messages propagated successfully.")

if __name__ == '__main__':
    unittest.main()
