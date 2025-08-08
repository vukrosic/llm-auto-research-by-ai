#!/usr/bin/env python3
"""
Quick test to verify the FP4Linear fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FP4Linear(nn.Module):
    """Manual FP4 quantized linear layer"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights as int8 and scale factors
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features, dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def quantize_fp4(self, weight):
        """Quantize weights to 4-bit"""
        # Simple symmetric quantization to 4-bit range [-8, 7]
        scale = weight.abs().max(dim=1, keepdim=True)[0] / 7.0
        scale = scale.clamp(min=1e-8)
        quantized = torch.round(weight / scale).clamp(-8, 7)
        return quantized.to(torch.int8), scale.squeeze()
    
    def dequantize_fp4(self, quantized_weight, scale):
        """Dequantize 4-bit weights back to float"""
        return quantized_weight.float() * scale.unsqueeze(1)
    
    @property
    def weight(self):
        """Provide weight property for compatibility"""
        return self.dequantize_fp4(self.weight_quantized, self.weight_scale)
    
    def forward(self, x):
        # Dequantize weights for computation
        weight_fp32 = self.dequantize_fp4(self.weight_quantized, self.weight_scale)
        return F.linear(x, weight_fp32, self.bias)

# Test the fix
print("Testing FP4Linear fix...")

# Create a regular linear layer
regular_linear = nn.Linear(10, 5)

# Create FP4 layer
fp4_layer = FP4Linear(10, 5)

# Quantize the regular layer's weights
with torch.no_grad():
    quantized_weight, scale = fp4_layer.quantize_fp4(regular_linear.weight.data)
    fp4_layer.weight_quantized.copy_(quantized_weight)
    fp4_layer.weight_scale.copy_(scale)
    if regular_linear.bias is not None:
        fp4_layer.bias.copy_(regular_linear.bias.data)

# Test forward pass
x = torch.randn(3, 10)
try:
    output = fp4_layer(x)
    print(f"✅ FP4Linear forward pass successful! Output shape: {output.shape}")
    
    # Test weight property
    weight = fp4_layer.weight
    print(f"✅ Weight property works! Weight shape: {weight.shape}")
    
    # Test optimizer compatibility
    optimizer = torch.optim.AdamW(fp4_layer.parameters(), lr=1e-4)
    print("✅ Optimizer creation successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("Test complete!")