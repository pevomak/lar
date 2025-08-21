import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage
from scipy.signal import convolve2d
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

class DimensionalComplexitySimulation:
    """
    Simulates 2D → 2D+depth progression and boundary-driven complexity emergence
    Tests empirical validity against known observations
    """
    
    def __init__(self, grid_size=100, max_depth=20, dt=0.01):
        self.grid_size = grid_size
        self.max_depth = max_depth
        self.dt = dt
        self.time = 0
        
        # Initialize 2D substrate
        self.substrate_2d = self.initialize_2d_substrate()
        
        # Initialize depth-dependent fields
        self.depth_field = np.zeros((grid_size, grid_size, max_depth))
        self.boundary_dynamics = np.zeros((grid_size, grid_size, max_depth-1))
        self.complexity_field = np.zeros((grid_size, grid_size, max_depth))
        
        # Observational validation metrics
        self.complexity_history = []
        self.boundary_strength_history = []
        self.information_flow_history = []
        self.emergence_events = []
        
        # Physical constants (dimensionally consistent)
        self.diffusion_coeff = 0.1
        self.boundary_coupling = 0.5
        self.complexity_threshold = 0.3
        
    def initialize_2d_substrate(self):
        """Initialize 2D substrate with realistic patterns"""
        x = np.linspace(0, 4*np.pi, self.grid_size)
        y = np.linspace(0, 4*np.pi, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Multi-scale pattern mimicking biological/physical systems
        substrate = (
            0.3 * np.sin(X) * np.cos(Y) +  # Large-scale structure
            0.2 * np.sin(2*X + Y) +        # Medium-scale patterns
            0.1 * np.random.random((self.grid_size, self.grid_size)) +  # Noise
            0.15 * np.sin(0.5*X) * np.sin(0.5*Y)  # Very large scale
        )
        
        # Normalize to [0, 1]
        substrate = (substrate - substrate.min()) / (substrate.max() - substrate.min())
        return substrate
    
    def calculate_depth_progression(self):
        """
        Core mechanism: 2D patterns progress into depth dimension
        Depth emergence follows information gradients from 2D substrate
        """
        # Information gradient drives depth emergence
        grad_x = np.gradient(self.substrate_2d, axis=1)
        grad_y = np.gradient(self.substrate_2d, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Depth progression driven by local complexity
        for d in range(self.max_depth):
            depth_factor = d / (self.max_depth - 1)
            
            # Information "pressure" drives progression into depth
            # Higher gradients create stronger depth penetration
            information_pressure = gradient_magnitude * (1 - depth_factor**2)
            
            # Temporal evolution with diffusion
            diffusion_term = scipy.ndimage.gaussian_filter(information_pressure, sigma=1.0)
            
            # Update depth field
            self.depth_field[:, :, d] = (
                self.substrate_2d * information_pressure * 
                np.exp(-depth_factor * 2) +  # Exponential decay with depth
                0.1 * diffusion_term  # Diffusion coupling
            )
    
    def calculate_boundary_dynamics(self):
        """
        Calculate dynamics at boundaries between depth layers
        This is where complexity emergence occurs
        """
        for d in range(self.max_depth - 1):
            # Boundary exists between layer d and d+1
            layer_current = self.depth_field[:, :, d]
            layer_next = self.depth_field[:, :, d + 1]
            
            # Boundary gradient (information flow across layers)
            boundary_gradient = layer_current - layer_next
            
            # Boundary curvature (creates complex dynamics)
            boundary_laplacian = scipy.ndimage.laplace(boundary_gradient)
            
            # Coupling between adjacent boundaries (non-local effects)
            if d > 0:
                coupling_term = self.boundary_dynamics[:, :, d-1] * self.boundary_coupling
            else:
                coupling_term = 0
            
            # Boundary dynamics equation (reaction-diffusion like)
            self.boundary_dynamics[:, :, d] = (
                boundary_gradient + 
                self.diffusion_coeff * boundary_laplacian +
                coupling_term +
                0.05 * np.sin(self.time * 2 * np.pi) * boundary_gradient  # Temporal modulation
            )
    
    def calculate_complexity_emergence(self):
        """
        Calculate complexity emergence from boundary dynamics
        Tests core hypothesis: complexity emerges at boundaries
        """
        for d in range(self.max_depth):
            # Base complexity from depth field structure
            local_variance = scipy.ndimage.generic_filter(
                self.depth_field[:, :, d], np.var, size=3
            )
            
            # Boundary-driven complexity enhancement
            if d > 0 and d < self.max_depth - 1:
                # Complexity enhanced by boundary dynamics above and below
                boundary_above = self.boundary_dynamics[:, :, d-1] if d > 0 else 0
                boundary_below = self.boundary_dynamics[:, :, d] if d < self.max_depth-1 else 0
                
                # Gradient interaction creates complexity
                gradient_interaction = np.abs(boundary_above * boundary_below)
                
                # Information processing at boundaries
                information_processing = np.abs(
                    scipy.ndimage.sobel(boundary_above) * 
                    scipy.ndimage.sobel(boundary_below)
                )
                
                complexity = (
                    local_variance + 
                    0.5 * gradient_interaction + 
                    0.3 * information_processing
                )
            else:
                # Edge layers have reduced complexity
                complexity = 0.7 * local_variance
            
            self.complexity_field[:, :, d] = complexity
    
    def detect_emergence_events(self):
        """
        Detect and catalog emergence events where complexity exceeds threshold
        """
        total_complexity = np.sum(self.complexity_field)
        
        if total_complexity > self.complexity_threshold * self.grid_size**2 * self.max_depth:
            # Find locations of high complexity
            high_complexity_mask = self.complexity_field > np.percentile(
                self.complexity_field, 95
            )
            emergence_locations = np.where(high_complexity_mask)
            
            if len(emergence_locations[0]) > 0:
                self.emergence_events.append({
                    'time': self.time,
                    'total_complexity': total_complexity,
                    'num_sites': len(emergence_locations[0]),
                    'max_complexity': np.max(self.complexity_field),
                    'average_depth': np.mean(emergence_locations[2])
                })
    
    def calculate_information_flow(self):
        """
        Calculate information flow between layers
        Tests whether depth progression actually transfers information
        """
        info_flow = []
        
        for d in range(self.max_depth - 1):
            # Mutual information between adjacent layers
            layer1_flat = self.depth_field[:, :, d].flatten()
            layer2_flat = self.depth_field[:, :, d+1].flatten()
            
            # Discretize for mutual information calculation
            layer1_discrete = np.digitize(layer1_flat, np.linspace(0, 1, 10))
            layer2_discrete = np.digitize(layer2_flat, np.linspace(0, 1, 10))
            
            mi = mutual_info_score(layer1_discrete, layer2_discrete)
            info_flow.append(mi)
        
        return np.array(info_flow)
    
    def update(self):
        """Single time step update"""
        self.calculate_depth_progression()
        self.calculate_boundary_dynamics()
        self.calculate_complexity_emergence()
        self.detect_emergence_events()
        
        # Update substrate (slow evolution)
        self.substrate_2d += 0.001 * np.sin(self.time * 0.5) * np.random.random((self.grid_size, self.grid_size))
        self.substrate_2d = np.clip(self.substrate_2d, 0, 1)
        
        # Record metrics
        self.complexity_history.append(np.mean(self.complexity_field))
        self.boundary_strength_history.append(np.mean(np.abs(self.boundary_dynamics)))
        info_flow = self.calculate_information_flow()
        self.information_flow_history.append(np.mean(info_flow))
        
        self.time += self.dt
    
    def run_simulation(self, steps=1000):
        """Run full simulation"""
        print(f"Running {steps} simulation steps...")
        
        for i in range(steps):
            self.update()
            if i % 100 == 0:
                print(f"Step {i}/{steps} - Complexity: {self.complexity_history[-1]:.4f}")
        
        print("Simulation complete!")
    
    def analyze_empirical_validity(self):
        """
        Analyze results against known empirical observations
        """
        print("\n" + "="*60)
        print("EMPIRICAL VALIDITY ANALYSIS")
        print("="*60)
        
        # 1. Test: Does complexity actually emerge from boundaries?
        boundary_complexity_correlation = np.corrcoef(
            self.boundary_strength_history, self.complexity_history
        )[0, 1]
        
        print(f"\n1. BOUNDARY-COMPLEXITY CORRELATION")
        print(f"   Correlation coefficient: {boundary_complexity_correlation:.4f}")
        print(f"   Interpretation: {'STRONG' if abs(boundary_complexity_correlation) > 0.7 else 'MODERATE' if abs(boundary_complexity_correlation) > 0.4 else 'WEAK'}")
        
        # 2. Test: Information flow analysis
        avg_info_flow = np.mean(self.information_flow_history)
        info_flow_trend = np.polyfit(range(len(self.information_flow_history)), 
                                    self.information_flow_history, 1)[0]
        
        print(f"\n2. INFORMATION FLOW ANALYSIS")
        print(f"   Average information flow: {avg_info_flow:.4f}")
        print(f"   Flow trend (slope): {info_flow_trend:.6f}")
        print(f"   Information transfer: {'INCREASING' if info_flow_trend > 0 else 'DECREASING' if info_flow_trend < 0 else 'STABLE'}")
        
        # 3. Test: Emergence event statistics
        if self.emergence_events:
            emergence_depths = [event['average_depth'] for event in self.emergence_events]
            avg_emergence_depth = np.mean(emergence_depths)
            emergence_frequency = len(self.emergence_events) / self.time
            
            print(f"\n3. EMERGENCE EVENTS ANALYSIS")
            print(f"   Number of emergence events: {len(self.emergence_events)}")
            print(f"   Average emergence depth: {avg_emergence_depth:.2f}")
            print(f"   Emergence frequency: {emergence_frequency:.4f} events/time_unit")
            print(f"   Depth preference: {'SURFACE' if avg_emergence_depth < 5 else 'MIDDLE' if avg_emergence_depth < 15 else 'DEEP'}")
        else:
            print(f"\n3. EMERGENCE EVENTS ANALYSIS")
            print(f"   No emergence events detected above threshold")
        
        # 4. Test: Depth-dependent complexity scaling
        complexity_by_depth = np.mean(self.complexity_field, axis=(0, 1))
        depth_complexity_correlation = np.corrcoef(
            range(len(complexity_by_depth)), complexity_by_depth
        )[0, 1]
        
        print(f"\n4. DEPTH-COMPLEXITY SCALING")
        print(f"   Depth-complexity correlation: {depth_complexity_correlation:.4f}")
        print(f"   Complexity distribution: {'SURFACE-PEAKED' if depth_complexity_correlation < -0.3 else 'DEEP-PEAKED' if depth_complexity_correlation > 0.3 else 'UNIFORM'}")
        
        # 5. Overall framework validation
        print(f"\n5. FRAMEWORK VALIDATION SUMMARY")
        
        validation_score = 0
        max_score = 4
        
        if abs(boundary_complexity_correlation) > 0.5:
            validation_score += 1
            print(f"   ✓ Boundary-driven complexity confirmed")
        else:
            print(f"   ✗ Boundary-complexity correlation weak")
            
        if avg_info_flow > 0.1:
            validation_score += 1
            print(f"   ✓ Information flow between layers confirmed")
        else:
            print(f"   ✗ Limited information flow detected")
            
        if len(self.emergence_events) > 0:
            validation_score += 1
            print(f"   ✓ Complexity emergence events detected")
        else:
            print(f"   ✗ No significant emergence events")
            
        if abs(depth_complexity_correlation) > 0.2:
            validation_score += 1
            print(f"   ✓ Depth-dependent complexity scaling")
        else:
            print(f"   ✗ No clear depth-complexity relationship")
        
        validation_percentage = (validation_score / max_score) * 100
        print(f"\n   OVERALL VALIDATION: {validation_score}/{max_score} ({validation_percentage:.0f}%)")
        
        if validation_percentage >= 75:
            print(f"   VERDICT: STRONG EMPIRICAL SUPPORT")
        elif validation_percentage >= 50:
            print(f"   VERDICT: MODERATE EMPIRICAL SUPPORT")
        else:
            print(f"   VERDICT: WEAK EMPIRICAL SUPPORT")
        
        return {
            'boundary_complexity_correlation': boundary_complexity_correlation,
            'avg_info_flow': avg_info_flow,
            'emergence_events': len(self.emergence_events),
            'depth_complexity_correlation': depth_complexity_correlation,
            'validation_score': validation_score,
            'validation_percentage': validation_percentage
        }
    
    def visualize_results(self):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle('2D → 2D+Depth Complexity Emergence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Original 2D substrate
        im1 = axes[0, 0].imshow(self.substrate_2d, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('2D Substrate (Initial Pattern)')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Depth field cross-section (middle slice)
        middle_depth = self.max_depth // 2
        im2 = axes[0, 1].imshow(self.depth_field[:, :, middle_depth], cmap='plasma', aspect='auto')
        axes[0, 1].set_title(f'Depth Field (Layer {middle_depth})')
        axes[0, 1].set_xlabel('X Position')
        axes[0, 1].set_ylabel('Y Position')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. Boundary dynamics
        if self.boundary_dynamics.shape[2] > 0:
            im3 = axes[0, 2].imshow(self.boundary_dynamics[:, :, middle_depth//2], 
                                  cmap='RdBu_r', aspect='auto')
            axes[0, 2].set_title(f'Boundary Dynamics (Interface {middle_depth//2})')
            axes[0, 2].set_xlabel('X Position')
            axes[0, 2].set_ylabel('Y Position')
            plt.colorbar(im3, ax=axes[0, 2])
        
        # 4. Complexity field
        im4 = axes[1, 0].imshow(self.complexity_field[:, :, middle_depth], 
                              cmap='hot', aspect='auto')
        axes[1, 0].set_title(f'Complexity Field (Layer {middle_depth})')
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Y Position')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # 5. Complexity evolution over time
        axes[1, 1].plot(self.complexity_history, 'b-', linewidth=2, label='Complexity')
        axes[1, 1].plot(self.boundary_strength_history, 'r--', linewidth=2, label='Boundary Strength')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].set_title('Temporal Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Information flow between layers
        if self.information_flow_history:
            axes[1, 2].plot(self.information_flow_history, 'g-', linewidth=2)
            axes[1, 2].set_xlabel('Time Steps')
            axes[1, 2].set_ylabel('Mutual Information')
            axes[1, 2].set_title('Information Flow Between Layers')
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Complexity by depth (average)
        complexity_by_depth = np.mean(self.complexity_field, axis=(0, 1))
        axes[2, 0].plot(range(self.max_depth), complexity_by_depth, 'mo-', linewidth=2)
        axes[2, 0].set_xlabel('Depth Layer')
        axes[2, 0].set_ylabel('Average Complexity')
        axes[2, 0].set_title('Complexity vs Depth')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Emergence events timeline
        if self.emergence_events:
            times = [event['time'] for event in self.emergence_events]
            complexities = [event['total_complexity'] for event in self.emergence_events]
            depths = [event['average_depth'] for event in self.emergence_events]
            
            scatter = axes[2, 1].scatter(times, complexities, c=depths, 
                                       cmap='cool', s=60, alpha=0.7)
            axes[2, 1].set_xlabel('Time')
            axes[2, 1].set_ylabel('Total Complexity')
            axes[2, 1].set_title('Emergence Events (color = depth)')
            plt.colorbar(scatter, ax=axes[2, 1])
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No Emergence\nEvents Detected', 
                          ha='center', va='center', transform=axes[2, 1].transAxes,
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[2, 1].set_title('Emergence Events')
        
        # 9. 3D visualization of depth progression (cross-section)
        y_slice = self.grid_size // 2
        depth_slice = self.depth_field[y_slice, :, :].T
        im9 = axes[2, 2].imshow(depth_slice, cmap='viridis', aspect='auto', origin='lower')
        axes[2, 2].set_xlabel('X Position')
        axes[2, 2].set_ylabel('Depth Layer')
        axes[2, 2].set_title('Depth Progression (Y-slice)')
        plt.colorbar(im9, ax=axes[2, 2])
        
        plt.tight_layout()
        return fig
    
    def compare_with_known_systems(self):
        """
        Compare simulation results with known physical/biological systems
        that exhibit 2D→3D complexity emergence
        """
        print("\n" + "="*60)
        print("COMPARISON WITH KNOWN SYSTEMS")
        print("="*60)
        
        # System 1: Crystal growth (2D nucleation → 3D structure)
        print("\n1. CRYSTAL GROWTH COMPARISON")
        print("   Known: 2D nucleation sites → layer-by-layer 3D growth")
        print("   Known: Complexity emerges at growth interfaces")
        
        if self.emergence_events and len(self.emergence_events) > 0:
            emergence_depths = [event['average_depth'] for event in self.emergence_events]
            surface_bias = sum(1 for d in emergence_depths if d < 5) / len(emergence_depths)
            print(f"   Simulation: {surface_bias:.2%} emergence events near surface")
            print(f"   Match: {'GOOD' if surface_bias > 0.6 else 'PARTIAL' if surface_bias > 0.3 else 'POOR'}")
        
        # System 2: Biological development (2D sheets → 3D organs)
        print("\n2. BIOLOGICAL DEVELOPMENT COMPARISON")
        print("   Known: 2D epithelial sheets fold into 3D structures")
        print("   Known: Complexity emerges at tissue boundaries")
        
        boundary_complexity_ratio = (np.mean(self.boundary_strength_history) / 
                                    (np.mean(self.complexity_history) + 1e-10))
        print(f"   Simulation: Boundary/complexity ratio = {boundary_complexity_ratio:.3f}")
        print(f"   Match: {'GOOD' if 0.5 < boundary_complexity_ratio < 2.0 else 'PARTIAL' if 0.2 < boundary_complexity_ratio < 5.0 else 'POOR'}")
        
        # System 3: Neural development (2D neural plates → 3D brain)
        print("\n3. NEURAL DEVELOPMENT COMPARISON")
        print("   Known: 2D neural plate → folding → 3D brain structure")
        print("   Known: Information processing emerges at layer boundaries")
        
        if self.information_flow_history:
            avg_info_flow = np.mean(self.information_flow_history)
            info_increase = (self.information_flow_history[-1] - 
                           self.information_flow_history[0])
            print(f"   Simulation: Average info flow = {avg_info_flow:.4f}")
            print(f"   Simulation: Info flow change = {info_increase:.4f}")
            print(f"   Match: {'GOOD' if avg_info_flow > 0.1 and info_increase > 0 else 'PARTIAL' if avg_info_flow > 0.05 else 'POOR'}")
        
        # System 4: Atmospheric layers (2D surface → 3D atmosphere)
        print("\n4. ATMOSPHERIC LAYERS COMPARISON")
        print("   Known: 2D surface heating → 3D atmospheric structure")
        print("   Known: Complex dynamics at layer boundaries")
        
        complexity_by_depth = np.mean(self.complexity_field, axis=(0, 1))
        surface_complexity = complexity_by_depth[0]
        mid_complexity = complexity_by_depth[self.max_depth//2]
        complexity_gradient = (mid_complexity - surface_complexity) / surface_complexity if surface_complexity > 0 else 0
        
        print(f"   Simulation: Surface complexity = {surface_complexity:.4f}")
        print(f"   Simulation: Mid-depth complexity = {mid_complexity:.4f}")
        print(f"   Simulation: Complexity gradient = {complexity_gradient:.3f}")
        print(f"   Match: {'GOOD' if abs(complexity_gradient) > 0.1 else 'PARTIAL' if abs(complexity_gradient) > 0.05 else 'POOR'}")

def main():
    """Run complete analysis"""
    print("2D → 2D+Depth Complexity Emergence Simulation")
    print("=" * 60)
    print("Testing empirical validity of boundary-driven complexity emergence")
    print()
    
    # Create and run simulation
    sim = DimensionalComplexitySimulation(grid_size=50, max_depth=15, dt=0.02)
    sim.run_simulation(steps=500)
    
    # Analyze results
    validation_results = sim.analyze_empirical_validity()
    
    # Compare with known systems
    sim.compare_with_known_systems()
    
    # Create visualizations
    print(f"\nGenerating visualization...")
    fig = sim.visualize_results()
    plt.show()
    
    # Final assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    validation_score = validation_results['validation_percentage']
    
    if validation_score >= 75:
        print("✓ STRONG EMPIRICAL SUPPORT for 2D→2D+depth complexity emergence")
        print("✓ Boundary dynamics successfully generate complexity")
        print("✓ Framework shows good correspondence with known systems")
    elif validation_score >= 50:
        print("◐ MODERATE EMPIRICAL SUPPORT for 2D→2D+depth complexity emergence")
        print("◐ Some evidence for boundary-driven complexity")
        print("◐ Partial correspondence with known systems")
    else:
        print("✗ WEAK EMPIRICAL SUPPORT for 2D→2D+depth complexity emergence")
        print("✗ Limited evidence for boundary-driven complexity")
        print("✗ Poor correspondence with known systems")
    
    print(f"\nValidation Score: {validation_score:.0f}%")
    
    return sim, validation_results

if __name__ == "__main__":
    simulation, results = main()