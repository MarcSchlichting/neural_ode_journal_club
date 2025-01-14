from manimlib import *

class ArrowAlongBezierShape(Scene):
    def construct(self):
        # Define control points for the Bézier curve
        start = LEFT
        control1 = UP + LEFT
        control2 = UP + RIGHT
        end = RIGHT
        
        # Create the Bézier curve path
        bezier_curve = CubicBezier(start, control1, control2, end)
        
        # Create a line along the Bézier curve to indicate the path
        path = Line(start, end)  # This line won't be used for the arrow shape but is for visualization
        
        # Create an arrow object that follows the Bézier curve
        arrow = Arrow(start, end, buff=0.1)
        arrow.set_angle(bezier_curve.angle_at_point(0))  # Start with the angle of the curve at the start point
        self.add(arrow)
        
        # Animate the arrow following the Bézier curve path
        self.play(MoveAlongPath(arrow, bezier_curve), run_time=3)
