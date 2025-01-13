from manimlib import *

class Presentation(Scene):
    def construct(self):
        paper_title = TexText(r"Inferring Latent Dynamics Underlying Neural\\Population Activity via Neural Differential Equations").move_to(0.5 * UP)
        authors = TexText(r"Timothy D. Kim, Thomas Z. Luo, Jonathan W. Pillow, Carlos D. Brody",font_size=20).move_to(0.5 * DOWN)
        orchestra_img = ImageMobject("./figures/orchestra.jpg").move_to(UP)
        orchstra_cedit = TexText(r"Image Source: Berliner Philharmoniker", font_size=12).to_edge(DOWN+RIGHT, buff=0.2)
        orchestra_latent_text = TexText(r"Like musicians in an orchestra, neurons\\follow underlying \textbf{latent} dynamics.").move_to(2 * DOWN)
        latent_explanation = TexText(r"something that is hidden or not immediateley obvious", font_size=24).move_to(DOWN+LEFT)
        # explanation_path = Line([-1,-1,0],[1,1,0])
        # explanation_path = CubicBezier(get_smooth_cubic_bezier_handle_points([[0,0,0],[1,1,0],[3,5,0]]))
        arr = Arrow([-1,-0.5,0], [1,0.5,0])
        
        motivation_texts  = [TexText(r"This a mathematically dense paper."), TexText(r"The introduced models might be helpful\\for a lot of you in systems neuroscience."), TexText(r"We will build the modeling framework from the end.").move_to(2*DOWN)]
        
        
        # Title Slide
        self.play(LaggedStart(Write(paper_title), Write(authors), lag_ratio=1))
        self.play(LaggedStart(FadeOut(authors), FadeOut(paper_title),lag_ratio=0.5))
        
        # Slide with Orchestra Picture
        self.play(LaggedStart(FadeIn(orchestra_img), FadeIn(orchstra_cedit),lag_ratio=1))
        self.play(Write(orchestra_latent_text))
        self.play(LaggedStart(FadeOut(orchestra_img),FadeOut(orchstra_cedit), MoveAlongPath(orchestra_latent_text, Line(2 * DOWN, UP))))
        self.play(LaggedStart(orchestra_latent_text[50:56].animate.set_color(RED), ShowCreation(arr)))
        self.play(Write(latent_explanation))
        self.play(LaggedStart(FadeOut(orchestra_latent_text), FadeOut(latent_explanation), FadeOut(arr)))
        
        # Motivation Slide
        self.play(Write(motivation_texts[0]))
        self.play(LaggedStart(motivation_texts[0].animate.move_to(2*UP), Write(motivation_texts[1]),lag_ratio=0.5))
        self.play(Write(motivation_texts[2]))
        self.play(LaggedStart(*[FadeOut(mt) for mt in motivation_texts]))
        
        # Spike Recordings
        
        
        
        
        
        