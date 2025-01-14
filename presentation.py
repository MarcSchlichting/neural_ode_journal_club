from manimlib import *

class Presentation(InteractiveScene):
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
        
        hh_vm = np.loadtxt("./data/hh_simulation_vm.txt")[:,::8]
        hh_t = np.loadtxt("./data/hh_simulation_t.txt")[:,::8]
        
        
        # # Title Slide
        # self.play(LaggedStart(Write(paper_title), Write(authors), lag_ratio=1))
        # self.play(LaggedStart(FadeOut(authors), FadeOut(paper_title),lag_ratio=0.5))
        
        # # Slide with Orchestra Picture
        # self.play(LaggedStart(FadeIn(orchestra_img), FadeIn(orchstra_cedit),lag_ratio=1))
        # self.play(Write(orchestra_latent_text))
        # self.play(LaggedStart(FadeOut(orchestra_img),FadeOut(orchstra_cedit), MoveAlongPath(orchestra_latent_text, Line(2 * DOWN, UP))))
        # self.play(LaggedStart(orchestra_latent_text[50:56].animate.set_color(RED), ShowCreation(arr)))
        # self.play(Write(latent_explanation))
        # self.play(LaggedStart(FadeOut(orchestra_latent_text), FadeOut(latent_explanation), FadeOut(arr)))
        
        # # Motivation Slide
        # self.play(Write(motivation_texts[0]))
        # self.play(LaggedStart(motivation_texts[0].animate.move_to(2*UP), Write(motivation_texts[1]),lag_ratio=0.5))
        # self.play(Write(motivation_texts[2]))
        # self.play(LaggedStart(*[FadeOut(mt) for mt in motivation_texts]))
        
        # # Spike Recordings
        # ax = [Axes([0,hh_t.max()],[hh_vm.min(), hh_vm.max()],height=0.4,width=6).to_edge(UP,buff=1).to_edge(LEFT,buff=1)]
        # # ax[0].get_y_axis_label("V_0",edge=[0,0,0])
        # xs = [np.stack([hh_t[0],hh_vm[0]],axis=1)]
        # for i in range(1,hh_vm.shape[0]):
        #     ax.append(Axes([0,hh_t.max()],[hh_vm.min(), hh_vm.max()],height=0.4,width=6).next_to(ax[-1],DOWN))
        #     xs.append(np.stack([hh_t[i],hh_vm[i]],axis=1))
        # self.play(LaggedStart([Write(TexText("Neuronal Voltage Recordings",font_size=20).next_to(ax[0],UP))]+[Write(Tex(fr"V_{{{i+1}}}",font_size=20).next_to(a,LEFT)) for i,a in enumerate(ax)]))
        # self.play(*[ShowCreation(VMobject().set_points(ax[i].c2p(*xs[i].T)).set_stroke(BLUE_B, 2),rate_func=linear,run_time=10) for i in range(len(xs))])
        
        # Poisson Process
        # poisson_process_title = TexText("Nonhomogeneous Poisson Processes").to_edge(UP+LEFT)
        # poisson_tex_one = Tex(R"""
        #                   t_{1}, t_{2}, t_{3},\ldots, t_{N} \sim \mathrm{PoissonProcess}(\lambda(t))
        #                   """)
        # poisson_tex_full = Tex(R"""
        #                   \begin{matrix}
        #                   t_{1}^{(1)}, &  t_{2}^{(1)}, & t_{3}^{(1)}, & \ldots, &  t_{N_1}^{(1)} \\
        #                   t_{1}^{(2)}, & t_{2}^{(2)}, & t_{3}^{(2)}, & \ldots, & t_{N_2}^{(2)} \\
        #                   t_{1}^{(3)}, & t_{2}^{(3)}, & t_{3}^{(3)}, & \ldots, & t_{N_3}^{(3)} \\
        #                     & &   \vdots \\
        #                   t_{1}^{(M)}, & t_{2}^{(M)}, & t_{3}^{(M)}, & \ldots, & t_{N_M}^{(M)}
        #                   \end{matrix} \sim \begin{bmatrix} \mathrm{PoissonProcess}(\lambda_1(t))\vphantom{0^{(0)}_0} \\ 
        #                    \mathrm{PoissonProcess}(\lambda_2(t))\vphantom{0^{(0)}_0} \\
        #                    \mathrm{PoissonProcess}(\lambda_3(t))\vphantom{0^{(0)}_0} \\
        #                        \vdots  \\
        #                     \mathrm{PoissonProcess}(\lambda_M(t)) \vphantom{0^{(0)}_0}
        #                    \end{bmatrix}
        #                   """, font_size=32)
        # spike_times = TexText("Spike Times",font_size=24).next_to(np.array([-2,-2.5,0]),LEFT)
        # spike_time_curves = [CubicBezier(np.array([-2,-2.5,0]), np.array([2,-2.5,0]),poisson_tex_one[i].get_bottom() + DOWN, poisson_tex_one[i].get_bottom() +0.2 *DOWN).set_stroke(WHITE,2) for i in [0,3,6,13]]
        # firing_rate = TexText("Firing Rate", font_size=24).next_to(np.array([3,2,0]),RIGHT)
        # firing_rate_curve = CubicBezier(np.array([3,2,0]), np.array([1,2,0]), poisson_tex_one[31].get_top()+0.8*UP,poisson_tex_one[31].get_top()+0.2*UP).set_stroke(WHITE,2)
        # num_neurons_group = Group()
        # num_neurons_group.add(Arrow(poisson_tex_full.get_corner(UP+LEFT)+0.5*LEFT,poisson_tex_full.get_corner(DOWN+LEFT)+0.5*LEFT, buff=0, stroke_width=0.5))
        # num_neurons_group.add(TexText("Number of Neurons",font_size=24).rotate(np.pi/2).next_to(num_neurons_group[0],LEFT))
        
        
        # self.play(Write(poisson_process_title))
        # self.play(Write(poisson_tex_one))
        # self.play(LaggedStart(Write(spike_times),*[ShowCreation(stc) for stc in spike_time_curves]))
        # self.play(LaggedStart(Write(firing_rate),ShowCreation(firing_rate_curve)))
        # self.play(LaggedStart(Uncreate(spike_times),
        #                       *[Uncreate(stc) for stc in spike_time_curves],
        #                       Uncreate(firing_rate),
        #                       Uncreate(firing_rate_curve),
        #                       Transform(poisson_tex_one, poisson_tex_full)))
        # self.play(ShowCreation(num_neurons_group))
        # self.play(LaggedStart(Uncreate(num_neurons_group),
        #           Uncreate(poisson_tex_one)))
        
        # # From Firing Rate to Latent Variables
        # fr_to_lv_title = TexText("From Firing Rates to Latent Variables").to_edge(UP+LEFT)
        # exp_aff_map_short = Tex(R"""
        #                   \mathbf{\lambda} = \mathrm{exp}\left(\mathbf{A}\mathbf{z}(t) + \mathbf{b}\right)
        #                   """)
        # exp_aff_map_long = Tex(R"""
        #                   \begin{bmatrix}
        #                   \lambda^{(1)}(t)\\
        #                   \lambda^{(2)}(t)\\
        #                   \lambda^{(3)}(t)\\
        #                   \vdots \\
        #                   \lambda^{(M)}(t)\\    
        #                   \end{bmatrix} = \mathrm{exp}\left(\begin{bmatrix}a_{11} & a_{12} & a_{13} & \ldots & a_{1K}\\
        #                       a_{21} & a_{22} & a_{23} &\ldots & a_{2K}\\
        #                           a_{31} & a_{32} & a_{33} &\ldots & a_{3K}\\
        #                          \vdots & \vdots & \vdots & \ddots & \vdots \\
        #                       a_{M1} & a_{M2} & a_{M3} & \ldots & a_{MK}
        #                       \end{bmatrix}\begin{bmatrix}z_1(t) \\ z_2(t) \\ z_3(t) \\ \vdots \\ z_K(t)\end{bmatrix} + 
        #                       \begin{bmatrix}b_1 \\ b_2 \\ b_3 \\ \vdots \\ b_K\end{bmatrix}\right)
        #                   """, font_size=32)
        # dim_lambda = Tex(R"\mathrm{Firing~Rates}\in\mathbb{R}^{M}",font_size=24).next_to(np.array([-4.5,-2.5,0]),RIGHT)
        # dim_lambda_curve = CubicBezier(np.array([-4.5,-2.5,0]),np.array([-4.5,-2.5,0])+LEFT, exp_aff_map_long[33].get_bottom()+0.8 * DOWN, exp_aff_map_long[33].get_bottom()+0.2 * DOWN).set_stroke(WHITE,2)
        # dim_A = Tex(R"\mathrm{Loading~Matrix}\in\mathbb{R}^{M\times K}",font_size=24).next_to(np.array([-1,2.5,0]),RIGHT)
        # dim_A_curve = CubicBezier(np.array([-1,2.5,0]),np.array([-1,2.5,0])+2 * LEFT,exp_aff_map_long.get_top()+0.8*UP,exp_aff_map_long.get_top()+0.2*UP).set_stroke(WHITE,2)
        # dim_z = Tex(R"\mathrm{Latent~Variables}\in\mathbb{R}^{K}",font_size=24).next_to(np.array([3.2,-2.75,0]),LEFT)
        # dim_z_curve = CubicBezier(np.array([3.2,-2.75,0]),np.array([3.2,-2.75,0])+RIGHT,np.array([2.8,-1.2,0])+DOWN,np.array([2.8,-1.2,0])).set_stroke(WHITE,2)
        # dim_b = Tex(R"\mathrm{Bias~Vector}\in\mathbb{R}^{K}",font_size=24).next_to(np.array([4,2,0]),RIGHT)
        # dim_b_curve = CubicBezier(np.array([4,2,0]),np.array([4,2,0])+LEFT,np.array([4.2,1.2,0])+0.5 * UP, np.array([4.2,1.2,0])).set_stroke(WHITE,2)
        # exp_expl = Tex(R"\mathrm{Firing~Rates}\in\mathbb{R}^+",font_size=24).next_to(np.array([-3,2,0]),LEFT)
        # exp_expl_curve = CubicBezier(np.array([-3,2,0]), np.array([-3,2,0])+RIGHT, np.array([-2.9,0.2,0])+UP, np.array([-2.9,0.2,0])).set_stroke(WHITE,2)
        
        # # self.play(Transform(poisson_process_title,fr_to_lv_title))
        # self.play(Write(exp_aff_map_short))
        # self.play(Transform(exp_aff_map_short,exp_aff_map_long),run_time=2)
        # self.play(LaggedStart(Write(dim_lambda),ShowCreation(dim_lambda_curve)))
        # self.play(LaggedStart(Write(dim_z[::-1]), ShowCreation(dim_z_curve)))
        # self.play(LaggedStart(Write(dim_A),ShowCreation(dim_A_curve)))
        # self.play(LaggedStart(Write(dim_b),ShowCreation(dim_b_curve)))
        # self.play(LaggedStart(Write(exp_expl[::-1]), ShowCreation(exp_expl_curve)))
        
        # self.play(LaggedStart(Uncreate(dim_lambda),
        #                       Uncreate(dim_lambda_curve),
        #                       Uncreate(dim_z[::-1]),
        #                       Uncreate(dim_z_curve),
        #                       Uncreate(dim_A),
        #                       Uncreate(dim_A_curve),
        #                       Uncreate(dim_z),
        #                       Uncreate(dim_z_curve),
        #                       Uncreate(dim_b),
        #                       Uncreate(dim_b_curve),
        #                       Uncreate(exp_expl[::-1]),
        #                       Uncreate(exp_expl_curve),
        #                       Uncreate(exp_aff_map_short)))
        
        
        # Latent Dynamics
        latent_dynamics_title = TexText("Latent Dynamics").to_edge(UP+LEFT)
        latent_dynamic_intro_text = TexText(r"Given a latent state $\mathbf{z}$, what\\is the rate of change of the latent variables?")
        latent_ode = Tex(R"\dfrac{\mathrm{d}\mathbf{z}}{\mathrm{d}t} = f(\mathbf{z},t,u)")
        latent_change = TexText("Latent Variables Rate of Change",font_size=24).next_to(np.array([-2,2,0]),LEFT)
        latent_change_curve = CubicBezier(np.array([-2,2,0]),np.array([-2,2,0])+2 *RIGHT,np.array([-1.2,0.6,0])+UP,np.array([-1.2,0.6,0])).set_stroke(WHITE,2)
        f_exp = TexText("An Arbitrary Non-Linear Function",font_size=24).next_to(np.array([-2,-1.5,0]),LEFT)
        f_exp_curve = CubicBezier(np.array([-2,-1.5,0]),np.array([-2,-1.5,0])+RIGHT,np.array([-0.15,-0.4,0])+DOWN,np.array([-0.15,-0.4,0])).set_stroke(WHITE,2)
        z_exp = TexText(r"\begin{flushleft}Current Values\\of Latent Variables\end{flushleft}",font_size=24).next_to(np.array([1,2.5,0]),RIGHT)
        z_exp_curve = CubicBezier(np.array([1,2.5,0]),np.array([1,2.5,0])+LEFT,np.array([0.3,0.2,0])+UP, np.array([0.3,0.2,0])).set_stroke(WHITE,2)
        t_exp = TexText("Time",font_size=24).next_to(np.array([1,-2,0]),RIGHT)
        t_exp_curve = CubicBezier(np.array([1,-2,0]),np.array([1,-2,0])+LEFT, np.array([0.75,-0.25,0])+DOWN, np.array([0.75,-0.25,0])).set_stroke(WHITE,2)
        
        self.add(latent_dynamics_title)
        # self.add(latent_dynamic_intro_text)
        self.add(latent_ode)
        self.add(NumberPlane())
        self.add(latent_change)
        self.add(latent_change_curve)
        self.add(f_exp)
        self.add(f_exp_curve)
        self.add(z_exp)
        self.add(z_exp_curve)
        self.add(t_exp)
        self.add(t_exp_curve)
 
        
        
        